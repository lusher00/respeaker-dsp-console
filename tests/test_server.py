"""Engine fed synthetic blocks in-process, plus the HTTP daemon end to end
with tests/fake_capture.py standing in for arecord. Needs no audio hardware,
no amixer and no model."""
import json
import math
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.environ.setdefault("RESPEAKER_STATE_FILE", os.path.join(tempfile.mkdtemp(), "s.json"))
import server  # noqa: E402
from doa import fractional_delay  # noqa: E402

FS = server.RATE


def burst_blocks(bearing, on_s=0.4, off_s=0.6, cycles=3, level=0.1, block=1024, seed=5):
    rng = np.random.default_rng(seed)
    tau = 0.058 * math.sin(math.radians(bearing)) / 343.0
    period = int((on_s + off_s) * FS)
    n_on = int(on_s * FS)
    for _ in range(cycles):
        src = np.zeros(period)
        src[:n_on] = rng.standard_normal(n_on) * level
        L = fractional_delay(src, tau * FS) + rng.standard_normal(period) * 1e-3
        R = src + rng.standard_normal(period) * 1e-3
        x = np.stack((L, R), 1).astype(np.float32)
        for i in range(0, period, block):
            yield x[i:i + block]


class FakeYam:
    """Stands in for classify.YamNet."""
    ready = True
    runtime = "fake"
    error = None
    last_ms = 1.0

    def __init__(self, label="Clapping"):
        self.label = label
        self.calls = 0

    def scores(self, wave):
        self.calls += 1
        assert len(wave) == 15600
        return np.array([0.9])

    def top(self, scores, k=5, ignore=()):
        return [{"label": self.label, "score": 0.9}]


class EngineTest(unittest.TestCase):
    def engine(self, **over):
        s = server.defaults()
        s.update(over)
        server.apply_codec = lambda s: {}
        return server.Engine(s, FakeYam())

    def test_bursts_become_events_with_bearing(self):
        e = self.engine()
        for b in burst_blocks(-25):
            e.process_block(b)
        st = e.state()
        self.assertTrue(st["track"]["tracking"])
        self.assertAlmostEqual(st["track"]["bearing_deg"], -25, delta=1.5)
        ev = e.events_since(0)
        self.assertGreaterEqual(len(ev), 2)
        for x in ev:
            self.assertAlmostEqual(x["bearing_deg"], -25, delta=1.5)
            self.assertAlmostEqual(x["duration_s"], 0.4, delta=0.1)

    def test_swap_channels_mirrors_bearing(self):
        e = self.engine(swap_channels=True)
        for b in burst_blocks(30, cycles=2):
            e.process_block(b)
        self.assertAlmostEqual(e.state()["track"]["bearing_deg"], -30, delta=1.5)

    def test_quiet_room_is_not_active(self):
        e = self.engine()
        rng = np.random.default_rng(1)
        for _ in range(100):
            e.process_block((rng.standard_normal((1024, 2)) * 1e-3).astype(np.float32))
        self.assertFalse(e.state()["frame"]["active"])
        self.assertEqual(e.events_since(0), [])

    def test_classifier_labels_attach_to_event(self):
        e = self.engine()
        # The first burst is absorbed: the noise floor starts at the first
        # frame's level, which is mid-burst. Bursts 2 and 3 become events.
        gen = burst_blocks(10, cycles=3)
        for i, b in enumerate(gen):
            e.process_block(b)
            if i == 58:      # t = 1.36 s, late in burst 2
                t = e.t_audio
                e._attach_labels(t - 0.975, t, [{"label": "Clapping", "score": 0.8}])
        ev = e.events_since(0)
        self.assertEqual(ev[0]["label"], "Clapping")
        self.assertIsNone(ev[1]["label"])

    def test_calibration_measures_spacing(self):
        e = self.engine(spacing_mm=40.0)        # deliberately wrong
        e.start_calibration(2.0)
        for b in burst_blocks(90, cycles=3):
            e.process_block(b)
        r = e.calibration_status()["result"]
        self.assertAlmostEqual(r["spacing_mm"], 58, delta=2)
        self.assertEqual(r["source_side"], "right")

    def test_settings_change_rebuilds(self):
        e = self.engine()
        e.update_settings({"nfft": 4096})
        for b in burst_blocks(0, cycles=1):
            e.process_block(b)
        self.assertEqual(e.nfft, 4096)
        e.update_settings({"nfft": 1024})        # next block is 2048 long, nfft 1024
        e.process_block(np.zeros((2048, 2), np.float32))
        self.assertEqual(e.nfft, 1024)

    def test_validate(self):
        cur = server.defaults()
        ok, err = server.validate({"spacing_mm": "61.5", "agc": "on", "nfft": "1024",
                                   "ignore_classes": "Silence, Music"}, cur)
        self.assertEqual(err, {})
        self.assertEqual(ok["spacing_mm"], 61.5)
        self.assertIs(ok["agc"], True)
        self.assertEqual(ok["nfft"], 1024)
        self.assertEqual(ok["ignore_classes"], ["Silence", "Music"])
        _, err = server.validate({"spacing_mm": 1, "bogus": 1, "nfft": 999}, cur)
        self.assertEqual(set(err), {"spacing_mm", "bogus", "nfft"})
        _, err = server.validate({"band_lo_hz": 3000, "band_hi_hz": 3050}, cur)
        self.assertIn("band_hi_hz", err)


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class HttpTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.port = free_port()
        cls.tmp = tempfile.mkdtemp()
        env = dict(os.environ,
                   RESPEAKER_HTTP_PORT=str(cls.port),
                   RESPEAKER_STATE_FILE=os.path.join(cls.tmp, "settings.json"),
                   RESPEAKER_CAPTURE_CMD=f"{sys.executable} {HERE}/fake_capture.py --bearing 40",
                   RESPEAKER_MODEL=os.path.join(cls.tmp, "missing.tflite"),
                   PATH=os.environ.get("PATH", ""))
        cls.proc = subprocess.Popen([sys.executable, os.path.join(ROOT, "server.py")], env=env,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.time() + 10
        while time.time() < deadline:
            try:
                cls.get("/health")
                break
            except OSError:
                time.sleep(0.2)

    @classmethod
    def tearDownClass(cls):
        cls.proc.terminate()
        cls.proc.wait(5)

    @classmethod
    def get(cls, path):
        with urllib.request.urlopen(f"http://127.0.0.1:{cls.port}{path}", timeout=3) as r:
            return json.loads(r.read()), r.headers

    def post(self, path, body):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}",
                                     data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=3) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_tracks_the_fake_source(self):
        deadline = time.time() + 8
        st = None
        while time.time() < deadline:
            st, hdr = self.get("/state")
            if st["track"]["tracking"]:
                break
            time.sleep(0.2)
        self.assertEqual(hdr["Access-Control-Allow-Origin"], "*")
        self.assertTrue(st["running"])
        self.assertAlmostEqual(st["track"]["bearing_deg"], 40, delta=2)

    def test_health_reports_missing_model_without_failing(self):
        h, _ = self.get("/health")
        self.assertFalse(h["classifier"]["ready"])
        self.assertTrue(h["classifier"]["error"])      # no runtime or no model
        self.assertIsNone(h["capture_error"])

    def test_spectrum_and_events_shape(self):
        time.sleep(1.5)
        sp, _ = self.get("/spectrum")
        self.assertEqual(len(sp["freq"]), len(sp["L"]))
        self.assertIn("response", sp)
        ev, _ = self.get("/events?since=0")
        self.assertIsInstance(ev["events"], list)

    def test_config_roundtrip_and_persist(self):
        code, r = self.post("/config", {"snr_db": 9.5})
        self.assertEqual(code, 200)
        self.assertTrue(r["ok"])
        saved = json.load(open(os.path.join(self.tmp, "settings.json")))
        self.assertEqual(saved["snr_db"], 9.5)
        code, r = self.post("/config", {"snr_db": 999})
        self.assertEqual(code, 400)
        self.assertIn("snr_db", r["errors"])

    def test_bad_json(self):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/config", data=b"{nope",
                                     method="POST")
        with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen(req, timeout=3)
        self.assertEqual(cm.exception.code, 400)


if __name__ == "__main__":
    unittest.main()
