#!/usr/bin/env python3
"""ReSpeaker 2-Mics HAT v2 sound daemon: direction, activity and sound ID.

Reads the HAT's two microphones continuously and answers, over HTTP:

  * where is the sound coming from     bearing, -90..+90 deg, + = right
  * is anything happening              band level against a tracked noise floor
  * what is it                         YAMNet top classes (AudioSet labels)
  * what happened recently             one event per sound, with bearing and class

Pipeline, all in one capture thread except the classifier:

  arecord (48 kHz, S16, 2 ch)
    -> frames of nfft samples, hop nfft/2
    -> FFT both channels
         -> band level, noise floor, activity
         -> GCC-PHAT delay -> bearing          (doa.PairDOA)
         -> bearing histogram -> track        (doa.BearingTracker)
         -> event segmentation
         -> averaged spectrum for the page
    -> L+R mono, decimate x3 to 16 kHz -> ring
  classifier thread: every hop_s, last 0.975 s of the ring -> YAMNet

Capture is a child arecord process rather than an ALSA binding: nothing to
compile, and the tests swap in a fake by setting RESPEAKER_CAPTURE_CMD.

HTTP, all JSON and CORS-open so the bot dashboard and the hailo-tracker page
can read it from another host:

  GET  /state       bearing, track, levels, current sound — small, poll at 10 Hz
  GET  /spectrum    averaged L/R spectrum (log-spaced) and the delay response
  GET  /events      recent sound events; ?since=<id> for only newer ones
  GET  /health      counters, timing, capture and classifier status
  GET  /config      runtime settings and their limits
  POST /config      partial settings update, validated; persisted
  POST /calibrate   {"seconds": 5} measure mic spacing from an end-on source
  GET  /calibrate   calibration progress / result
  POST /track/reset clear the bearing track
  GET  /            the console page

Env (see config/respeaker-console.env.example): RESPEAKER_DEVICE,
RESPEAKER_CARD, RESPEAKER_RATE, RESPEAKER_HTTP_PORT, RESPEAKER_MODEL,
RESPEAKER_THREADS, RESPEAKER_STATE_FILE, RESPEAKER_CAPTURE_CMD.
"""

import json
import math
import os
import shlex
import subprocess
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from classify import SAMPLE_RATE as CLS_RATE, WINDOW as CLS_WINDOW, YamNet
from doa import BearingTracker, Decimator, NoiseFloor, PairDOA, Ring, spacing_from_endfire

ROOT = Path(__file__).resolve().parent
CARD = os.environ.get("RESPEAKER_CARD", "seeed2micvoicec")
DEVICE = os.environ.get("RESPEAKER_DEVICE", f"hw:CARD={CARD},DEV=0")
RATE = int(os.environ.get("RESPEAKER_RATE", "48000"))
HTTP_PORT = int(os.environ.get("RESPEAKER_HTTP_PORT", "8082"))
MODEL = os.environ.get("RESPEAKER_MODEL", str(ROOT / "models" / "yamnet.tflite"))
LABELS = str(ROOT / "models" / "yamnet_class_map.csv")
THREADS = int(os.environ.get("RESPEAKER_THREADS", "2"))
CAPTURE_CMD = os.environ.get("RESPEAKER_CAPTURE_CMD", "")
# systemd's StateDirectory= sets STATE_DIRECTORY; outside systemd, next to the code.
STATE_FILE = os.environ.get(
    "RESPEAKER_STATE_FILE",
    os.path.join(os.environ.get("STATE_DIRECTORY", str(ROOT)), "settings.json"))

if RATE % CLS_RATE:
    sys.exit(f"RESPEAKER_RATE must be a multiple of {CLS_RATE} (got {RATE})")


def log(msg):
    print(msg, flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Runtime settings: name -> (default, type, min, max) or (default, choices)
SPEC = {
    # geometry
    "spacing_mm":          (58.0, float, 10.0, 300.0),
    "speed_of_sound":      (343.0, float, 300.0, 380.0),
    "swap_channels":       (False, bool),
    # direction finding
    "nfft":                (2048, (1024, 2048, 4096)),
    "band_lo_hz":          (300.0, float, 50.0, 10000.0),
    "band_hi_hz":          (4000.0, float, 200.0, 20000.0),
    "snr_db":              (8.0, float, 0.0, 40.0),
    "coherence_min":       (0.25, float, 0.0, 1.0),
    "floor_rise_db_s":     (3.0, float, 0.1, 60.0),
    "track_half_life_s":   (0.75, float, 0.05, 10.0),
    "track_min_strength":  (1.5, float, 0.0, 50.0),
    "event_hangover_s":    (0.3, float, 0.05, 5.0),
    # identification
    "classify_enabled":    (True, bool),
    "classify_only_active": (True, bool),
    "classify_hop_s":      (0.5, float, 0.1, 5.0),
    "class_min_score":     (0.15, float, 0.0, 1.0),
    "ignore_classes":      (["Silence"], list),
    # codec (TLV320AIC3104)
    "pga_db":              (16.0, float, 0.0, 59.5),
    "agc":                 (False, bool),
    "adc_hpf":             ("0.0045xFs", ("Disabled", "0.0045xFs", "0.0125xFs", "0.025xFs")),
}
CODEC_KEYS = ("pga_db", "agc", "adc_hpf")
DOA_KEYS = ("spacing_mm", "speed_of_sound", "nfft", "band_lo_hz", "band_hi_hz")


def defaults():
    return {k: (list(v[0]) if isinstance(v[0], list) else v[0]) for k, v in SPEC.items()}


def validate(incoming, current):
    """Returns (clean_changes, errors). Unknown keys are errors, not ignored."""
    out, errors = {}, {}
    for k, v in (incoming or {}).items():
        spec = SPEC.get(k)
        if spec is None:
            errors[k] = "unknown setting"
            continue
        try:
            if len(spec) == 2 and isinstance(spec[1], tuple):
                if isinstance(spec[0], int) and not isinstance(spec[0], bool):
                    v = int(v)
                if v not in spec[1]:
                    raise ValueError(f"must be one of {list(spec[1])}")
            elif spec[1] is bool:
                if isinstance(v, str):
                    v = v.strip().lower() in ("1", "true", "on", "yes")
                v = bool(v)
            elif spec[1] is list:
                if isinstance(v, str):
                    v = [s.strip() for s in v.split(",") if s.strip()]
                v = [str(s) for s in v]
            else:
                v = float(v)
                if not math.isfinite(v) or not (spec[2] <= v <= spec[3]):
                    raise ValueError(f"must be {spec[2]}..{spec[3]}")
        except (TypeError, ValueError) as e:
            errors[k] = str(e)
            continue
        out[k] = v
    merged = dict(current, **out)
    if merged["band_hi_hz"] <= merged["band_lo_hz"] + 100:
        errors["band_hi_hz"] = "must be at least 100 Hz above band_lo_hz"
    if merged["band_hi_hz"] > RATE / 2 * 0.95:
        errors["band_hi_hz"] = f"must be below {RATE / 2 * 0.95:.0f} Hz at {RATE} Hz"
    return (out if not errors else {}), errors


def load_settings():
    s = defaults()
    try:
        saved = json.loads(Path(STATE_FILE).read_text())
        clean, errors = validate({k: v for k, v in saved.items() if k in SPEC}, s)
        if errors:
            log(f"[settings] ignored invalid saved values: {errors}")
        s.update(clean)
    except FileNotFoundError:
        pass
    except Exception as e:                          # noqa: BLE001
        log(f"[settings] could not read {STATE_FILE}: {e}")
    return s


def save_settings(s):
    try:
        p = Path(STATE_FILE)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(s, indent=2))
        tmp.replace(p)
        return None
    except OSError as e:
        return str(e)


# ─────────────────────────────────────────────────────────────────────────────
def amixer(*args):
    """One amixer call against the HAT by card name. The card's index moves
    with whatever else enumerates first (HDMI on a Pi 4/5), its name does not."""
    try:
        r = subprocess.run(["amixer", "-q", "-c", CARD, "sset", *args],
                           capture_output=True, text=True, timeout=3)
        return None if r.returncode == 0 else (r.stderr.strip() or f"exit {r.returncode}")
    except (OSError, subprocess.SubprocessError) as e:
        return str(e)


def apply_codec(s):
    """TLV320AIC3104 capture controls. PGA is 0..119 in 0.5 dB steps.
    AGC defaults off: it gain-rides each channel on its own, which leaves the
    bearing alone (PHAT ignores magnitude) but makes the level/noise-floor
    gate chase the AGC instead of the room."""
    errs = {}
    steps = int(round(s["pga_db"] * 2))
    for name, args in (("pga_db", ("PGA", f"{steps},{steps}")),
                       ("agc", ("AGC", "on" if s["agc"] else "off")),
                       ("adc_hpf", ("ADC HPF Cut-off", f"{s['adc_hpf']},{s['adc_hpf']}"))):
        e = amixer(*args)
        if e:
            errs[name] = e
    return errs


# ─────────────────────────────────────────────────────────────────────────────
class Engine:
    def __init__(self, settings, classifier):
        self.s = settings
        self.lock = threading.Lock()
        self.cls = classifier
        self.started = time.monotonic()
        self.h = dict(frames=0, active_frames=0, capture_starts=0, capture_errors=0,
                      overruns=0, clipped=0, bytes=0, events=0, classify_runs=0,
                      classify_skipped=0, classify_errors=0)
        self.proc_ms = deque(maxlen=200)
        self.capture_error = None
        self.codec_errors = {}
        self.last_stderr = deque(maxlen=5)
        self._build()
        self.floor = NoiseFloor(settings["floor_rise_db_s"])
        self.tracker = BearingTracker(settings["track_half_life_s"], 3.0,
                                      settings["track_min_strength"])
        self.decim = Decimator(RATE, RATE // CLS_RATE)
        self.ring = Ring(CLS_RATE * 2)
        self.t_audio = 0.0                  # seconds of audio processed
        self.last_active_t = -1e9
        self.frame = {}
        self.levels = [None, None]
        self.band_db = None
        self.peak = [None, None]
        self.spec_avg = None
        self.resp_avg = None
        self.sound = {"top": [], "t": None}
        self.events = deque(maxlen=100)
        self.cur_event = None
        self.next_event_id = 1
        self.calib = {"running": False}
        self._stop = threading.Event()
        self.proc = None

    # -- (re)build anything that depends on geometry/band/nfft ---------------
    def _build(self):
        s = self.s
        self.doa = PairDOA(RATE, s["nfft"], s["spacing_mm"] / 1000.0,
                           (s["band_lo_hz"], s["band_hi_hz"]), s["speed_of_sound"])
        self.nfft = s["nfft"]
        self.hop = self.nfft // 2
        self.buf = np.zeros((self.nfft, 2), np.float32)
        self.buf_fill = 0
        # log-spaced display groups: max of each group of FFT bins
        idx = np.unique(np.round(np.geomspace(2, self.nfft // 2, 220)).astype(int))
        self.disp_idx = idx
        self.disp_freq = (idx * RATE / self.nfft).round(1)
        self.spec_avg = None
        self.resp_avg = None
        self.rebuild = False

    def update_settings(self, changes):
        with self.lock:
            self.s.update(changes)
            if any(k in changes for k in DOA_KEYS):
                self.rebuild = True
            self.floor.rise = self.s["floor_rise_db_s"]
            self.tracker.half_life_s = self.s["track_half_life_s"]
            self.tracker.min_strength = self.s["track_min_strength"]
        if any(k in changes for k in CODEC_KEYS):
            self.codec_errors = apply_codec(self.s)

    # -- capture ---------------------------------------------------------------
    def capture_cmd(self):
        if CAPTURE_CMD:
            return shlex.split(CAPTURE_CMD) + ["--rate", str(RATE)]
        # --buffer-time 500 ms: room for a slow classifier frame on a Pi Zero
        # without an overrun; arecord prints "overrun!!!" when it does happen.
        return ["arecord", "-q", "-D", DEVICE, "-f", "S16_LE", "-r", str(RATE),
                "-c", "2", "-t", "raw", "--buffer-time=500000"]

    def _stderr_reader(self, proc):
        for line in proc.stderr:
            line = line.decode(errors="replace").strip()
            if not line:
                continue
            if "overrun" in line:
                self.h["overruns"] += 1
            else:
                self.last_stderr.append(line)

    def run_capture(self):
        self.codec_errors = apply_codec(self.s)
        while not self._stop.is_set():
            try:
                self.h["capture_starts"] += 1
                self.proc = subprocess.Popen(self.capture_cmd(), stdout=subprocess.PIPE,
                                             stderr=subprocess.PIPE, bufsize=0)
                threading.Thread(target=self._stderr_reader, args=(self.proc,),
                                 daemon=True).start()
                self.capture_error = None
                self._read_loop(self.proc.stdout)
                rc = self.proc.wait(timeout=2)
                if not self._stop.is_set():
                    tail = "; ".join(self.last_stderr) or "no output"
                    self.capture_error = f"capture exited ({rc}): {tail}"
            except Exception as e:                  # noqa: BLE001
                self.capture_error = f"capture: {e}"
            finally:
                if self.proc and self.proc.poll() is None:
                    self.proc.kill()
            if not self._stop.is_set():
                self.h["capture_errors"] += 1
                log(f"[capture] {self.capture_error}; retry in 2 s")
                self._stop.wait(2)

    def _read_loop(self, f):
        pending = b""
        while not self._stop.is_set():
            need = self.hop * 4 - len(pending)
            chunk = f.read(need)
            if not chunk:
                return
            self.h["bytes"] += len(chunk)
            pending += chunk
            if len(pending) < self.hop * 4:
                continue
            block = np.frombuffer(pending, "<i2").reshape(-1, 2).astype(np.float32) / 32768.0
            pending = b""
            t0 = time.perf_counter()
            self.process_block(block)
            self.proc_ms.append((time.perf_counter() - t0) * 1000)

    # -- DSP, one hop --------------------------------------------------------
    def process_block(self, block):
        with self.lock:
            if self.rebuild:
                self._build()
            s = self.s
            if s["swap_channels"]:
                block = block[:, ::-1]
            n = len(block)
            dt = n / RATE
            self.t_audio += dt
            t = self.t_audio

            pk = np.max(np.abs(block), axis=0)
            if np.any(pk >= 0.999):
                self.h["clipped"] += 1
            ms = np.mean(block * block, axis=0)
            self.levels = [round(10 * math.log10(max(float(m), 1e-12)), 1) for m in ms]
            self.peak = [round(20 * math.log10(max(float(p), 1e-6)), 1) for p in pk]

            # classifier input: L+R mono at 16 kHz
            self.ring.push(self.decim.process(block.mean(axis=1)))

            # sliding analysis frame
            blk = block[-self.nfft:]         # a block from before an nfft change can be longer
            self.buf = np.roll(self.buf, -len(blk), axis=0)
            self.buf[-len(blk):] = blk
            self.buf_fill = min(self.nfft, self.buf_fill + len(blk))
            if self.buf_fill < self.nfft:
                return
            self.h["frames"] += 1

            XL, XR = self.doa.spectra(self.buf)
            band_db = self.doa.band_level_db(XL, XR)
            floor = self.floor.update(band_db, dt)
            m = self.doa.measure(XL, XR)
            snr = band_db - floor
            active = (snr >= s["snr_db"] and m["coherence"] >= s["coherence_min"]
                      and m["in_range"])

            self.tracker.decay(dt)
            if active:
                self.h["active_frames"] += 1
                self.last_active_t = t
                self.tracker.add(m["bearing_deg"], m["coherence"], t)
            self._events(t, active, m, band_db)
            self._calibrate(m, snr)

            self.band_db = round(band_db, 1)
            self.frame = {"bearing_deg": round(m["bearing_deg"], 1),
                          "coherence": round(m["coherence"], 3),
                          "tau_us": round(m["tau_s"] * 1e6, 1),
                          "in_range": m["in_range"], "active": active,
                          "snr_db": round(snr, 1), "floor_db": round(floor, 1)}

            p = np.abs(np.stack((XL, XR))) ** 2 * (2.0 / (self.nfft * self.doa.win_power))
            sdb = 10 * np.log10(np.maximum(np.maximum.reduceat(p, self.disp_idx, axis=1), 1e-12))
            a = 0.3
            self.spec_avg = sdb if self.spec_avg is None or self.spec_avg.shape != sdb.shape \
                else (1 - a) * self.spec_avg + a * sdb
            r = m["response"]
            self.resp_avg = r if self.resp_avg is None or self.resp_avg.shape != r.shape \
                else 0.6 * self.resp_avg + 0.4 * r

    def _events(self, t, active, m, level):
        ev = self.cur_event
        if active:
            if ev is None:
                ev = self.cur_event = {"t0": t, "wall": time.time(), "b": [], "w": [],
                                       "peak_db": level, "labels": {}}
            ev["b"].append(m["bearing_deg"])
            ev["w"].append(m["coherence"])
            ev["peak_db"] = max(ev["peak_db"], level)
            ev["t_last"] = t
        elif ev is not None and t - ev["t_last"] > self.s["event_hangover_s"]:
            self.cur_event = None
            if len(ev["b"]) < 2:                    # one frame is a tick, not an event
                return
            order = np.argsort(ev["b"])
            b = np.asarray(ev["b"])[order]
            w = np.cumsum(np.asarray(ev["w"])[order])
            med = float(b[np.searchsorted(w, w[-1] / 2)])
            rec = {"id": self.next_event_id, "time": round(ev["wall"], 3),
                   "t0": ev["t0"], "t1": ev["t_last"],
                   "duration_s": round(ev["t_last"] - ev["t0"], 2),
                   "bearing_deg": round(med, 1),
                   "spread_deg": round(float(np.percentile(b, 90) - np.percentile(b, 10)), 1),
                   "coherence": round(float(np.mean(ev["w"])), 3),
                   "peak_db": round(ev["peak_db"], 1), "frames": len(ev["b"]),
                   "labels": ev["labels"]}
            self.next_event_id += 1
            self.h["events"] += 1
            self.events.append(rec)

    # -- calibration ---------------------------------------------------------
    def start_calibration(self, seconds):
        with self.lock:
            self.calib = {"running": True, "until": self.t_audio + seconds,
                          "seconds": seconds, "taus": [], "result": None}

    def _calibrate(self, m, snr):
        c = self.calib
        if not c.get("running"):
            return
        # every coherent, loud frame — including out-of-range ones, which are
        # exactly what a too-small configured spacing produces
        if snr >= self.s["snr_db"] and m["coherence"] >= max(0.3, self.s["coherence_min"]):
            c["taus"].append(m["tau_s"])
        if self.t_audio >= c["until"]:
            c["running"] = False
            sp = spacing_from_endfire(c["taus"], self.s["speed_of_sound"])
            sign = None
            if c["taus"]:
                sign = "right" if float(np.median(c["taus"])) < 0 else "left"
            c["result"] = {"frames": len(c["taus"]),
                           "spacing_mm": None if sp is None else round(sp * 1000, 1),
                           "source_side": sign}

    def calibration_status(self):
        with self.lock:
            c = self.calib
            out = {"running": bool(c.get("running"))}
            if c.get("running"):
                out["remaining_s"] = round(max(0.0, c["until"] - self.t_audio), 1)
                out["frames"] = len(c["taus"])
            if c.get("result") is not None:
                out["result"] = c["result"]
            return out

    # -- classifier thread ---------------------------------------------------
    def run_classifier(self):
        while not self._stop.is_set():
            hop = self.s["classify_hop_s"]
            if self._stop.wait(hop):
                return
            if not (self.cls and self.cls.ready and self.s["classify_enabled"]):
                continue
            with self.lock:
                t_end = self.t_audio
                recent = t_end - self.last_active_t <= CLS_WINDOW / CLS_RATE
                filled = self.ring.filled >= CLS_WINDOW
                wave = self.ring.last(CLS_WINDOW) if filled else None
            if wave is None:
                continue
            if self.s["classify_only_active"] and not recent:
                self.h["classify_skipped"] += 1
                with self.lock:
                    self.sound = {"top": [], "t": t_end, "idle": True}
                continue
            try:
                scores = self.cls.scores(wave)
            except Exception as e:                  # noqa: BLE001
                self.h["classify_errors"] += 1
                log(f"[classify] {e}")
                continue
            self.h["classify_runs"] += 1
            top = self.cls.top(scores, 5, set(self.s["ignore_classes"]))
            with self.lock:
                self.sound = {"top": top, "t": t_end, "idle": False}
                self._attach_labels(t_end - CLS_WINDOW / CLS_RATE, t_end, top)

    def _attach_labels(self, w0, w1, top):
        """Fold a classifier window into every event it overlaps: the open one
        and any closed event that ended inside the window. Per label, keep the
        best score seen."""
        targets = []
        if self.cur_event is not None:
            targets.append(self.cur_event)
        for ev in reversed(self.events):
            if ev["t1"] < w0:
                break
            if ev["t0"] <= w1:
                targets.append(ev)
        minsc = self.s["class_min_score"]
        for ev in targets:
            for c in top:
                if c["score"] >= minsc and c["score"] > ev["labels"].get(c["label"], 0):
                    ev["labels"][c["label"]] = c["score"]

    # -- read side -----------------------------------------------------------
    def state(self):
        with self.lock:
            snd = dict(self.sound)
            if snd.get("t") is not None:
                snd["age_s"] = round(self.t_audio - snd.pop("t"), 2)
            else:
                snd.pop("t", None)
            return {"time": round(time.time(), 3),
                    "running": self.capture_error is None and self.h["frames"] > 0,
                    "level_db": self.levels, "peak_db": self.peak,
                    "band_db": self.band_db, "frame": self.frame,
                    "track": self.tracker.state(self.t_audio),
                    "in_event": self.cur_event is not None,
                    "sound": snd}

    def spectrum(self):
        with self.lock:
            out = {"freq": self.disp_freq.tolist(),
                   "band": [self.s["band_lo_hz"], self.s["band_hi_hz"]]}
            if self.spec_avg is not None:
                out["L"] = self.spec_avg[0].round(1).tolist()
                out["R"] = self.spec_avg[1].round(1).tolist()
            if self.resp_avg is not None:
                out["response"] = {"tau_us": (self.doa.tau * 1e6).round(2).tolist(),
                                   "r": self.resp_avg.round(3).tolist(),
                                   "tau_max_us": round(self.doa.tau_max * 1e6, 2)}
            return out

    def events_since(self, since):
        with self.lock:
            out = []
            for e in self.events:
                if e["id"] > since:
                    d = {k: v for k, v in e.items() if k not in ("t0", "t1")}
                    d["labels"] = dict(sorted(e["labels"].items(), key=lambda kv: -kv[1])[:5])
                    d["label"] = next(iter(d["labels"]), None)
                    out.append(d)
            return out

    def health(self):
        with self.lock:
            pm = list(self.proc_ms)
            hop_ms = self.hop / RATE * 1000
            return {
                "uptime_s": round(time.monotonic() - self.started, 1),
                "device": CAPTURE_CMD or DEVICE, "card": CARD, "rate": RATE,
                "nfft": self.nfft, "hop_ms": round(hop_ms, 2),
                "audio_s": round(self.t_audio, 1),
                "proc_ms_mean": round(sum(pm) / len(pm), 3) if pm else None,
                "proc_ms_max": round(max(pm), 3) if pm else None,
                "capture_error": self.capture_error,
                "codec_errors": self.codec_errors,
                "counters": dict(self.h),
                "classifier": {
                    "ready": bool(self.cls and self.cls.ready),
                    "runtime": self.cls.runtime if self.cls else None,
                    "error": self.cls.error if self.cls else "disabled",
                    "last_ms": self.cls.last_ms if self.cls else None,
                    "model": MODEL},
                "state_file": STATE_FILE,
            }

    def stop(self):
        self._stop.set()
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()


# ─────────────────────────────────────────────────────────────────────────────
ENGINE = None


def _np_default(o):
    """numpy scalars that slip into a response: serialise, don't 500."""
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(f"{type(o).__name__} is not JSON serializable")


class Handler(BaseHTTPRequestHandler):
    server_version = "respeaker-console/2"

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        if not isinstance(body, (bytes, bytearray)):
            body = json.dumps(body, separators=(",", ":"), default=_np_default).encode()
        try:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass                                    # page closed mid-poll

    def _json_body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n > 65536:
            raise ValueError("body too large")
        raw = self.rfile.read(n) if n else b"{}"
        data = json.loads(raw or b"{}")
        if not isinstance(data, dict):
            raise ValueError("body must be a JSON object")
        return data

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path in ("/", "/index.html"):
            return self._send(200, (ROOT / "static" / "index.html").read_bytes(),
                              "text/html; charset=utf-8")
        if u.path == "/state":
            return self._send(200, ENGINE.state())
        if u.path == "/spectrum":
            return self._send(200, ENGINE.spectrum())
        if u.path == "/events":
            try:
                since = int(q.get("since", ["0"])[0])
            except ValueError:
                since = 0
            return self._send(200, {"events": ENGINE.events_since(since)})
        if u.path == "/health":
            return self._send(200, ENGINE.health())
        if u.path == "/config":
            limits = {k: ({"choices": list(v[1])} if len(v) == 2 and isinstance(v[1], tuple)
                          else {"type": v[1].__name__} if len(v) == 2
                          else {"min": v[2], "max": v[3]}) for k, v in SPEC.items()}
            return self._send(200, {"settings": ENGINE.s, "limits": limits})
        if u.path == "/calibrate":
            return self._send(200, ENGINE.calibration_status())
        return self._send(404, {"ok": False, "error": "not found"})

    def do_POST(self):
        u = urlparse(self.path)
        try:
            data = self._json_body()
        except (ValueError, json.JSONDecodeError) as e:
            return self._send(400, {"ok": False, "error": str(e)})
        if u.path == "/config":
            clean, errors = validate(data, ENGINE.s)
            if errors:
                return self._send(400, {"ok": False, "errors": errors})
            ENGINE.update_settings(clean)
            err = save_settings(ENGINE.s)
            return self._send(200, {"ok": True, "changed": sorted(clean), "save_error": err,
                                    "codec_errors": ENGINE.codec_errors, "settings": ENGINE.s})
        if u.path == "/calibrate":
            try:
                secs = min(max(float(data.get("seconds", 5)), 1.0), 30.0)
            except (TypeError, ValueError):
                return self._send(400, {"ok": False, "error": "seconds must be a number"})
            ENGINE.start_calibration(secs)
            return self._send(200, {"ok": True, "seconds": secs})
        if u.path == "/track/reset":
            with ENGINE.lock:
                ENGINE.tracker.reset()
            return self._send(200, {"ok": True})
        return self._send(404, {"ok": False, "error": "not found"})


def main():
    global ENGINE
    settings = load_settings()
    cls = YamNet(MODEL, LABELS, THREADS)
    if cls.error:
        log(f"[classify] unavailable: {cls.error} — direction finding runs without it")
    else:
        log(f"[classify] YAMNet on {cls.runtime}, {THREADS} threads")
    ENGINE = Engine(settings, cls)
    threading.Thread(target=ENGINE.run_capture, name="capture", daemon=True).start()
    threading.Thread(target=ENGINE.run_classifier, name="classify", daemon=True).start()
    srv = ThreadingHTTPServer(("0.0.0.0", HTTP_PORT), Handler)
    srv.daemon_threads = True
    log(f"respeaker-console on :{HTTP_PORT}  device {CAPTURE_CMD or DEVICE}  "
        f"spacing {settings['spacing_mm']} mm  band {settings['band_lo_hz']:.0f}-"
        f"{settings['band_hi_hz']:.0f} Hz")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        ENGINE.stop()


if __name__ == "__main__":
    main()
