import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from doa import (BearingTracker, Decimator, NoiseFloor, PairDOA, Ring,  # noqa: E402
                 fractional_delay, spacing_from_endfire)

FS, N, D = 48000, 2048, 0.058


def stereo_from(bearing, n=N, level=0.1, seed=1, spacing=D, noise=0.0):
    """Noise from `bearing` (deg, + = right): the right mic hears it first."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n * 3) * level
    tau = spacing * math.sin(math.radians(bearing)) / 343.0
    L = fractional_delay(x, tau * FS)[n:2 * n]
    R = x[n:2 * n]
    if noise:
        L = L + rng.standard_normal(n) * noise
        R = R + rng.standard_normal(n) * noise
    return np.stack((L, R), 1).astype(np.float32)


class PairDOATest(unittest.TestCase):
    def setUp(self):
        self.d = PairDOA(FS, N, D)

    def test_bearing_across_the_front(self):
        for b in (-75, -45, -20, -5, 0, 5, 20, 45, 75):
            m = self.d.measure(*self.d.spectra(stereo_from(b)))
            self.assertAlmostEqual(m["bearing_deg"], b, delta=1.0, msg=f"bearing {b}")
            self.assertGreater(m["coherence"], 0.9)
            self.assertTrue(m["in_range"])

    def test_sign_convention_right_is_positive(self):
        m = self.d.measure(*self.d.spectra(stereo_from(40)))
        self.assertGreater(m["bearing_deg"], 0)
        self.assertLess(m["tau_s"], 0)       # right mic earlier

    def test_uncorrelated_noise_is_incoherent(self):
        rng = np.random.default_rng(3)
        fr = (rng.standard_normal((N, 2)) * 0.1).astype(np.float32)
        m = self.d.measure(*self.d.spectra(fr))
        self.assertLess(m["coherence"], 0.2)

    def test_moderate_snr_unbiased(self):
        # ~6 dB SNR, uncorrelated per mic: single frames scatter (RMS ~2 deg)
        # but do not lean; the tracker averages the scatter out.
        err = np.array([self.d.measure(*self.d.spectra(stereo_from(30, seed=s, noise=0.05)))
                        ["bearing_deg"] - 30 for s in range(40)])
        self.assertLess(abs(err.mean()), 1.0)
        self.assertLess(np.sqrt(np.mean(err ** 2)), 3.0)

    def test_wrong_spacing_shows_out_of_range(self):
        small = PairDOA(FS, N, 0.030)          # configured 30 mm, real 58 mm
        m = small.measure(*small.spectra(stereo_from(80)))
        self.assertFalse(m["in_range"])

    def test_band_level_matches_white_noise(self):
        # white noise RMS 0.1 (-20 dBFS) over 300-4000 Hz of 24 kHz -> -28.1 dB
        lv = self.d.band_level_db(*self.d.spectra(stereo_from(0, level=0.1)))
        self.assertAlmostEqual(lv, -28.1, delta=1.5)

    def test_too_narrow_band_rejected(self):
        with self.assertRaises(ValueError):
            PairDOA(FS, 256, D, band=(1000, 1100))


class TrackerTest(unittest.TestCase):
    def test_one_outlier_does_not_move_track(self):
        t = BearingTracker(half_life_s=1.0, min_strength=1.0)
        for i in range(10):
            t.add(20, 0.8, i * 0.02)
            t.decay(0.02)
        t.add(-60, 0.9, 0.2)
        st = t.state(0.2)
        self.assertTrue(st["tracking"])
        self.assertAlmostEqual(st["bearing_deg"], 20, delta=1)

    def test_track_fades_out(self):
        t = BearingTracker(half_life_s=0.5, min_strength=1.0)
        for _ in range(10):
            t.add(10, 1.0, 0)
        self.assertTrue(t.state(0)["tracking"])
        t.decay(5.0)
        self.assertFalse(t.state(5)["tracking"])

    def test_sub_degree_refine(self):
        t = BearingTracker()
        for _ in range(5):
            t.add(12.4, 1.0, 0)
        self.assertAlmostEqual(t.peak()[0], 12.4, delta=0.2)


class NoiseFloorTest(unittest.TestCase):
    def test_follows_down_instantly_up_slowly(self):
        f = NoiseFloor(rise_db_per_s=3.0)
        f.update(-60, 0.02)
        self.assertEqual(f.update(-70, 0.02), -70)
        self.assertAlmostEqual(f.update(-40, 1.0), -67.0)


class DecimatorTest(unittest.TestCase):
    def tone_gain(self, hz, block=1024):
        d = Decimator(FS, 3)
        t = np.arange(FS) / FS
        x = np.sin(2 * np.pi * hz * t).astype(np.float32)
        y = np.concatenate([d.process(x[i:i + block]) for i in range(0, len(x), block)])
        self.assertEqual(len(y), FS // 3)
        return np.sqrt(np.mean(y[2000:] ** 2)) / np.sqrt(0.5)

    def test_passband_and_stopband(self):
        self.assertAlmostEqual(self.tone_gain(1000), 1.0, delta=0.02)
        self.assertLess(self.tone_gain(12000), 0.01)

    def test_odd_block_sizes_keep_phase(self):
        d = Decimator(FS, 3)
        x = np.random.default_rng(0).standard_normal(10007).astype(np.float32)
        ref = Decimator(FS, 3).process(x)
        parts, i = [], 0
        for n in (1, 2, 500, 1023, 7, 8474):
            parts.append(d.process(x[i:i + n]))
            i += n
        np.testing.assert_allclose(np.concatenate(parts), ref, atol=1e-5)


class MiscTest(unittest.TestCase):
    def test_ring(self):
        r = Ring(5)
        r.push(np.arange(3, dtype=np.float32))
        r.push(np.arange(3, 6, dtype=np.float32))
        np.testing.assert_array_equal(r.last(5), [1, 2, 3, 4, 5])
        self.assertEqual(r.filled, 5)

    def test_spacing_from_endfire(self):
        taus = [D / 343.0 * s for s in (0.97, 0.99, 1.0, 1.0, -0.98, 0.95, 1.0)]
        self.assertAlmostEqual(spacing_from_endfire(taus) * 1000, 58, delta=1)
        self.assertIsNone(spacing_from_endfire([1e-4]))


if __name__ == "__main__":
    unittest.main()
