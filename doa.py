"""Two-microphone direction finding, activity detection and tracking. numpy only.

Geometry
--------
Two microphones on a line, spacing d metres apart. A far-field source at
bearing theta (0 = straight ahead of the board, perpendicular to the mic axis,
positive = to the right) reaches the right mic earlier than the left by

    tau = d * sin(theta) / c

so the measured inter-mic delay maps to sin(theta), never to theta itself.
Two consequences that are geometry, not software:

  * front/back ambiguity — a source at 30 deg right-front and one at 150 deg
    right-rear give the same delay. Bearings are reported in -90..+90.
  * resolution falls off toward the mic axis (+-90 deg): d(theta) = c*d(tau) /
    (d*cos(theta)). Near broadside a quarter-sample error is ~1 deg; at 70 deg
    it is ~3 deg.

Turning toward a reported bearing still converges on the source from the rear
half: a rear-right source reads as front-right, the robot turns right, the
true bearing passes 90 deg and from there on reads correctly. Only a source
exactly behind (reads 0) gives no turn.

Delay estimator
---------------
GCC-PHAT evaluated directly on a fine delay grid (SRP-PHAT for one mic pair):

    R(tau) = 1/K * sum_k Re{ X_L[k] X_R*[k] / |X_L[k] X_R*[k]| * exp(-j 2 pi f_k tau) }

over the bins k inside the analysis band. PHAT weighting throws away
magnitude and keeps phase, so R is in -1..1 and is the fraction of the band
that agrees on one delay — a usable coherence measure on its own. Evaluating
on a grid of delays (default quarter-sample steps) instead of an inverse FFT
gives sub-sample resolution without zero padding, and only costs
K x n_tau multiplies (about 160 x 170 at the defaults).

The grid deliberately extends past the physical limit +-d/c (to
tau_search_m / c). A peak outside +-d/c means either the configured spacing
is wrong or the frame is reverberation/noise; calibrate() uses those frames
to measure the spacing.
"""

import math

import numpy as np

SPEED_OF_SOUND = 343.0          # m/s at 20 C; +0.6 m/s per deg C


# ─────────────────────────────────────────────────────────────────────────────
class PairDOA:
    """One mic pair. process() takes one stereo frame, returns a measurement."""

    def __init__(self, fs, nfft, spacing_m, band=(300.0, 4000.0), c=SPEED_OF_SOUND,
                 tau_search_m=0.15, tau_step_samples=0.25):
        self.fs = float(fs)
        self.nfft = int(nfft)
        self.spacing_m = float(spacing_m)
        self.c = float(c)
        self.band = (float(band[0]), float(band[1]))
        self.window = np.hanning(self.nfft).astype(np.float32)
        self.win_power = float(np.sum(self.window ** 2))
        freqs = np.fft.rfftfreq(self.nfft, 1.0 / self.fs)
        lo, hi = self.band
        lo = max(lo, freqs[1])
        hi = min(hi, self.fs / 2 * 0.98)
        self.kband = np.nonzero((freqs >= lo) & (freqs <= hi))[0]
        if len(self.kband) < 4:
            raise ValueError(f"band {band} holds fewer than 4 FFT bins at nfft={nfft}")
        n = int(math.ceil(tau_search_m / self.c * self.fs / tau_step_samples))
        self.tau = (np.arange(-n, n + 1) * tau_step_samples / self.fs)          # s
        f = freqs[self.kband]
        self.steer = np.exp(-2j * np.pi * np.outer(self.tau, f)).astype(np.complex64)
        self.tau_max = self.spacing_m / self.c

    def bearing_of(self, tau):
        """Delay (s, positive = right mic later) to bearing in degrees, + = right."""
        s = -tau / self.tau_max
        return math.degrees(math.asin(max(-1.0, min(1.0, s))))

    def spectra(self, frame):
        """frame: (nfft, 2) float32, -1..1. Returns the two rfft arrays."""
        X = np.fft.rfft(frame * self.window[:, None], axis=0)
        return X[:, 0], X[:, 1]

    def band_level_db(self, XL, XR):
        """Mean-square level of the band-limited signal, both mics, in dBFS
        (0 dBFS = a full-scale square wave; a full-scale sine is -3)."""
        k = self.kband
        p = (np.sum(np.abs(XL[k]) ** 2) + np.sum(np.abs(XR[k]) ** 2)) / 2.0
        ms = 2.0 * p / (self.nfft * self.win_power)
        return 10.0 * math.log10(max(ms, 1e-12))

    def measure(self, XL, XR):
        k = self.kband
        cross = XL[k] * np.conj(XR[k])
        cross = cross / (np.abs(cross) + 1e-12)
        R = (self.steer @ cross.astype(np.complex64)).real / len(k)
        i = int(np.argmax(R))
        tau = float(self.tau[i])
        if 0 < i < len(R) - 1:                      # parabolic sub-grid refine
            a, b, cc = R[i - 1], R[i], R[i + 1]
            den = a - 2 * b + cc
            if den < 0:
                off = float(0.5 * (a - cc) / den)
                tau += off * float(self.tau[1] - self.tau[0])
        return {
            "tau_s": tau,
            "coherence": float(R[i]),
            "bearing_deg": self.bearing_of(tau),
            "in_range": abs(tau) <= self.tau_max * 1.05,
            "response": R,
        }


# ─────────────────────────────────────────────────────────────────────────────
class NoiseFloor:
    """Minimum-follower: drops to a quieter level at once, climbs toward a
    louder one at rise_db_per_s. A sustained sound is absorbed into the floor
    after (level - floor) / rise seconds — at 3 dB/s, a constant 12 dB hum
    stops counting as activity in about 4 s. Short sounds barely move it."""

    def __init__(self, rise_db_per_s=3.0):
        self.rise = float(rise_db_per_s)
        self.db = None

    def update(self, level_db, dt):
        if self.db is None or level_db < self.db:
            self.db = level_db
        else:
            self.db = min(level_db, self.db + self.rise * dt)
        return self.db


# ─────────────────────────────────────────────────────────────────────────────
class BearingTracker:
    """Decaying bearing histogram, 1 deg bins over -90..+90.

    Each accepted frame adds a bump at its bearing, weighted by its coherence,
    with a width that grows toward +-90 where the geometry is less precise.
    The whole histogram decays with the given half-life. The track is the
    histogram peak: a single stray frame cannot move it, two sources
    alternating show up as two peaks, and a source that stops fades out on
    its own instead of being held forever."""

    BINS = np.arange(-90, 91, dtype=np.float32)

    def __init__(self, half_life_s=0.75, sigma_deg=3.0, min_strength=0.6):
        self.half_life_s = float(half_life_s)
        self.sigma_deg = float(sigma_deg)
        self.min_strength = float(min_strength)
        self.h = np.zeros(len(self.BINS), np.float32)
        self.last_hit = None

    def decay(self, dt):
        if self.half_life_s > 0:
            self.h *= 0.5 ** (dt / self.half_life_s)

    def add(self, bearing_deg, weight, now):
        cosb = max(math.cos(math.radians(bearing_deg)), 0.25)
        sigma = self.sigma_deg / cosb
        self.h += weight * np.exp(-0.5 * ((self.BINS - bearing_deg) / sigma) ** 2)
        self.last_hit = now

    def peak(self):
        i = int(np.argmax(self.h))
        strength = float(self.h[i])
        b = float(self.BINS[i])
        if 0 < i < len(self.h) - 1:
            a, m, c = self.h[i - 1], self.h[i], self.h[i + 1]
            den = a - 2 * m + c
            if den < 0:
                b += float(0.5 * (a - c) / den)
        return float(b), strength

    def state(self, now):
        b, s = self.peak()
        tracking = s >= self.min_strength
        return {"tracking": tracking,
                "bearing_deg": round(b, 1) if tracking else None,
                "strength": round(s, 3),
                "age_s": None if self.last_hit is None else round(now - self.last_hit, 3)}

    def reset(self):
        self.h[:] = 0
        self.last_hit = None


# ─────────────────────────────────────────────────────────────────────────────
class Decimator:
    """Integer-factor FIR decimator with state, for streaming blocks of any
    length. Windowed-sinc low-pass at cutoff_hz (default 0.45 x output rate)."""

    def __init__(self, fs_in, factor, taps=96, cutoff_hz=None):
        self.factor = int(factor)
        fs_out = fs_in / self.factor
        fc = (cutoff_hz or 0.45 * fs_out) / fs_in
        n = np.arange(taps) - (taps - 1) / 2.0
        h = 2 * fc * np.sinc(2 * fc * n) * np.blackman(taps)
        self.h = (h / np.sum(h)).astype(np.float32)
        self.hist = np.zeros(taps - 1, np.float32)
        self.phase = 0          # index in the next block of the next kept sample

    def process(self, x):
        buf = np.concatenate((self.hist, x.astype(np.float32)))
        y = np.convolve(buf, self.h, mode="valid")          # len(x) samples
        out = y[self.phase::self.factor]
        self.phase = (self.phase - len(x)) % self.factor
        self.hist = buf[-(len(self.h) - 1):]
        return out


class Ring:
    """Fixed-length float32 ring for the classifier's input history."""

    def __init__(self, n):
        self.buf = np.zeros(n, np.float32)
        self.n = n
        self.filled = 0

    def push(self, x):
        x = x[-self.n:]
        k = len(x)
        self.buf = np.roll(self.buf, -k)
        self.buf[-k:] = x
        self.filled = min(self.n, self.filled + k)

    def last(self, m):
        return self.buf[-m:].copy()


# ─────────────────────────────────────────────────────────────────────────────
def fractional_delay(x, delay_samples):
    """Delay x by a fractional number of samples (frequency-domain shift,
    circular). Used by the tests and the calibration self-check."""
    n = len(x)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n)
    return np.fft.irfft(X * np.exp(-2j * np.pi * f * delay_samples), n)


def spacing_from_endfire(taus, c=SPEED_OF_SOUND, pct=95):
    """Mic spacing implied by delays measured with the source on the mic axis
    (directly off one end of the board). Uses a high percentile of |tau| rather
    than the max so one reverberant frame can't set it."""
    taus = np.abs(np.asarray(taus, float))
    if len(taus) < 5:
        return None
    return float(np.percentile(taus, pct) * c)
