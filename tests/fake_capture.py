#!/usr/bin/env python3
"""Stand-in for arecord: raw S16_LE stereo on stdout, noise bursts arriving
from a fixed bearing at the given mic spacing, over a low uncorrelated floor.

    RESPEAKER_CAPTURE_CMD="python3 tests/fake_capture.py --bearing 30" ./server.py

Paced to real time unless --fast. Exits after --seconds (0 = forever).
"""
import argparse
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from doa import fractional_delay  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rate", type=int, default=48000)
    p.add_argument("--bearing", type=float, default=30.0, help="deg, + = right")
    p.add_argument("--spacing-mm", type=float, default=58.0)
    p.add_argument("--on", type=float, default=0.4, help="burst length, s")
    p.add_argument("--off", type=float, default=0.6, help="gap between bursts, s")
    p.add_argument("--level", type=float, default=0.1, help="burst RMS")
    p.add_argument("--floor", type=float, default=0.001, help="background RMS")
    p.add_argument("--seconds", type=float, default=0.0)
    p.add_argument("--fast", action="store_true")
    a = p.parse_args()

    rng = np.random.default_rng(7)
    fs = a.rate
    # right mic earlier by tau for a source on the right -> delay the left channel
    tau = a.spacing_mm / 1000.0 * math.sin(math.radians(a.bearing)) / 343.0
    period = int((a.on + a.off) * fs)
    n_on = int(a.on * fs)
    out = sys.stdout.buffer
    t_start = time.monotonic()
    produced = 0
    while a.seconds <= 0 or produced < a.seconds * fs:
        src = np.zeros(period)
        burst = rng.standard_normal(n_on) * a.level
        ramp = min(480, n_on // 4)
        env = np.ones(n_on)
        env[:ramp] = np.linspace(0, 1, ramp)
        env[-ramp:] = np.linspace(1, 0, ramp)
        src[:n_on] = burst * env
        L = fractional_delay(src, tau * fs) + rng.standard_normal(period) * a.floor
        R = src + rng.standard_normal(period) * a.floor
        pcm = np.clip(np.stack((L, R), 1) * 32767, -32768, 32767).astype("<i2")
        for i in range(0, period, 1024):
            chunk = pcm[i:i + 1024]
            try:
                out.write(chunk.tobytes())
                out.flush()
            except BrokenPipeError:
                return
            produced += len(chunk)
            if not a.fast:
                lag = produced / fs - (time.monotonic() - t_start)
                if lag > 0:
                    time.sleep(lag)


if __name__ == "__main__":
    main()
