"""Smoke bench: NumPy linear fractional RMS on an exp+sine stream."""

from __future__ import annotations

import time

import numpy as np

from dsp.rms import linear_fractional_rms

FS = 102400.0
WINDOW = 16384


def main() -> None:
    t = np.arange(0.0, 15.0, 1.0 / FS)
    tone = 1000.0 / (2.0 * np.pi)
    signal = 10.0 * np.exp(-t / 2.5) + np.sqrt(2.0) * np.sin(2 * np.pi * tone * t + 0.1)

    def once():
        return [
            linear_fractional_rms(signal[s : s + WINDOW], FS)
            for s in range(0, signal.size - WINDOW + 1, WINDOW)
        ]

    once()
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        found = once()
        times.append(time.perf_counter() - t0)
    elapsed = min(times)
    ppm = np.array([abs(est.rms - 1.0) * 1e6 for est in found if est.strategy == "linear_frac"])
    print(
        f"{len(found)} windows in {elapsed:.3f} s "
        f"({elapsed / len(found) * 1e3:.3f} ms/window), "
        f"median |error| {np.median(ppm):.3f} ppm, max {np.max(ppm):.3f} ppm"
    )


if __name__ == "__main__":
    main()
