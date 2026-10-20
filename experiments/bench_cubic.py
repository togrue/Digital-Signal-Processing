"""Smoke bench: adaptive-order (cubic) fractional RMS on an exp+sine stream."""

from __future__ import annotations

import time

import numpy as np

from dsp.rms import cubic_fractional_rms

FS = 102400.0
WINDOW = 16384
ACCURATE = 1.0


def main() -> None:
    t = np.arange(0.0, 15.0, 1.0 / FS)
    tone = 1000.0 / (2.0 * np.pi)
    y = 10.0 * np.exp(-t / 2.5) + np.sqrt(2.0) * ACCURATE * np.sin(
        2 * np.pi * tone * t + 0.1
    )

    def once():
        found = []
        period = None
        for start in range(0, y.size - WINDOW + 1, WINDOW):
            est = cubic_fractional_rms(y[start : start + WINDOW], FS, period)
            period = est.period_samples
            found.append(est)
        return found

    once()
    t0 = time.perf_counter()
    found = once()
    elapsed = time.perf_counter() - t0
    ppm = np.array([abs(est.rms - ACCURATE) / ACCURATE * 1e6 for est in found])
    methods = {est.method for est in found}
    trusted = sum(est.trustworthy for est in found)
    print(
        f"{len(found)} windows in {elapsed:.3f} s "
        f"({elapsed / len(found) * 1e3:.2f} ms/window), "
        f"median |error| {np.median(ppm):.4f} ppm, max {np.max(ppm):.4f} ppm, "
        f"methods {sorted(methods)}, trustworthy {trusted}/{len(found)}"
    )


if __name__ == "__main__":
    main()
