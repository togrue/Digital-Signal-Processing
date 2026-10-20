"""Compare NumPy vs Numba linear fractional RMS throughput and agreement."""

from __future__ import annotations

import time

import numpy as np

from dsp.rms import linear_fractional_rms as linear_numpy
from dsp.rms.linear_numba import linear_fractional_rms, warmup

FS = 102400.0
WINDOW = 16384


def main() -> None:
    t = np.arange(0.0, 15.0, 1.0 / FS)
    tone = 1000.0 / (2.0 * np.pi)
    signal = 10.0 * np.exp(-t / 2.5) + np.sqrt(2.0) * np.sin(2 * np.pi * tone * t + 0.1)
    starts = list(range(0, signal.size - WINDOW + 1, WINDOW))

    print("compiling…")
    warmup(WINDOW, FS)

    def bench(fn, repeats=7):
        def once():
            return [fn(signal[s : s + WINDOW], FS) for s in starts]

        once()
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            out = once()
            times.append(time.perf_counter() - t0)
        return min(times), out

    t_np, out_np = bench(linear_numpy)
    t_nb, out_nb = bench(linear_fractional_rms)
    n = len(starts)
    print(f"numpy:  {t_np / n * 1e3:.3f} ms/win")
    print(f"numba:  {t_nb / n * 1e3:.3f} ms/win  ({t_np / t_nb:.2f}× vs numpy)")

    max_diff = max(abs(a.rms - b.rms) for a, b in zip(out_np, out_nb))
    strat_ok = all(a.strategy == b.strategy for a, b in zip(out_np, out_nb))
    ppm = np.array([abs(e.rms - 1.0) * 1e6 for e in out_nb if e.strategy == "linear_frac"])
    print(f"max |rms diff| vs numpy: {max_diff:.3e}")
    print(f"strategies match: {strat_ok}")
    print(f"numba median |error| {np.median(ppm):.3f} ppm, max {np.max(ppm):.3f} ppm")


if __name__ == "__main__":
    main()
