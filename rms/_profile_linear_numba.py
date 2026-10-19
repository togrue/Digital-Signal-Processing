"""Per-stage timing of linear_frac_rms_numba + Numba inspect helpers."""

from __future__ import annotations

import time

import numpy as np
from numba import njit

import linear_frac_rms_numba as m

FS = 102400.0
N = 16384


@njit(cache=False, fastmath=True)
def _resid_rms(y, i0, need, n_frac, intercept, slope, buf):
    # buf unused; keeps the old stage timer API stable.
    return m._frac_rms_linear(y, i0, n_frac, intercept, slope)


def _time_calls(fn, args_list, repeats: int = 9) -> float:
    fn(*args_list[0])
    best = 1e9
    for _ in range(repeats):
        t0 = time.perf_counter()
        for args in args_list:
            fn(*args)
        best = min(best, time.perf_counter() - t0)
    return best / len(args_list)


def main() -> None:
    t = np.arange(0.0, 15.0, 1.0 / FS)
    y_full = (
        10.0 * np.exp(-t / 2.5)
        + np.sqrt(2.0) * np.sin(2 * np.pi * (1000.0 / (2 * np.pi)) * t + 0.1)
    )
    windows = [
        np.ascontiguousarray(y_full[s : s + N], dtype=np.float64)
        for s in range(0, y_full.size - N + 1, N)
    ]

    m.warmup(N, FS)
    work0 = np.empty(N, dtype=np.float64)
    m._rough_linear_detrend(windows[0], work0)
    m._frac_zero_crossings(work0)
    m._linear_from_moving_average(windows[0], 643.4)
    buf0 = np.empty(N, dtype=np.float64)
    _resid_rms(windows[0], 0, N, float(N), 0.0, 0.0, buf0)
    m._linear_fractional_rms_core(windows[0], FS)

    works = [np.empty(N, dtype=np.float64) for _ in windows]
    detrend_args = [(y, w) for y, w in zip(windows, works)]
    for y, w in detrend_args:
        m._rough_linear_detrend(y, w)
    zc_args = [(w,) for _, w in detrend_args]

    fit_args = []
    resid_args = []
    for y, w in detrend_args:
        ok, za, zb, n_per = m._frac_zero_crossings(w)
        period = (zb - za) / n_per
        fit_args.append((y, period))
        ok, intercept, slope = m._linear_from_moving_average(y, period)
        i0 = int(za)
        n_frac = zb - za
        need = int(np.ceil(n_frac))
        buf = np.empty(N, dtype=np.float64)
        resid_args.append((y, i0, need, n_frac, intercept, slope, buf))

    full_args = [(y, FS) for y in windows]
    wrap_args = [(y, FS) for y in windows]

    print(f"{len(windows)} windows, best-of-9, ms/win:")
    rows = [
        ("rough detrend", _time_calls(m._rough_linear_detrend, detrend_args)),
        ("zero crossings", _time_calls(m._frac_zero_crossings, zc_args)),
        ("movavg LS fit", _time_calls(m._linear_from_moving_average, fit_args)),
        ("residual+frac_rms", _time_calls(_resid_rms, resid_args)),
        ("full core", _time_calls(m._linear_fractional_rms_core, full_args)),
        (
            "python wrapper",
            _time_calls(lambda y, fs: m.linear_fractional_rms(y, fs), wrap_args),
        ),
    ]
    total_stages = sum(r[1] for r in rows[:4])
    for name, sec in rows:
        share = ""
        if name in {"rough detrend", "zero crossings", "movavg LS fit", "residual+frac_rms"}:
            share = f"  ({100 * sec / total_stages:5.1f}% of staged sum)"
        print(f"  {name:18s} {sec * 1e3:7.4f}{share}")
    print(f"  staged sum         {total_stages * 1e3:7.4f}")

    core = m._linear_fractional_rms_core
    sig = list(core.overloads.keys())[0]
    llvm = core.inspect_llvm(sig)
    asm = core.inspect_asm(sig)
    print(f"\nCompiled core signature: {sig}")
    print(f"LLVM IR size: {len(llvm):,} chars")
    print(f"ASM size:     {len(asm):,} chars")
    asm_l = asm.lower()
    for kw in ("sqrtsd", "divsd", "mulsd", "addsd", "subsd", "callq", "ja ", "jb ", "jbe"):
        print(f"  asm '{kw}': {asm_l.count(kw)}")

    # Rough op counts for the moving-average double pass vs single-pass potential
    # period ~643, stride 4, m ~ (16384-643)/4 ≈ 3900 nodes; each node touches csum+y
    period = fit_args[0][1]
    half = period * 0.5
    i0 = int(np.ceil(half)) + 1
    i1 = int(np.floor(N - 1 - half))
    stride = max(1, min(4, int(period // 32)))
    m_nodes = 1 + (i1 - 1 - i0) // stride
    print(f"\nTypical fit nodes m={m_nodes}, stride={stride}, period={period:.2f}")
    print("Fit currently scans those nodes twice (mean, then cov).")


if __name__ == "__main__":
    main()
