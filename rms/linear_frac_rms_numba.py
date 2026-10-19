"""Numba-accelerated linear-trend fractional AC RMS.

Same algorithm and public result shape as ``linear_frac_rms``, with the
per-window work compiled by Numba. Strategy codes inside the JIT path are
mapped back to the usual strings at the Python boundary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numba import njit

# Strategy codes returned by the JIT kernel.
_STR_INVALID = 0
_STR_WINDOW_AC = 1
_STR_LINEAR_FRAC = 2

_STRATEGY_NAME = {
    _STR_INVALID: "invalid",
    _STR_WINDOW_AC: "window_ac",
    _STR_LINEAR_FRAC: "linear_frac",
}

_JIT = dict(cache=True, fastmath=True)


@dataclass(frozen=True)
class LinearRmsEstimate:
    """RMS of one window and the strategy that produced it."""

    rms: float
    strategy: str
    frequency_hz: float = math.nan
    period_samples: float | None = None

    def __iter__(self):
        yield self.rms
        yield self.strategy


@njit(**_JIT)
def _interp_zero_crossing(y, i):
    y0 = y[i]
    y1 = y[i + 1]
    dy = y1 - y0
    if dy == 0.0:
        return float(i)
    return i - y0 / dy


@njit(**_JIT)
def _window_ac_rms(y):
    n = y.shape[0]
    if n == 0:
        return np.nan
    mean = 0.0
    energy = 0.0
    for i in range(n):
        v = y[i]
        mean += v
        energy += v * v
    mean /= n
    return math.sqrt(max(energy / n - mean * mean, 0.0))


@njit(**_JIT)
def _frac_rms_linear(y, i0, n, intercept, slope):
    """Fractional RMS of (y - intercept - slope*i) over length n, DC removed."""
    n_int = int(n)
    frac = n - n_int
    total = 0.0
    energy = 0.0
    for j in range(n_int):
        i = i0 + j
        v = y[i] - (intercept + slope * i)
        total += v
        energy += v * v
    if frac > 0.0:
        i = i0 + n_int
        last = y[i] - (intercept + slope * i)
        total += last * frac
        energy += last * last * frac
    avg = total / n
    return math.sqrt(max(energy / n - avg * avg, 0.0))


@njit(**_JIT)
def _frac_zero_crossings(y):
    """Return (ok, zero_a, zero_b, n_per). ok==0 means failure."""
    n = y.shape[0]
    crossings = np.empty(n - 1, dtype=np.int64)
    k = 0
    prev = y[0] < 0.0
    for i in range(1, n):
        cur = y[i] < 0.0
        if cur != prev:
            crossings[k] = i - 1
            k += 1
            prev = cur
    n_per = (k - 1) // 2
    if n_per <= 0:
        return 0, 0.0, 0.0, 0
    zero_a = _interp_zero_crossing(y, crossings[0])
    zero_b = _interp_zero_crossing(y, crossings[n_per * 2])
    if zero_b <= zero_a:
        return 0, 0.0, 0.0, 0
    return 1, zero_a, zero_b, n_per


@njit(**_JIT)
def _rough_linear_detrend(y, out):
    """Write mean-free linear-detrended residual into out."""
    n = y.shape[0]
    center = 0.5 * (n - 1)
    denom = n * (n * n - 1) / 12.0
    sum_y = 0.0
    sum_iy = 0.0
    for i in range(n):
        v = y[i]
        sum_y += v
        sum_iy += i * v
    y_mean = sum_y / n
    if denom <= 0.0:
        for i in range(n):
            out[i] = y[i] - y_mean
        return
    # cov(i, y) with centered i: sum(i*y) - center*sum(y)
    slope = (sum_iy - center * sum_y) / denom
    for i in range(n):
        out[i] = y[i] - y_mean - slope * (i - center)


@njit(**_JIT)
def _linear_from_moving_average(y, period):
    """Return (ok, intercept, slope) from a centered fractional MA.

    Length L = k*period with k = max(1, floor(n_cycles/4)). At centre i the
    average runs from i-L/2 to i+L/2 (partial first/last coefficients).
    """
    n = y.shape[0]
    if not math.isfinite(period) or period <= 1.0 or period >= n:
        return 0, 0.0, 0.0
    n_cycles = n / period
    k_periods = int(n_cycles // 4)
    if k_periods < 1:
        k_periods = 1
    length = k_periods * period
    if length >= n:
        return 0, 0.0, 0.0

    half = length * 0.5
    i0 = int(math.ceil(half)) + 1
    i1 = int(math.floor(n - 1 - half))
    if i1 - i0 < 2:
        return 0, 0.0, 0.0
    stride = int(period // 32)
    if stride < 1:
        stride = 1
    if stride > 4:
        stride = 4
    m = 1 + (i1 - 1 - i0) // stride
    if m < 2:
        return 0, 0.0, 0.0

    csum = np.empty(n + 1, dtype=np.float64)
    csum[0] = 0.0
    for i in range(n):
        csum[i + 1] = csum[i] + y[i]

    kr = int(math.floor(half))
    fr = half - kr
    if fr == 0.0:
        kl = kr
        fl = 0.0
    else:
        kl = kr + 1
        fl = 1.0 - fr

    t_mean = i0 + 0.5 * (m - 1) * stride
    denom = (stride * stride) * m * (m * m - 1) / 12.0
    if denom <= 0.0:
        return 0, 0.0, 0.0

    sum_a = 0.0
    sum_ta = 0.0
    for k in range(m):
        i = i0 + k * stride
        avg = csum[i + kr] - csum[i - kl]
        if fr != 0.0:
            avg += y[i + kr] * fr
        if fl != 0.0:
            avg -= y[i - kl] * fl
        avg /= length
        sum_a += avg
        sum_ta += i * avg
    a_mean = sum_a / m
    slope = (sum_ta - t_mean * sum_a) / denom
    intercept = a_mean - slope * t_mean
    return 1, intercept, slope


@njit(**_JIT)
def _linear_fractional_rms_core(y, fs):
    """Return (strategy_code, rms, frequency_hz, period_samples).

    Caller must pass a non-empty finite contiguous float64 window.
    """
    n = y.shape[0]
    work = np.empty(n, dtype=np.float64)
    _rough_linear_detrend(y, work)

    ok, zero_a, zero_b, n_per = _frac_zero_crossings(work)
    if ok == 0:
        return _STR_WINDOW_AC, _window_ac_rms(y), np.nan, np.nan

    period = (zero_b - zero_a) / n_per
    if not math.isfinite(period) or period <= 1.0:
        return _STR_WINDOW_AC, _window_ac_rms(y), np.nan, np.nan
    if n / period < 1.95:
        return _STR_WINDOW_AC, _window_ac_rms(y), np.nan, np.nan

    ok, intercept, slope = _linear_from_moving_average(y, period)
    if ok == 0:
        return _STR_WINDOW_AC, _window_ac_rms(y), np.nan, np.nan

    n_frac = zero_b - zero_a
    i0 = int(zero_a)
    if i0 < 0 or i0 + int(math.ceil(n_frac)) > n:
        return _STR_WINDOW_AC, _window_ac_rms(y), np.nan, np.nan

    rms = _frac_rms_linear(y, i0, n_frac, intercept, slope)
    return _STR_LINEAR_FRAC, rms, fs / period, period


def linear_fractional_rms(y_win, fs: float) -> LinearRmsEstimate:
    """Fractional AC RMS with linear trend removal (Numba).

    Parameters
    ----------
    y_win :
        One analysis window.
    fs :
        Sample rate in Hz.
    """
    y = np.ascontiguousarray(y_win, dtype=np.float64)
    if y.ndim != 1:
        raise ValueError("y_win must be one-dimensional")
    # Keep the finite check in Python: fastmath JIT does not honor IEEE NaNs.
    if y.size == 0 or not np.isfinite(y).all():
        return LinearRmsEstimate(math.nan, "invalid")

    code, rms, freq, period = _linear_fractional_rms_core(y, float(fs))
    strategy = _STRATEGY_NAME[int(code)]
    if strategy == "window_ac":
        return LinearRmsEstimate(float(rms), "window_ac")
    return LinearRmsEstimate(
        rms=float(rms),
        strategy="linear_frac",
        frequency_hz=float(freq),
        period_samples=float(period),
    )


def warmup(n: int = 16384, fs: float = 102400.0) -> None:
    """Compile the JIT kernels on a dummy window."""
    linear_fractional_rms(np.zeros(n, dtype=np.float64), fs)


if __name__ == "__main__":
    import time

    from linear_frac_rms import linear_fractional_rms as linear_numpy

    fs = 102400.0
    window = 16384
    t = np.arange(0.0, 15.0, 1.0 / fs)
    tone = 1000.0 / (2.0 * np.pi)
    signal = 10.0 * np.exp(-t / 2.5) + np.sqrt(2.0) * np.sin(2 * np.pi * tone * t + 0.1)
    starts = list(range(0, signal.size - window + 1, window))

    print("compiling…")
    warmup(window, fs)

    def bench(fn, repeats=7):
        def once():
            return [fn(signal[s : s + window], fs) for s in starts]

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
