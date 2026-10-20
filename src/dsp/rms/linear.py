"""Linear-trend fractional AC RMS.

Estimates a period from zero crossings, removes a linear drift, then takes
a fractional RMS over whole periods.

The drift is a least-squares line fitted to a centered moving average whose
length is ``k`` periods (``k = max(1, floor(n_cycles/4))``). The average is
fractional and centered on each fit node: edges at ``±L/2`` give partial
weights on the first and last samples (each about ``{L}/2`` when that is
below half a sample).

Period is refined in two passes: a rough linear detrend gives an initial
period and MA line; zero crossings are then re-taken on that residual and
the line is refit. A single raw-signal linear detrend correlates with a
non-integer-period tone and biases the period (tens of ppm of RMS) even
when no drift is present.

If the window cannot support that path, the plain window AC RMS is returned
and ``strategy`` names the fallback.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LinearRmsEstimate:
    """RMS of one window and the strategy that produced it.

    Strategies:
    - ``linear_frac``: linear drift removed, fractional RMS over periods
    - ``window_ac``: fallback — plain AC RMS of the whole window
    - ``invalid``: empty or non-finite input
    """

    rms: float
    strategy: str
    frequency_hz: float = math.nan
    period_samples: float | None = None

    def __iter__(self):
        yield self.rms
        yield self.strategy


def _interp_zero_crossing(y: np.ndarray, i: int) -> float:
    y0 = float(y[i])
    y1 = float(y[i + 1])
    dy = y1 - y0
    if dy == 0.0:
        return float(i)
    return i - y0 / dy


def _frac_zero_crossings(y: np.ndarray) -> tuple[float, float, int] | None:
    crossings = np.flatnonzero(np.diff(np.signbit(y)))
    n_per = (crossings.size - 1) // 2
    if n_per <= 0:
        return None
    zero_a = _interp_zero_crossing(y, int(crossings[0]))
    zero_b = _interp_zero_crossing(y, int(crossings[n_per * 2]))
    if zero_b <= zero_a:
        return None
    return zero_a, zero_b, n_per


def _frac_rms(y: np.ndarray, n: float) -> float:
    n_int = int(n)
    frac = n - n_int
    seg = y[:n_int]
    total = float(np.sum(seg))
    energy = float(np.dot(seg, seg))
    if frac > 0.0:
        last = float(y[n_int])
        total += last * frac
        energy += last * last * frac
    avg = total / n
    return math.sqrt(max(energy / n - avg * avg, 0.0))


def _window_ac_rms(y: np.ndarray) -> float:
    if y.size == 0:
        return math.nan
    mean = float(y.mean())
    return math.sqrt(max(float(np.dot(y, y) / y.size) - mean * mean, 0.0))


def _rough_linear_detrend(y: np.ndarray) -> np.ndarray:
    """Mean-free residual after removing a least-squares line."""
    n = y.size
    x = np.arange(n, dtype=np.float64)
    x -= x.mean()
    y0 = y - y.mean()
    denom = float(x @ x)
    if denom <= 0.0:
        return y0
    slope = float(x @ y0) / denom
    return y0 - slope * x


def _linear_from_moving_average(y: np.ndarray, period: float) -> tuple[float, float] | None:
    """Intercept and slope of a line fit to a centered fractional MA.

    Averaging length is ``L = k * period`` with
    ``k = max(1, floor(n_cycles / 4))``. At each integer centre ``i`` the
    average runs from ``i - L/2`` to ``i + L/2``. When ``L`` is not an
    integer the first and last samples take the partial edge weights
    (``≈ {L}/2`` each when that quantity is below 0.5).
    """
    n = y.size
    if not math.isfinite(period) or period <= 1.0 or period >= n:
        return None
    n_cycles = n / period
    k_periods = max(1, int(n_cycles // 4))
    length = k_periods * period
    if length >= n:
        return None

    half = length * 0.5
    i0 = math.ceil(half) + 1
    i1 = math.floor(n - 1 - half)
    if i1 - i0 < 2:
        return None
    stride = max(1, min(4, int(period // 32)))
    centres = np.arange(i0, i1, stride)
    if centres.size < 2:
        return None

    csum = np.empty(n + 1, dtype=np.float64)
    csum[0] = 0.0
    np.cumsum(y, out=csum[1:])

    # Edges at ±half: same fractional layout as cubic_frac_rms.
    kr = math.floor(half)
    fr = half - kr
    if fr == 0.0:
        kl, fl = kr, 0.0
    else:
        kl, fl = kr + 1, 1.0 - fr

    avg = csum[centres + kr] - csum[centres - kl]
    m = avg.shape[0]
    if m < 2:
        return None
    if fr != 0.0:
        avg = avg + y[centres + kr][:m] * fr
    if fl != 0.0:
        avg = avg - y[centres - kl][:m] * fl
    avg = avg / length

    tt = centres[:m].astype(np.float64)
    t_mean = float(tt.mean())
    a_mean = float(avg.mean())
    tc = tt - t_mean
    denom = float(tc @ tc)
    if denom <= 0.0:
        return None
    slope = float(tc @ (avg - a_mean)) / denom
    intercept = a_mean - slope * t_mean
    return intercept, slope


def linear_fractional_rms(y_win, fs: float) -> LinearRmsEstimate:
    """Fractional AC RMS with linear trend removal.

    Parameters
    ----------
    y_win :
        One analysis window.
    fs :
        Sample rate in Hz.

    Returns
    -------
    LinearRmsEstimate
        ``(rms, strategy)`` when iterated.
    """
    y = np.ascontiguousarray(y_win, dtype=np.float64)
    if y.ndim != 1:
        raise ValueError("y_win must be one-dimensional")
    if y.size == 0 or not np.isfinite(y).all():
        return LinearRmsEstimate(math.nan, "invalid")

    zc = _frac_zero_crossings(_rough_linear_detrend(y))
    if zc is None:
        return LinearRmsEstimate(_window_ac_rms(y), "window_ac")

    zero_a, zero_b, n_per = zc
    period = (zero_b - zero_a) / n_per
    if not math.isfinite(period) or period <= 1.0:
        return LinearRmsEstimate(_window_ac_rms(y), "window_ac")

    if y.size / period < 1.95:
        return LinearRmsEstimate(_window_ac_rms(y), "window_ac")

    fit = _linear_from_moving_average(y, period)
    if fit is None:
        return LinearRmsEstimate(_window_ac_rms(y), "window_ac")

    intercept, slope = fit

    # Refine period on the MA-detrended residual so a non-integer-period
    # tone cannot bias the rough linear detrend into a wrong period.
    t = np.arange(y.size, dtype=np.float64)
    zc2 = _frac_zero_crossings(y - (intercept + slope * t))
    if zc2 is not None:
        zero_a, zero_b, n_per = zc2
        period2 = (zero_b - zero_a) / n_per
        if math.isfinite(period2) and period2 > 1.0 and y.size / period2 >= 1.95:
            fit2 = _linear_from_moving_average(y, period2)
            if fit2 is not None:
                period = period2
                intercept, slope = fit2

    n_frac = zero_b - zero_a
    i0 = int(zero_a)
    if i0 < 0 or i0 + math.ceil(n_frac) > y.size:
        return LinearRmsEstimate(_window_ac_rms(y), "window_ac")

    need = int(math.ceil(n_frac))
    idx = np.arange(i0, i0 + need, dtype=np.float64)
    residual = y[i0 : i0 + need] - (intercept + slope * idx)
    rms = _frac_rms(residual, n_frac)
    return LinearRmsEstimate(
        rms=rms,
        strategy="linear_frac",
        frequency_hz=fs / period,
        period_samples=period,
    )
