"""Fractional RMS with an explicit drift model.

A window is accepted as periodic only when a lag makes the residual repeat
and the two halves of the window agree on that period. The drift removed
before the RMS is the highest order the window can actually support:

- cubic, when several cycles are visible
- linear, when only a couple of cycles are visible
- a constant, when about one cycle is visible

Anything else is the AC RMS of the whole window, and the result says why
the period model was not used. ``trustworthy`` is true only for an accepted
period model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

_FULL_BASIS: dict[tuple[int, int], np.ndarray] = {}


def _full_basis(n: int, order: int) -> np.ndarray:
    key = (n, order)
    cached = _FULL_BASIS.get(key)
    if cached is None:
        uu = (np.arange(n) - (n - 1) / 2) / n
        cached = np.column_stack([uu ** k for k in range(order + 1)])
        _FULL_BASIS[key] = cached
    return cached


def _interp_zero_crossing(y: np.ndarray, i: int) -> float:
    """Fractional index of the zero between samples i and i + 1."""
    y0 = float(y[i])
    y1 = float(y[i + 1])
    dy = y1 - y0
    if dy == 0.0:
        return float(i)
    return i - y0 / dy


def _period_hint(y: np.ndarray, fallback: float | None) -> float | None:
    centered = y - y.mean()
    crossings = np.flatnonzero(np.diff(np.signbit(centered)))
    n_per = (crossings.size - 1) // 2
    if n_per <= 0:
        return fallback
    zero_a = _interp_zero_crossing(centered, int(crossings[0]))
    zero_b = _interp_zero_crossing(centered, int(crossings[n_per * 2]))
    return (zero_b - zero_a) / n_per


def _frac_rms(y: np.ndarray, n: float) -> float:
    """RMS over n samples, with a fractional weight on the last sample."""
    n_int = int(n)
    frac = n - n_int
    total = float(np.sum(y[:n_int]))
    if frac > 0.0:
        total += float(y[n_int]) * frac
    avg = total / n
    sq = float(np.sum((y[:n_int] - avg) ** 2))
    if frac > 0.0:
        sq += (float(y[n_int]) - avg) ** 2 * frac
    return math.sqrt(sq / n)


def _moving_average(csum, y, i0, i1, stride, period, out, scratch):
    """Centered one-period moving average on the strided fit nodes.

    Samples sit on integers, so the fractional part of ``idx ± period/2``
    is constant along the stride. The prefix sum is then a pair of slices.
    """
    half = period * 0.5
    kr = math.floor(half)
    fr = half - kr
    if fr == 0.0:
        kl = kr
        fl = 0.0
    else:
        kl = kr + 1
        fl = 1.0 - fr

    right = slice(i0 + kr, i1 + kr, stride)
    left = slice(i0 - kl, i1 - kl, stride)
    cr = csum[right]
    cl = csum[left]
    m = cr.shape[0]
    dst = out[:m]
    np.subtract(cr, cl, out=dst)
    if fr != 0.0:
        scaled = scratch[:m]
        np.multiply(y[right], fr, out=scaled)
        dst += scaled
    if fl != 0.0:
        scaled = scratch[:m]
        np.multiply(y[left], fl, out=scaled)
        dst -= scaled
    dst *= 1.0 / period
    return dst


def _search_stride(period: float, n: int) -> int:
    """Step between samples used while locating the period.

    The lag cost is an average, and it is smooth in the lag. Comparing every
    few samples does not move the minimum when each cycle still contains
    several dozen of those samples. Short periods, and periods that leave
    only a short overlap, stay on every sample so the cost cannot alias.
    """
    stride = int(period // 32)
    if stride <= 1 or n - period < 64 * stride:
        return 1
    return min(stride, 4)


def _lag_mismatch(resid, period, err, scratch, stride: int = 1) -> float:
    """Mean-square change after a shift of one period.

    ``i + period`` has the same fractional part at every integer ``i``, so
    the shifted residual is a blend of two contiguous slices. ``stride``
    skips samples inside the search; the reported RMS still uses all of them.
    """
    p = float(period)
    n_tot = resid.shape[0]
    if not math.isfinite(p) or p <= 0.0:
        return np.inf
    k = math.floor(p)
    frac = p - k
    n = n_tot - k - 1
    if frac > 0.0:
        n -= 1
    if n <= 0:
        return np.inf

    if stride <= 1:
        a = resid[:n]
        b = resid[k : k + n]
        e = err[:n]
        if frac == 0.0:
            np.subtract(a, b, out=e)
        else:
            c = resid[k + 1 : k + 1 + n]
            scaled = scratch[:n]
            np.multiply(b, 1.0 - frac, out=scaled)
            np.multiply(c, frac, out=e)
            scaled += e
            np.subtract(a, scaled, out=e)
        return float(e @ e) / n

    a = resid[:n:stride]
    b = resid[k : k + n : stride]
    if a.size == 0:
        return np.inf
    if frac == 0.0:
        diff = a - b
    else:
        c = resid[k + 1 : k + 1 + n : stride]
        diff = a - (1.0 - frac) * b - frac * c
    return float(diff @ diff) / a.size


class _WindowModel:
    """Prefix sum, fit normal equations, and work buffers for one window."""

    def __init__(self, y: np.ndarray, order: int, fit_stride: int):
        self.y = y
        self.n = y.shape[0]
        self.order = order
        self.stride = fit_stride
        self.csum = np.empty(self.n + 1, dtype=np.float64)
        self.csum[0] = 0.0
        np.cumsum(y, out=self.csum[1:])
        self.center = (self.n - 1) / 2.0
        self.trend = np.empty(self.n, dtype=np.float64)
        self.resid = np.empty(self.n, dtype=np.float64)
        self.work = np.empty(self.n, dtype=np.float64)
        self.scratch = np.empty(self.n, dtype=np.float64)
        self._grid = None
        self._bt = None
        self._ata_inv = None

    def _prepare_grid(self, period: float) -> None:
        half = period * 0.5
        i0 = math.ceil(half) + 1
        i1 = math.floor(self.n - 1 - half)
        key = (i0, i1)
        if key == self._grid:
            return
        idx = np.arange(i0, i1, self.stride)
        self._grid = key
        if idx.size < self.order + 1:
            self._bt = None
            self._ata_inv = None
            return
        u = (idx - self.center) / self.n
        basis = np.column_stack([u ** k for k in range(self.order + 1)])
        self._bt = np.ascontiguousarray(basis.T)
        # Small normal matrix. An explicit inverse turns every later trial
        # into a matvec; the fit nodes only change when `period` crosses a
        # half-sample boundary. A short or bunched grid is singular: the
        # caller then drops this drift order instead of extrapolating it.
        try:
            self._ata_inv = np.linalg.inv(self._bt @ basis)
        except np.linalg.LinAlgError:
            self._bt = None
            self._ata_inv = None

    def coef(self, period: float) -> np.ndarray | None:
        self._prepare_grid(period)
        if self._ata_inv is None:
            return None
        moving = _moving_average(
            self.csum, self.y, self._grid[0], self._grid[1],
            self.stride, period, self.work, self.scratch,
        )
        if moving.size < self.order + 1:
            return None
        return self._ata_inv @ (self._bt @ moving)

    def fill_residual(self, period: float) -> np.ndarray | None:
        coef = self.coef(period)
        if coef is None:
            return None
        np.matmul(_full_basis(self.n, self.order), coef, out=self.trend)
        np.subtract(self.y, self.trend, out=self.resid)
        return self.resid

    def lag(self, resid: np.ndarray, period: float, stride: int = 1) -> float:
        return _lag_mismatch(resid, period, self.work, self.scratch, stride)

    def period_mismatch(self, period: float, stride: int = 1) -> float:
        resid = self.fill_residual(period)
        if resid is None:
            return 1e300
        cost = self.lag(resid, period, stride)
        if not math.isfinite(cost):
            return 1e300
        return cost


def _period_bounds(period: float, fs: float, half_hz: float) -> tuple[float, float]:
    freq = fs / period
    lo = fs / (freq + half_hz)
    hi = fs / max(freq - half_hz, 1.0)
    if lo > hi:
        lo, hi = hi, lo
    return lo, hi


def _parabolic_min(obj, lo: float, hi: float) -> float:
    """Minimum from three samples, kept only if the vertex is actually lower.

    With the drift held fixed the lag cost is quadratic in the fractional
    part of the period. A refitted drift barely moves across a fraction of
    a sample, so the same three-point fit matches the joint search there.
    """
    mid = 0.5 * (lo + hi)
    y_lo = obj(lo)
    y_mid = obj(mid)
    y_hi = obj(hi)
    denom = (lo - mid) * (lo - hi) * (mid - hi)
    a = (hi * (y_mid - y_lo) + mid * (y_lo - y_hi) + lo * (y_hi - y_mid)) / denom
    b = (
        hi * hi * (y_lo - y_mid)
        + mid * mid * (y_hi - y_lo)
        + lo * lo * (y_mid - y_hi)
    ) / denom
    points = ((y_lo, lo), (y_mid, mid), (y_hi, hi))
    if not math.isfinite(a) or a <= 0.0:
        return min(points)[1]
    vert = -b / (2.0 * a)
    if not math.isfinite(vert) or vert < lo or vert > hi or obj(vert) > min(points)[0]:
        return min(points)[1]
    return vert


def _refine_period(obj, period: float, fs: float, half_hz: float, xatol: float, max_period: float) -> float:
    lo, hi = _period_bounds(period, fs, half_hz)
    # A period has to leave a sample of overlap, or the lag cost is undefined.
    lo = max(lo, 2.0)
    hi = min(hi, max_period)
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
        return float(min(max(period, 2.0), max_period))
    # A bracket that stays between two integers does not cross the kink in
    # the lag cost, so a parabola is enough. Wider brackets use Brent.
    if math.floor(lo) == math.floor(hi) and lo != math.floor(lo):
        return _parabolic_min(obj, lo, hi)
    res = minimize_scalar(obj, bounds=(lo, hi), method="bounded", options={"xatol": xatol})
    return float(res.x)


@dataclass(frozen=True)
class RmsEstimate:
    """RMS of one window and the rule that produced it.

    ``method`` is the formula behind ``rms``: ``cubic``, ``quadratic``,
    ``linear``, or ``dc`` remove a drift of that order and integrate whole
    periods. ``window`` is the AC RMS of the whole window. ``invalid`` means
    the samples were empty or non-finite.

    ``reason`` is why that method was selected:

    - ``repeating``, ``few_cycles``, ``one_cycle``: a period was accepted and
      the drift order is the one the window can support
    - ``no_period``: not enough zero crossings, and no usable earlier period
    - ``no_signal``: the window is constant
    - ``aperiodic``: a lag was tried and the residual did not repeat
    - ``unsteady``: the two halves of the window disagree on the period
    - ``period_too_long``: the period leaves no room for a drift fit
    - ``invalid_input``

    ``trustworthy`` is true only when a period was accepted. A fallback RMS
    is still a defined number; the flag says it is not a confirmed
    periodic-component RMS.

    ``period_samples`` is what to pass as ``period_hint`` next. It is the
    refined period when one was accepted, the previous hint when this window
    had no crossings, and None when a period was rejected.

    Iterating yields ``rms``, ``frequency_hz``, ``period_samples``.
    """

    rms: float
    frequency_hz: float
    period_samples: float | None
    method: str
    trustworthy: bool
    reason: str

    def __iter__(self):
        yield self.rms
        yield self.frequency_hz
        yield self.period_samples


def _window_ac_rms(y: np.ndarray) -> float:
    if y.size == 0:
        return math.nan
    ac = y - y.mean()
    return math.sqrt(float(ac @ ac) / y.size)


def _fit_nodes(n: int, period: float, stride: int) -> int:
    if not math.isfinite(period) or period <= 0.0:
        return 0
    half = period * 0.5
    i0 = math.ceil(half) + 1
    i1 = math.floor(n - 1 - half)
    if i1 <= i0:
        return 0
    return (i1 - i0 + stride - 1) // stride


def _supported_order(n: int, period: float, stride: int, max_order: int) -> int | None:
    """Highest drift degree this period leaves enough room to observe.

    The moving average is known only on an interior span of about
    ``n - period`` samples, and a degree-d fit needs ``d + 1`` nodes there.
    A cubic also needs several cycles, so it is not extrapolated from one
    cycle in the middle of the window. A line needs a couple of cycles. One
    cycle can support a constant.
    """
    if not math.isfinite(period) or period <= 1.0 or period >= n:
        return None
    nodes = _fit_nodes(n, period, stride)
    cycles = n / period
    # Just under 4 cycles still has the moving average over most of the
    # window. The margin keeps a period a fraction of a sample long of
    # exactly four cycles from dropping to a line.
    if cycles >= 3.9 and nodes >= max_order + 1:
        return max_order
    # 1.95 rather than 2 so a period that lands a fraction of a sample
    # long of exactly two cycles still counts as two cycles.
    if max_order >= 1 and cycles >= 1.95 and nodes >= 2:
        return 1
    if cycles >= 1.0 and nodes >= 1:
        return 0
    return None


def _method_name(order: int) -> str:
    return {0: "dc", 1: "linear", 2: "quadratic", 3: "cubic"}.get(order, f"poly{order}")


def _method_reason(order: int) -> str:
    if order >= 2:
        return "repeating"
    if order == 1:
        return "few_cycles"
    return "one_cycle"


def _unsteady(y: np.ndarray, fs: float) -> bool:
    """True when the first and second half disagree on the period."""
    mid = y.size // 2
    if mid < 8:
        return False
    first = _period_hint(y[:mid], None)
    second = _period_hint(y[mid:], None)
    if first is None or second is None or first <= 0.0 or second <= 0.0:
        return False
    freq_a = fs / first
    freq_b = fs / second
    # A few percent across the window is a sweep, not a stationary tone.
    # The exponential drift in a normal record moves the halves by far less.
    return abs(freq_a - freq_b) > 0.02 * max(freq_a, freq_b)


def _lag_costs(model: _WindowModel, resid: np.ndarray, period: float, stride: int):
    """Lag cost at ``period`` and half a sample to either side.

    A tone is a dip: the center is far below both neighbors. Noise and a
    constant are flat across that half sample. A zero-crossing hint can sit
    on the side of the dip, so one neighbor being lower is not a rejection;
    the period search exists to walk downhill.
    """
    n = resid.shape[0]
    costs = []
    for delta in (0.0, -0.5, 0.5):
        trial = period + delta
        if not (1.0 < trial < n - 2):
            return None
        cost = model.lag(resid, trial, stride)
        if not math.isfinite(cost):
            return None
        costs.append(cost)
    return costs


def _flat_costs(costs) -> bool:
    return max(costs) <= 1.5 * max(min(costs), 1e-18)


def _is_dip(costs) -> bool:
    base, left, right = costs
    shoulder = min(left, right)
    if shoulder <= 1e-18:
        return False
    return base <= left and base <= right and base < 0.25 * shoulder


def _fallback(y: np.ndarray, reason: str, carry: float | None) -> RmsEstimate:
    return RmsEstimate(
        rms=_window_ac_rms(y),
        frequency_hz=math.nan,
        period_samples=carry,
        method="window",
        trustworthy=False,
        reason=reason,
    )


def _plausible_hint(period: float | None, n: int) -> float | None:
    if period is None or not math.isfinite(period):
        return None
    if 1.0 < period < 0.5 * n:
        return float(period)
    return None


def cubic_fractional_rms(
    y_win,
    fs: float,
    period_hint: float | None = None,
    *,
    order: int = 3,
    fit_stride: int = 4,
    coarse_half_hz: float = 0.6,
    fine_half_hz: float = 0.05,
    coarse_xatol: float = 1e-6,
    fine_xatol: float = 1e-9,
) -> RmsEstimate:
    """RMS of one analysis window, with the rule that produced it.

    Parameters
    ----------
    y_win :
        Samples for one window.
    fs :
        Sample rate in Hz.
    period_hint :
        Period in samples from the previous window. Used when this window
        has no zero crossing. Pass ``RmsEstimate.period_samples`` back in.
    order :
        Highest drift degree to attempt. Lower degrees are used when the
        window cannot support this one.
    """
    y = np.ascontiguousarray(y_win, dtype=np.float64)
    if y.ndim != 1:
        raise ValueError("y_win must be one-dimensional")
    if y.size == 0 or not np.isfinite(y).all():
        return RmsEstimate(math.nan, math.nan, None, "invalid", False, "invalid_input")

    carry = _plausible_hint(period_hint, y.size)
    if _window_ac_rms(y) == 0.0:
        return _fallback(y, "no_signal", carry)

    hint = _period_hint(y, period_hint)
    if hint is None or not math.isfinite(hint) or hint <= 1.0:
        return _fallback(y, "no_period", None)
    if _unsteady(y, fs):
        return _fallback(y, "unsteady", None)

    chosen = _supported_order(y.size, hint, fit_stride, order)
    if chosen is None:
        return _fallback(y, "period_too_long", None)

    model = _WindowModel(y, chosen, fit_stride)
    frozen = model.fill_residual(hint)
    if frozen is None:
        return _fallback(y, "period_too_long", None)
    coarse_stride = _search_stride(hint, y.size)
    hinted = _lag_costs(model, frozen, hint, coarse_stride)
    if hinted is None or _flat_costs(hinted):
        return _fallback(y, "aperiodic", None)

    max_period = y.size - 2.0
    period = _refine_period(
        lambda p, resid=frozen, stride=coarse_stride: (
            cost if math.isfinite(cost := model.lag(resid, p, stride)) else 1e300
        ),
        hint, fs, coarse_half_hz, coarse_xatol, max_period,
    )
    supported = _supported_order(y.size, period, fit_stride, order)
    if supported is None:
        return _fallback(y, "period_too_long", None)
    if supported != chosen:
        chosen = supported
        model = _WindowModel(y, chosen, fit_stride)
    fine_stride = _search_stride(period, y.size)
    period = _refine_period(
        lambda p, stride=fine_stride: model.period_mismatch(p, stride),
        period, fs, fine_half_hz, fine_xatol, max_period,
    )
    supported = _supported_order(y.size, period, fit_stride, order)
    if supported is None:
        return _fallback(y, "period_too_long", None)
    if supported != chosen:
        chosen = supported
        model = _WindowModel(y, chosen, fit_stride)

    resid = model.fill_residual(period)
    if resid is None:
        return _fallback(y, "period_too_long", None)
    refined = _lag_costs(model, resid, period, fine_stride)
    if refined is None or not _is_dip(refined):
        return _fallback(y, "aperiodic", None)

    n_per = int(y.size / period)
    if n_per <= 0:
        return _fallback(y, "period_too_long", None)
    return RmsEstimate(
        rms=_frac_rms(resid, n_per * period),
        frequency_hz=fs / period,
        period_samples=period,
        method=_method_name(chosen),
        trustworthy=True,
        reason=_method_reason(chosen),
    )
