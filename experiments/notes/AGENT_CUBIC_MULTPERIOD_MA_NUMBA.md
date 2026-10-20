# Agent task: multi-period centered MA for cubic RMS + Numba port

> WIP design note kept under `notes/`. Paths below are relative to the `rms/` package root.

## Goal

Apply the same moving-average improvement used in `linear_frac_rms.py` / `linear_frac_rms_numba.py` to the cubic drift estimator, then add a Numba implementation of the cubic algorithm.

Do **not** remove cubic’s decision/fallback API (`RmsEstimate` with `method`, `reason`, `trustworthy`). Only change how the drift moving average is formed, then port.

Reference implementations (read these first):

- `linear_frac_rms.py` — `_linear_from_moving_average` (centered multi-period fractional MA)
- `linear_frac_rms_numba.py` — Numba style, `fastmath=True`, finite-check in Python
- `cubic_frac_rms.py` — current cubic path (`_moving_average`, `_WindowModel`)
- `test_linear_frac_rms.py` / `test_linear_frac_rms_numba.py` — test patterns to mirror

---

## Part 1 — Multi-period centered fractional MA in `cubic_frac_rms.py`

### Current behavior

`_moving_average(...)` averages length **`period`** (one cycle), centered at each fit node, with fractional edges at `±period/2`:

- `half = period * 0.5`
- `kr = floor(half)`, `fr = half - kr`
- if `fr == 0`: `kl, fl = kr, 0`; else `kl, fl = kr+1, 1-fr`
- `avg = (csum[i+kr]-csum[i-kl] + fr*y[i+kr] - fl*y[i-kl]) / period`

`_WindowModel._prepare_grid(period)` and `coef(period)` pass that same `period` as the MA length, so the fit grid is also sized for a one-period half-width.

### Required change

At each trial period `P`, choose an averaging length of **`k` periods**:

```text
k = max(1, floor((n / P) / 4))
L = k * P
```

Use **`L`** (not `P`) as the moving-average window length everywhere the centered fractional sum is formed.

Keep using **`P`** as:

- the lag for periodicity / period search
- the span for fractional RMS over whole periods
- the value stored in `period_samples` / frequency

### Edge weights (centered)

At integer centre `i`, average from `i - L/2` to `i + L/2`.

When `L` is non-integer, **both** ends are partial. With `{L} = L - floor(L)`, if `{L}/2 < 0.5` those end weights are each ≈ `{L}/2`. Implement via the existing half-edge scheme with `half = L/2` (same as today, but `L` instead of `P`).

### Grid / fit nodes

`_prepare_grid` must size the interior from **`L`**, not `P`:

```text
half = L * 0.5
i0 = ceil(half) + 1
i1 = floor(n - 1 - half)
```

If `L` is so large that fewer than `order+1` nodes remain, the existing singular-fit / order-drop / fallback paths should still run (do not invent new strategies).

### Suggested API shape (minimal churn)

Prefer a small helper rather than scattering `k` logic:

```python
def _ma_length(n: int, period: float) -> float | None:
    """Centered MA length in samples: k periods, k = max(1, floor(n_cycles/4))."""
    if not math.isfinite(period) or period <= 1.0 or period >= n:
        return None
    k = max(1, int((n / period) // 4))
    length = k * period
    if length >= n:
        return None
    return length
```

Then:

- `_moving_average(..., length, ...)` divides by `length` and uses `half = length * 0.5`
- `_WindowModel.coef` / `_prepare_grid` take `period` for the tone, derive `length = _ma_length(n, period)`, build grid and MA from `length`

Update docstrings/module blurb to state the MA is `k` periods with `k = max(1, floor(n_cycles/4))`.

### Tests / checks for Part 1

1. Existing cubic self-test / notebook-style stream (exp decay + sine) still runs without exceptions.
2. On that signal, trustworthy cubic windows remain the norm; median |error| should stay at ppm-class (do not regress by orders of magnitude).
3. Add or extend a unit test that for a window with ~25 cycles, `_ma_length` yields `k == 6` and `L ≈ 6*P`.
4. Compare briefly to one-period MA if easy: late flat windows should not get *worse* by a large factor; early curved windows should stay strong.

Keep the public function name `cubic_fractional_rms` and `RmsEstimate` fields unchanged.

---

## Part 2 — Numba port: `cubic_frac_rms_numba.py`

### Scope

New file alongside `cubic_frac_rms.py`. Same public result idea:

- Prefer returning the same `RmsEstimate` dataclass (import from `cubic_frac_rms` **or** duplicate a compatible frozen dataclass — importing is fine if it avoids drift).
- Public entry: `cubic_fractional_rms(y_win, fs, period_hint=None, **kwargs) -> RmsEstimate`
- Optional `warmup(n, fs)` to compile kernels.

### What to JIT

Numba does **not** like SciPy `minimize_scalar` inside `njit`. Structure as:

1. **Python orchestration** (thin): validate input, `ascontiguousarray`, finite check (`np.isfinite` in Python — `fastmath` breaks NaN checks), call JIT core, map codes → `RmsEstimate`.
2. **JIT core** for the hot numeric path with the **multi-period MA** from Part 1 already applied.

Minimum viable JIT surface (mirror `linear_frac_rms_numba.py` style):

- zero-crossing / period hint
- centered multi-period MA + polynomial LS fit for a fixed order (or order chosen in Python then passed in)
- residual fill
- lag mismatch
- fractional RMS
- dip / flat checks if they stay simple arithmetic

Period refinement:

- **Option A (preferred for parity):** keep Brent/`minimize_scalar` in Python; each objective evaluation calls a small `@njit` function `period_mismatch(y, period, order, ...)`.
- **Option B:** replace fine search with a coarse grid + parabolic refine in pure njit (document any accuracy tradeoff).

Use `@njit(cache=True, fastmath=True)` for kernels. Do finite/empty checks in Python before entering JIT.

### Behavioral parity

Match `cubic_frac_rms.py` decisions as closely as practical:

- same strategy/reason strings
- same `trustworthy` rules
- same multi-period MA length rule
- RMS within ~1e-9–1e-12 of NumPy on clean tones (loosen slightly for `fastmath` on near-null residuals)

### Tests: `test_cubic_frac_rms_numba.py`

Mirror `test_linear_frac_rms_numba.py`:

- module-scoped `warmup`
- parametrize DC, slow sine+drift, fast sine, short cycles, sweep, ramp, empty, NaN
- assert strategy/method+reason match NumPy cubic where both take the main path
- assert RMS close on trustworthy windows
- reject 2-D input in the Python wrapper

Also add a smoke bench under `if __name__ == "__main__"` comparing ms/window NumPy vs Numba on the 15 s exp+sine stream (16384 windows), after warmup.

### Dependencies

- Require `numba` (already used by linear numba module).
- Keep `scipy` for the NumPy cubic file; Numba file should avoid SciPy inside JIT (Python-side refine is OK).

---

## Part 3 — Implementation order

1. Patch `cubic_frac_rms.py` MA length + grid (`Part 1`); run existing cubic checks.
2. Add `cubic_frac_rms_numba.py` with MA rule identical to Part 1.
3. Add `test_cubic_frac_rms_numba.py`; run linear + cubic tests.
4. Short note in the cubic module docstring (and optionally notebook) that drift MA length is `k` periods.

---

## Out of scope

- Removing cubic fallbacks / `trustworthy` / reason strings  
- Porting the notebook cells unless asked  
- Module-level buffer pools (linear deliberately dropped those for simplicity)  
- Changing the linear files except if a shared helper is clearly useful (prefer duplicating the small `_ma_length` rule to avoid coupling)

---

## Acceptance criteria

- [ ] Cubic NumPy uses centered MA of length `k*P`, `k = max(1, floor(n_cycles/4))`
- [ ] Fit grid half-width uses `L`, lag/RMS still use `P`
- [ ] `cubic_frac_rms_numba.py` exists, warms up, matches NumPy on core sine+drift cases
- [ ] Tests added and passing
- [ ] Bench printed: Numba faster than NumPy cubic on the standard stream (exact factor unconstrained)
