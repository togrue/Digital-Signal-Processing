# Accurate fractional RMS

AC RMS on a fixed analysis window is often wrong for real periodic signals:

1. **Windowing** — the window usually holds a non-integer number of periods, so a naive RMS includes a partial cycle.
2. **Slow drift** — an unsettled AC-coupling capacitor (or similar) adds a decaying DC offset that leaks into the AC RMS.

This package detects a period, removes a slow polynomial drift, then computes RMS over a **fractional** span of whole periods (partial weight on the last sample). When the period model is not trustworthy, it falls back to plain window AC RMS and reports why.

Typical use case: high sample-rate measurement streams (e.g. audio / vibration / instrumentation) where you want ppm-class RMS on tones sitting on a slow exponential or linear baseline.

## Algorithms

### Linear drift + fractional RMS (`linear_frac_rms`)

1. Rough linear detrend → zero crossings → period estimate.
2. Centered moving average of length \(L = k \cdot P\) periods, with  
   \(k = \max(1, \lfloor n_\text{cycles}/4 \rfloor)\). Content that repeats at period \(P\) cancels; the average tracks the drift.
3. Least-squares line on that moving average; subtract from the window.
4. Fractional AC RMS from the first to last zero crossing (whole periods).

**Strategies** (`LinearRmsEstimate.strategy`):

| Strategy | Meaning |
|----------|---------|
| `linear_frac` | Period accepted; linear drift removed; fractional RMS |
| `window_ac` | Fallback: AC RMS of the whole window |
| `invalid` | Empty or non-finite input |

Numba twin: `linear_frac_rms_numba` — same algorithm and result shape, JIT hot path.

### Adaptive-order drift + fractional RMS (`cubic_frac_rms`)

Same idea as the linear path, but the drift order is whatever the window can support:

| Cycles in window | Drift model |
|------------------|-------------|
| several (~4+) | cubic |
| a couple (~2+) | linear |
| about one | constant (DC) |

The period is the lag that makes the residual **repeat**, with a check that both halves of the window agree. Only accepted period models set `trustworthy=True`.

**Result** (`RmsEstimate`): `rms`, `frequency_hz`, `period_samples`, `method`, `trustworthy`, `reason`.  
Iterating yields `(rms, frequency_hz, period_samples)` so you can carry `period_samples` as the next window’s `period_hint`.

## Quick start

```python
import numpy as np
from linear_frac_rms import linear_fractional_rms
from cubic_frac_rms import cubic_fractional_rms

fs = 102_400.0
# y_win: 1-D float array, one analysis window
est = linear_fractional_rms(y_win, fs)
print(est.rms, est.strategy, est.frequency_hz)

# Optional: reuse refined period across windows
period = None
for y_win in windows:
    est = cubic_fractional_rms(y_win, fs, period_hint=period)
    period = est.period_samples
    if est.trustworthy:
        print(est.rms, est.method, est.frequency_hz)
```

Faster linear path (after first compile):

```python
from linear_frac_rms_numba import linear_fractional_rms, warmup

warmup()  # optional; compiles kernels once
est = linear_fractional_rms(y_win, fs)
```

## Layout

```
rms/
├── linear_frac_rms.py          # NumPy linear drift + fractional RMS
├── linear_frac_rms_numba.py    # Numba port of the linear path
├── cubic_frac_rms.py           # Adaptive-order (cubic/linear/dc) path
├── test_linear_frac_rms.py
├── test_linear_frac_rms_numba.py
├── _profile_linear_numba.py    # micro-benchmark helper
├── AccurateRMS.ipynb           # exploration notebook (signal model + plots)
├── requirements.txt
└── notes/                      # design / WIP notes (not required to use the library)
```

## Dependencies

- **Required:** Python 3.10+, NumPy
- **Cubic path:** SciPy (`scipy.optimize.minimize_scalar`)
- **Numba path / tests:** Numba, pytest
- **Notebook:** Matplotlib (and SciPy for the FIR AC-coupling demo)

```bash
pip install -r requirements.txt
```

## Tests & smoke benches

From this directory:

```bash
pytest -q
python linear_frac_rms.py          # timing + ppm on 15 s exp+sine stream
python linear_frac_rms_numba.py
python cubic_frac_rms.py
```

## Design notes (short)

- Fractional RMS: integrate energy and mean over \(n = n_\text{int} + f\) samples with weight \(f\) on the last sample, then \(\sqrt{\mathbb{E}[y^2] - (\mathbb{E}[y])^2}\).
- Moving-average length scales with available cycles so late, nearly flat windows still get a stable drift estimate without drowning early curved windows.
- Fallbacks are intentional: a number is always returned when the input is finite; `strategy` / `trustworthy` / `reason` tell you whether to trust it as a periodic-component RMS.

See `notes/` for longer agent/design write-ups (e.g. multi-period MA + planned Numba cubic port).

## License

Add a license before publishing if you intend others to reuse this code.
