# `dsp.rms` — accurate fractional RMS

AC RMS on a fixed analysis window is often wrong for real periodic signals:

1. **Windowing** — the window usually holds a non-integer number of periods, so a naive RMS includes a partial cycle.
2. **Slow drift** — an unsettled AC-coupling capacitor (or similar) adds a decaying DC offset that leaks into the AC RMS.

This package detects a period, removes a slow polynomial drift, then computes RMS over a **fractional** span of whole periods (partial weight on the last sample). When the period model is not trustworthy, it falls back to plain window AC RMS and reports why.

```python
from dsp.rms import linear_fractional_rms, cubic_fractional_rms
from dsp.rms.linear_numba import linear_fractional_rms as linear_fractional_rms_numba, warmup
```

## Algorithms

### Linear drift + fractional RMS (`dsp.rms.linear`)

1. Rough linear detrend → zero crossings → initial period.
2. Centered moving average of length \(L = k \cdot P\) periods, with  
   \(k = \max(1, \lfloor n_\text{cycles}/4 \rfloor)\). Content that repeats at period \(P\) cancels; the average tracks the drift.
3. Least-squares line on that moving average; re-take zero crossings on the residual and refit the line (avoids period bias when a non-integer-period tone correlates with the rough detrend).
4. Fractional AC RMS from the first to last zero crossing (whole periods).

**Strategies** (`LinearRmsEstimate.strategy`):

| Strategy | Meaning |
|----------|---------|
| `linear_frac` | Period accepted; linear drift removed; fractional RMS |
| `window_ac` | Fallback: AC RMS of the whole window |
| `invalid` | Empty or non-finite input |

Numba twin: `dsp.rms.linear_numba` — same algorithm and result shape, JIT hot path.

### Adaptive-order drift + fractional RMS (`dsp.rms.cubic`)

Same idea as the linear path, but the drift order is whatever the window can support:

| Cycles in window | Drift model |
|------------------|-------------|
| several (~4+) | cubic |
| a couple (~2+) | linear |
| about one | constant (DC) |

The period is the lag that makes the residual **repeat**, with a check that both halves of the window agree. Only accepted period models set `trustworthy=True`.

**Result** (`RmsEstimate`): `rms`, `frequency_hz`, `period_samples`, `method`, `trustworthy`, `reason`.  
Iterating yields `(rms, frequency_hz, period_samples)` so you can carry `period_samples` as the next window’s `period_hint`.

## Design notes (short)

- Fractional RMS: integrate energy and mean over \(n = n_\text{int} + f\) samples with weight \(f\) on the last sample, then \(\sqrt{\mathbb{E}[y^2] - (\mathbb{E}[y])^2}\).
- Moving-average length scales with available cycles so late, nearly flat windows still get a stable drift estimate without drowning early curved windows.
- Fallbacks are intentional: a number is always returned when the input is finite; `strategy` / `trustworthy` / `reason` tell you whether to trust it as a periodic-component RMS.

Longer write-ups live in `experiments/notes/`.
