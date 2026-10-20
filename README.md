# Digital Signal Processing

Reference implementations of DSP algorithms, packaged as an installable Python library.

## Install

```bash
pip install -e ".[dev]"          # library + tests + Numba + matplotlib
# or
pip install -e .                 # core only (NumPy + SciPy)
pip install -e ".[numba]"        # + Numba acceleration for the linear RMS path
```

## Quick start

```python
import numpy as np
from dsp.rms import linear_fractional_rms, cubic_fractional_rms

fs = 102_400.0
# y_win: 1-D float array, one analysis window
est = linear_fractional_rms(y_win, fs)
print(est.rms, est.strategy, est.frequency_hz)

period = None
for y_win in windows:
    est = cubic_fractional_rms(y_win, fs, period_hint=period)
    period = est.period_samples
    if est.trustworthy:
        print(est.rms, est.method, est.frequency_hz)
```

Faster linear path (optional Numba extra):

```python
from dsp.rms.linear_numba import linear_fractional_rms, warmup

warmup()
est = linear_fractional_rms(y_win, fs)
```

## Layout

```
├── src/dsp/                 # installable package
│   └── rms/                 # accurate fractional AC RMS
│       ├── linear.py        # NumPy linear drift + fractional RMS
│       ├── linear_numba.py  # Numba port of the linear path
│       └── cubic.py         # adaptive-order (cubic/linear/dc) path
├── tests/                   # pytest suite
├── experiments/             # benches, notebook, design notes
└── pyproject.toml
```

## Packages

| Import | Description |
|--------|-------------|
| [`dsp.rms`](src/dsp/rms/) | Accurate AC RMS for periodic signals: fractional periods and polynomial drift removal (linear / cubic), with optional Numba acceleration |

See [`src/dsp/rms/`](src/dsp/rms/) module docstrings and `experiments/` for API details and exploration.

## Tests & benches

```bash
pytest -q
python experiments/bench_linear.py
python experiments/bench_cubic.py
python experiments/bench_linear_numba.py   # needs [numba]
```
