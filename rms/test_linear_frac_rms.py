"""Tests for linear_frac_rms."""

from __future__ import annotations

import math

import numpy as np
import pytest

from linear_frac_rms import LinearRmsEstimate, linear_fractional_rms


FS = 102400.0
N = 16384
ACCURATE = 1.0


def _sine(
    n: int = N,
    fs: float = FS,
    freq: float = 1000.0 / (2 * np.pi),
    rms: float = ACCURATE,
    phase: float = 0.1,
    dc: float = 0.0,
    slope: float = 0.0,
) -> np.ndarray:
    t = np.arange(n) / fs
    return (
        dc
        + slope * np.arange(n)
        + np.sqrt(2.0) * rms * np.sin(2 * np.pi * freq * t + phase)
    )


def _sweep(n: int = N, fs: float = FS, f0: float = 50.0, f1: float = 4000.0) -> np.ndarray:
    t = np.arange(n) / fs
    phase = 2 * np.pi * (f0 * t + 0.5 * (f1 - f0) / t[-1] * t * t)
    return np.sqrt(2.0) * ACCURATE * np.sin(phase)


def test_invalid_empty():
    est = linear_fractional_rms(np.array([]), FS)
    assert est.strategy == "invalid"
    assert math.isnan(est.rms)


def test_invalid_nan():
    est = linear_fractional_rms(np.array([1.0, math.nan, 2.0]), FS)
    assert est.strategy == "invalid"
    assert math.isnan(est.rms)


def test_rejects_2d():
    with pytest.raises(ValueError, match="one-dimensional"):
        linear_fractional_rms(np.zeros((4, 4)), FS)


def test_dc_fallback():
    est = linear_fractional_rms(np.full(N, 3.5), FS)
    assert est.strategy == "window_ac"
    assert est.rms == 0.0


def test_slow_sine_with_drift():
    y = _sine(dc=5.0, slope=-5.0 / N)
    est = linear_fractional_rms(y, FS)
    assert est.strategy == "linear_frac"
    assert abs(est.rms - ACCURATE) / ACCURATE * 1e6 < 50.0
    assert est.period_samples is not None
    assert math.isfinite(est.frequency_hz)


def test_fast_sine_with_drift():
    y = _sine(freq=5000.0, dc=1.0, slope=2.0 / N)
    est = linear_fractional_rms(y, FS)
    assert est.strategy == "linear_frac"
    assert abs(est.rms - ACCURATE) / ACCURATE * 1e6 < 50.0


def test_two_cycle_does_not_crash():
    y = _sine(freq=2.2 * FS / N, dc=0.5)
    est = linear_fractional_rms(y, FS)
    assert est.strategy in {"linear_frac", "window_ac"}
    assert math.isfinite(est.rms)


def test_one_cycle_fallback_or_estimate():
    y = _sine(freq=1.1 * FS / N)
    est = linear_fractional_rms(y, FS)
    assert est.strategy in {"linear_frac", "window_ac"}
    assert math.isfinite(est.rms)


def test_sweep_does_not_crash():
    est = linear_fractional_rms(_sweep(), FS)
    assert est.strategy in {"linear_frac", "window_ac"}
    assert math.isfinite(est.rms)


def test_ramp_near_zero_ac():
    est = linear_fractional_rms(np.linspace(-2.0, 5.0, N), FS)
    assert est.strategy in {"linear_frac", "window_ac"}
    assert math.isfinite(est.rms)
    # After a successful linear removal a pure ramp has negligible AC.
    if est.strategy == "linear_frac":
        assert est.rms < 1e-9


def test_tiny_noise_does_not_crash():
    y = 1e-6 * np.random.default_rng(0).standard_normal(N)
    est = linear_fractional_rms(y, FS)
    assert est.strategy in {"linear_frac", "window_ac"}
    assert math.isfinite(est.rms)


def test_unpack_rms_strategy():
    est = linear_fractional_rms(_sine(), FS)
    rms, strategy = est
    assert rms == est.rms
    assert strategy == est.strategy
    assert isinstance(est, LinearRmsEstimate)


def test_phase_stability_slow_sine():
    """RMS should stay accurate across random phases."""
    rng = np.random.default_rng(1)
    errors = []
    for _ in range(40):
        phase = float(rng.uniform(0, 2 * np.pi))
        y = _sine(phase=phase, dc=3.0, slope=-4.0 / N)
        est = linear_fractional_rms(y, FS)
        assert est.strategy == "linear_frac"
        errors.append(abs(est.rms - ACCURATE) / ACCURATE * 1e6)
    assert float(np.median(errors)) < 20.0
    assert float(np.max(errors)) < 200.0


def test_float32_input_accepted():
    y = _sine(dc=1.0, slope=-1.0 / N).astype(np.float32)
    est = linear_fractional_rms(y, FS)
    assert est.strategy == "linear_frac"
    assert abs(est.rms - ACCURATE) / ACCURATE * 1e6 < 100.0


def test_repeated_calls_stable():
    y = _sine(dc=2.0, slope=-3.0 / N)
    a = linear_fractional_rms(y, FS)
    b = linear_fractional_rms(y, FS)
    assert a.strategy == b.strategy == "linear_frac"
    assert a.rms == b.rms
    assert a.period_samples == b.period_samples


def test_window_stream_accuracy():
    """Notebook-style sliding windows on exp-decay + sine."""
    duration = 2.0
    t = np.arange(0, duration, 1 / FS)
    y = 10 * np.exp(-t / 2.5) + np.sqrt(2.0) * ACCURATE * np.sin(
        2 * np.pi * (1000.0 / (2 * np.pi)) * t + 0.1
    )
    errors = []
    for start in range(0, y.size - N + 1, N):
        est = linear_fractional_rms(y[start : start + N], FS)
        assert est.strategy == "linear_frac"
        errors.append(abs(est.rms - ACCURATE) / ACCURATE * 1e6)
    assert float(np.median(errors)) < 50.0
    assert float(np.max(errors)) < 500.0
