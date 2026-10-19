"""Tests for the Numba linear fractional RMS path."""

from __future__ import annotations

import math

import numpy as np
import pytest

from linear_frac_rms import linear_fractional_rms as numpy_rms
from linear_frac_rms_numba import linear_fractional_rms as numba_rms
from linear_frac_rms_numba import warmup

FS = 102400.0
N = 16384
ACCURATE = 1.0


@pytest.fixture(scope="module", autouse=True)
def _compile_kernels():
    warmup(N, FS)


def _sine(
    freq: float = 1000.0 / (2 * np.pi),
    rms: float = ACCURATE,
    phase: float = 0.1,
    dc: float = 0.0,
    slope: float = 0.0,
) -> np.ndarray:
    t = np.arange(N) / FS
    return (
        dc
        + slope * np.arange(N)
        + np.sqrt(2.0) * rms * np.sin(2 * np.pi * freq * t + phase)
    )


@pytest.mark.parametrize(
    "y",
    [
        np.full(N, 3.5),
        _sine(dc=5.0, slope=-5.0 / N),
        _sine(freq=5000.0, dc=1.0, slope=2.0 / N),
        _sine(freq=2.2 * FS / N, dc=0.5),
        _sine(freq=1.1 * FS / N),
        np.linspace(-2.0, 5.0, N),
        1e-6 * np.random.default_rng(0).standard_normal(N),
        np.array([]),
        np.array([1.0, math.nan, 2.0]),
    ],
    ids=[
        "dc",
        "slow_sine",
        "fast_sine",
        "two_cycle",
        "one_cycle",
        "ramp",
        "tiny_noise",
        "empty",
        "nan",
    ],
)
def test_matches_numpy(y):
    a = numpy_rms(y, FS)
    b = numba_rms(y, FS)
    assert a.strategy == b.strategy
    if math.isnan(a.rms):
        assert math.isnan(b.rms)
    else:
        # fastmath can leave tiny residuals on near-null signals (e.g. ramp).
        tol = 1e-9 * max(1.0, abs(a.rms), abs(b.rms))
        assert abs(a.rms - b.rms) <= tol


def test_slow_sine_accuracy():
    est = numba_rms(_sine(dc=5.0, slope=-5.0 / N), FS)
    assert est.strategy == "linear_frac"
    assert abs(est.rms - ACCURATE) / ACCURATE * 1e6 < 50.0


def test_rejects_2d():
    with pytest.raises(ValueError, match="one-dimensional"):
        numba_rms(np.zeros((4, 4)), FS)
