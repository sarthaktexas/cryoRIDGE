"""Tests for windowed smoothness reliability."""

from __future__ import annotations

import numpy as np

from cryoem_mrc.reliability import windowed_smoothness


def _smooth_blob(n: int = 32) -> np.ndarray:
    z, y, x = np.ogrid[:n, :n, :n]
    c = (n - 1) / 2.0
    r2 = (z - c) ** 2 + (y - c) ** 2 + (x - c) ** 2
    return np.exp(-r2 / (2 * 4.0**2)).astype(np.float32)


def test_windowed_smoothness_positive_on_structured_blob() -> None:
    rho = _smooth_blob()
    v = windowed_smoothness(rho, window=5)
    assert v.shape == rho.shape
    assert float(v.mean()) > 0.0
    assert np.isfinite(v).all()
