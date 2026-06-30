"""Archived half-map T/V decomposition for legacy validation scripts."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from ..local_stats import gradient_magnitude


def _uf(x: np.ndarray, size: int) -> np.ndarray:
    return ndimage.uniform_filter(np.asarray(x, dtype=np.float64), size=int(size), mode="nearest")


def _match_dtype(volume: np.ndarray, arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr, dtype=volume.dtype)


def _maybe_window(arr: np.ndarray, window: int) -> np.ndarray:
    w = int(window)
    if w <= 1:
        return np.asarray(arr, dtype=np.float64)
    return _uf(arr, w)


def windowed_halfmap_disagreement(
    delta_rho: np.ndarray,
    *,
    window: int = 5,
    sigma: float = 1.0,
    eps: float = 1e-12,
) -> np.ndarray:
    """Windowed half-map disagreement: (1 / (2 sigma^2)) * mean_W(delta_rho^2)."""
    d = np.asarray(delta_rho)
    sig2 = float(sigma) ** 2 + eps
    t_raw = 0.5 * (1.0 / sig2) * np.asarray(d, dtype=np.float64) ** 2
    return _match_dtype(d, _maybe_window(t_raw, window))


def lh_decomposition(
    rho: np.ndarray,
    delta_rho: np.ndarray,
    *,
    window: int = 1,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> dict[str, np.ndarray]:
    """Legacy T/V/L/H bundle for archive scripts only."""
    if rho.shape != delta_rho.shape:
        raise ValueError(f"Shape mismatch: rho {rho.shape} vs delta_rho {delta_rho.shape}")
    v = np.asarray(rho)
    a, b = float(alpha), float(beta)
    d = np.asarray(delta_rho, dtype=np.float64)

    t_raw = 0.5 * a * (d * d)
    grad_sq = gradient_magnitude(v) ** 2
    v_raw = 0.5 * b * grad_sq

    t = _maybe_window(t_raw, window)
    pot = _maybe_window(v_raw, window)
    lagrangian = t - pot
    hamiltonian = t + pot

    return {
        "halfmap_disagreement": _match_dtype(v, t),
        "smoothness": _match_dtype(v, pot),
        "L_balance": _match_dtype(v, lagrangian),
        "H_sum": _match_dtype(v, hamiltonian),
    }


def classify_tv_regime(
    halfmap_disagreement: np.ndarray,
    smoothness: np.ndarray,
    mask: np.ndarray,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """Three-way phase portrait relative to in-mask medians (archive only)."""
    t = np.asarray(halfmap_disagreement, dtype=np.float64)
    v = np.asarray(smoothness, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    zones = np.zeros(t.shape, dtype=np.uint8)
    if not m.any():
        return zones

    t_med = float(np.median(t[m]))
    v_med = float(np.median(v[m]))
    t_hi = t > t_med
    v_hi = v > v_med

    featureless = m & (~t_hi) & (~v_hi)
    flexible = m & t_hi & (~v_hi)
    rigid = m & (~t_hi) & v_hi
    both_hi = m & t_hi & v_hi

    zones[featureless] = 0
    zones[flexible] = 1
    zones[rigid] = 2
    if both_hi.any():
        zones[both_hi] = np.where(t[both_hi] >= v[both_hi], 1, 2).astype(np.uint8)

    return zones
