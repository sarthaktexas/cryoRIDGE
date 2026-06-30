"""Density source for reliability / feature pipeline."""

from __future__ import annotations

from typing import Literal

import numpy as np

DensitySource = Literal["avg_half", "primary"]


def zscore_global(volume: np.ndarray) -> np.ndarray:
    v = np.asarray(volume, dtype=np.float64)
    mu = float(v.mean())
    sig = float(v.std())
    return ((v - mu) / (sig + 1e-6)).astype(np.float32)


def zscore_halfmap_average(half1: np.ndarray, half2: np.ndarray) -> np.ndarray:
    """Global z-score of ρ = ½(h₁+h₂) for smoothness reliability ranking."""
    rho = 0.5 * (np.asarray(half1, dtype=np.float32) + np.asarray(half2, dtype=np.float32))
    return zscore_global(rho)


def rho_normalized_for_reliability(
    *,
    source: DensitySource,
    half1: np.ndarray,
    half2: np.ndarray,
    features_density_normalized: np.ndarray | None = None,
    primary_volume: np.ndarray | None = None,
) -> np.ndarray:
    """
    ρ for smoothness reliability ranking.

    ``avg_half`` (default): z-score(½(h₁+h₂)), matched to half-map CC validation.
    ``primary``: z-scored deposited map from feature NPZ or raw primary volume (sensitivity).
    """
    if source == "primary":
        if features_density_normalized is not None:
            return np.asarray(features_density_normalized, dtype=np.float32)
        if primary_volume is None:
            raise ValueError("primary density source requires features NPZ or primary_volume")
        return zscore_global(primary_volume)
    return zscore_halfmap_average(half1, half2)


# Deprecated alias.
rho_normalized_for_lh = rho_normalized_for_reliability
