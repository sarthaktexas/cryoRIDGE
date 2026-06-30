"""Map reliability scores for model-building guidance (build / caution / omit zones).

Windowed squared-gradient smoothness from globally z-scored half-map average rho_tilde.
Primary ranked export: reliability_score (in-mask percentile rank of smoothness).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy import ndimage

from style.palette import BUILD_ZONE_COLORS

from .io import save_volume_like_reference
from .local_stats import gradient_magnitude

if TYPE_CHECKING:
    from matplotlib.colors import ListedColormap

BUILD_ZONE_LABELS: dict[int, str] = {0: "omit", 1: "caution", 2: "build"}


def build_zone_colormap() -> ListedColormap:
    """ListedColormap for zone values 0/1/2 (omit / caution / build)."""
    from matplotlib.colors import ListedColormap

    return ListedColormap([BUILD_ZONE_COLORS[z] for z in (0, 1, 2)])


def windowed_smoothness(
    rho: np.ndarray,
    *,
    window: int = 5,
    kappa: float = 1.0,
) -> np.ndarray:
    """
    Windowed smoothness map: (kappa/2) * mean_W(||grad rho||^2).

    ``rho`` must be globally z-scored half-map average (``density_normalized``).
    Higher values indicate sharper local density structure (used for reliability ranking).
    """
    v = np.asarray(rho)
    grad_sq = gradient_magnitude(v) ** 2
    raw = 0.5 * float(kappa) * grad_sq
    w = int(window)
    if w <= 1:
        smoothed = np.asarray(raw, dtype=np.float64)
    else:
        smoothed = ndimage.uniform_filter(np.asarray(raw, dtype=np.float64), size=w, mode="nearest")
    return np.asarray(smoothed, dtype=v.dtype)


def percentile_rank_in_mask(
    volume: np.ndarray,
    mask: np.ndarray,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Map in-mask voxels to (0, 1] by rank; outside mask = 0.

    Higher values = higher rank among macromolecular voxels.
    """
    v = np.asarray(volume, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    out = np.zeros_like(v, dtype=np.float32)
    if not m.any():
        return out
    vals = v[m]
    order = np.argsort(vals, kind="mergesort")
    ranks = np.empty_like(vals, dtype=np.float64)
    ranks[order] = np.arange(1, vals.size + 1, dtype=np.float64)
    out[m] = (ranks / (vals.size + eps)).astype(np.float32)
    return out


def compute_reliability_maps(
    rho: np.ndarray,
    *,
    window: int = 5,
    kappa: float = 1.0,
    mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """
    Reliability maps from z-scored averaged-map density ``rho`` (rho_tilde).

    **Primary export:** ``reliability_score`` — in-mask percentile rank of
    ``reliability_smoothness`` (higher = more reliable).
    """
    smooth = np.asarray(
        windowed_smoothness(rho, window=window, kappa=kappa),
        dtype=np.float32,
    )
    out: dict[str, np.ndarray] = {"reliability_smoothness": smooth}
    if mask is not None:
        out["reliability_score"] = percentile_rank_in_mask(smooth, mask)
    else:
        out["reliability_score"] = smooth
    return out


def classify_build_zones(
    reliability_score: np.ndarray,
    mask: np.ndarray,
    *,
    build_pct: float = 66.67,
    caution_pct: float = 33.33,
) -> np.ndarray:
    """
    Discrete zone labels inside ``mask`` (uint8):

    - 0 = omit / low confidence (below ``caution_pct``)
    - 1 = caution (middle tercile by default)
    - 2 = build with confidence (top tercile by default)

    Outside mask = 0.
    """
    r = np.asarray(reliability_score, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    zones = np.zeros(r.shape, dtype=np.uint8)
    if not m.any():
        return zones
    vals = r[m]
    t_lo = float(np.percentile(vals, caution_pct))
    t_hi = float(np.percentile(vals, build_pct))
    inside = m.copy()
    zones[inside & (r < t_lo)] = 0
    zones[inside & (r >= t_lo) & (r < t_hi)] = 1
    zones[inside & (r >= t_hi)] = 2
    return zones


def attach_reliability_to_features(
    features: dict[str, np.ndarray],
    half1: np.ndarray,
    half2: np.ndarray,
    *,
    window: int = 5,
    kappa: float = 1.0,
    mask: np.ndarray | None = None,
    compute_zones: bool = True,
) -> dict[str, np.ndarray]:
    """
    Add reliability maps to a feature dict (requires ``density_normalized``).

    Half-maps are validated for grid alignment but are not used in the
    smoothness reliability path.

    Returns the same dict, updated in place and also returned for chaining.
    """
    if "density_normalized" not in features:
        raise KeyError("features must contain density_normalized")
    if half1.shape != half2.shape:
        raise ValueError(f"Half-map shape mismatch: {half1.shape} vs {half2.shape}")
    rho = np.asarray(features["density_normalized"])
    if rho.shape != half1.shape:
        raise ValueError(f"Feature grid {rho.shape} != half-map grid {half1.shape}")
    rel = compute_reliability_maps(rho, window=window, kappa=kappa, mask=mask)
    features.update(rel)
    if compute_zones and mask is not None:
        features["build_zone"] = classify_build_zones(rel["reliability_score"], mask)
    return features


def save_reliability_mrc(
    reference_path: str | Path,
    reliability: np.ndarray,
    out_path: str | Path | None = None,
    *,
    label: str = "reliability_score (cryoem_mrc)",
) -> Path:
    """Write reliability volume on the reference grid."""
    reference_path = Path(reference_path)
    if out_path is None:
        out_path = reference_path.with_name(f"{reference_path.stem}_reliability.mrc")
    else:
        out_path = Path(out_path)
    save_volume_like_reference(
        reference_path, reliability, out_path, dtype=np.float32, extra_label=label[:80]
    )
    return out_path


def save_build_zone_mrc(
    reference_path: str | Path,
    zones: np.ndarray,
    out_path: str | Path | None = None,
) -> Path:
    """Write build-zone labels (0/1/2) as MRC on the reference grid."""
    reference_path = Path(reference_path)
    if out_path is None:
        out_path = reference_path.with_name(f"{reference_path.stem}_build_zones.mrc")
    else:
        out_path = Path(out_path)
    save_volume_like_reference(
        reference_path,
        zones.astype(np.float32),
        out_path,
        dtype=np.float32,
        extra_label="build_zone 0=omit 1=caution 2=build",
    )
    return out_path
