"""2D slice helpers for map panel exports (no matplotlib)."""

from __future__ import annotations

import numpy as np

# (y0, y1, x0, x1) half-open row/col bounds on an XY slice (Z fixed).
SliceCrop = tuple[int, int, int, int]


def pick_slice_index(
    mask: np.ndarray,
    *,
    axis: int = 0,
    min_voxels: int = 500,
) -> int:
    """
    Choose a slice index along ``axis`` with substantial mask coverage.

    Prefers the slice with the most in-mask voxels; falls back to the volume center.
    """
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 3:
        raise ValueError(f"mask must be 3D, got {m.shape}")
    counts = m.sum(axis=tuple(i for i in range(3) if i != axis))
    if counts.size == 0:
        return 0
    if counts.max() < min_voxels:
        return int(m.shape[axis] // 2)
    threshold = 0.9 * float(counts.max())
    candidates = np.flatnonzero(counts >= threshold)
    center = (m.shape[axis] - 1) / 2.0
    return int(candidates[np.argmin(np.abs(candidates - center))])


def extract_slice(
    volume: np.ndarray,
    *,
    axis: int = 0,
    index: int,
) -> np.ndarray:
    """2D slice from a (Z, Y, X) volume."""
    vol = np.asarray(volume)
    if axis == 0:
        return vol[index, :, :]
    if axis == 1:
        return vol[:, index, :]
    return vol[:, :, index]


def slice_crop_from_mask(
    mask_sl: np.ndarray,
    *,
    pad_voxels: int = 24,
) -> SliceCrop:
    """Tight bounding box around in-mask pixels on a 2D slice, plus ``pad_voxels``."""
    m = np.asarray(mask_sl, dtype=bool)
    ny, nx = m.shape
    if not m.any():
        return (0, ny, 0, nx)
    ys, xs = np.nonzero(m)
    y0 = max(0, int(ys.min()) - pad_voxels)
    y1 = min(ny, int(ys.max()) + 1 + pad_voxels)
    x0 = max(0, int(xs.min()) - pad_voxels)
    x1 = min(nx, int(xs.max()) + 1 + pad_voxels)
    return (y0, y1, x0, x1)


def crop_slice_2d(sl: np.ndarray, crop: SliceCrop) -> np.ndarray:
    """Crop a 2D array with ``(y0, y1, x0, x1)`` bounds."""
    y0, y1, x0, x1 = crop
    return np.asarray(sl)[y0:y1, x0:x1]


def mask_slice_values(
    sl: np.ndarray,
    mask_sl: np.ndarray,
    *,
    outside: float = np.nan,
) -> np.ndarray:
    """Return slice with out-of-mask voxels replaced by ``outside`` (default NaN)."""
    out = np.asarray(sl, dtype=np.float64).copy()
    m = np.asarray(mask_sl, dtype=bool)
    out[~m] = outside
    return out


def apply_contour_mask(
    volume: np.ndarray,
    mask: np.ndarray,
    *,
    outside: float = np.nan,
) -> np.ndarray:
    """Restrict a 3D volume to the analysis contour; solvent set to ``outside``."""
    out = np.asarray(volume, dtype=np.float64).copy()
    m = np.asarray(mask, dtype=bool)
    if out.shape != m.shape:
        raise ValueError(f"volume shape {out.shape} != mask shape {m.shape}")
    out[~m] = outside
    return out.astype(np.asarray(volume).dtype, copy=False)


def robust_limits(
    sl: np.ndarray,
    *,
    lo_pct: float = 2.0,
    hi_pct: float = 98.0,
    mask_sl: np.ndarray | None = None,
) -> tuple[float, float]:
    v = np.asarray(sl, dtype=np.float64).ravel()
    if mask_sl is not None:
        v = v[np.asarray(mask_sl, dtype=bool).ravel()]
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(v, (lo_pct, hi_pct))
    if hi <= lo:
        hi = lo + 1e-6
    return float(lo), float(hi)


def locres_robust_limits(
    res_sl: np.ndarray,
    mask_sl: np.ndarray,
    *,
    lo_pct: float = 5.0,
    hi_pct: float = 95.0,
    default_lo: float = 2.0,
    default_hi: float = 8.0,
) -> tuple[float, float]:
    """In-mask percentile limits for local-resolution slice panels."""
    masked = mask_slice_values(res_sl, mask_sl)
    finite = masked[np.isfinite(masked)]
    if finite.size == 0:
        return default_lo, default_hi
    return (
        float(np.nanpercentile(finite, lo_pct)),
        float(np.nanpercentile(finite, hi_pct)),
    )
