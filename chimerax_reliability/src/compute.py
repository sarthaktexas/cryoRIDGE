"""Half-map reliability (same math as cryoem_mrc.reliability + reliability_driver)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

BUILD_ZONE_PALETTE = "0,red:0.999,red:1,yellow:1.999,yellow:2,green"

# Thesis palette (style/palette.py) for discrete zone surface coloring.
BUILD_ZONE_RGBA: dict[int, tuple[float, float, float, float]] = {
    0: (0.910, 0.188, 0.227, 1.0),  # #E8303A omit
    1: (0.961, 0.773, 0.094, 1.0),  # #F5C518 caution
    2: (0.231, 0.749, 0.416, 1.0),  # #3BBF6A build
}

# Reliability score: white (0) → blue (0.5) → purple (1), fixed range 0–1.
RELIABILITY_PALETTE = "0,white:0.5,blue:1,purple"


@dataclass(frozen=True)
class VolumeBbox:
    z0: int
    z1: int
    y0: int
    y1: int
    x0: int
    x1: int

    @property
    def slices(self) -> tuple[slice, slice, slice]:
        return (slice(self.z0, self.z1), slice(self.y0, self.y1), slice(self.x0, self.x1))

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.z1 - self.z0, self.y1 - self.y0, self.x1 - self.x0)

    @property
    def n_voxels(self) -> int:
        nz, ny, nx = self.shape
        return int(nz * ny * nx)


@dataclass(frozen=True)
class ReliabilityResult:
    reliability_score: np.ndarray
    build_zone: np.ndarray
    mask_voxels: int
    zone_counts: dict[int, int]
    crop_log: str | None


def pad_voxels_for_filters(*, window: int = 5) -> int:
    return max(int(window) // 2, 1)


def zscore_halfmap_average(half1: np.ndarray, half2: np.ndarray) -> np.ndarray:
    """Match cryoem_mrc.density_source.zscore_halfmap_average (float64 stats)."""
    rho = 0.5 * (np.asarray(half1, dtype=np.float64) + np.asarray(half2, dtype=np.float64))
    mu = float(rho.mean())
    sig = float(rho.std())
    return ((rho - mu) / (sig + 1e-6)).astype(np.float32)


def build_contour_mask(density: np.ndarray, contour: float) -> np.ndarray:
    return np.asarray(density, dtype=np.float64) >= float(contour)


def gradient_magnitude(volume: np.ndarray) -> np.ndarray:
    gz, gy, gx = np.gradient(volume)
    return np.sqrt(gz * gz + gy * gy + gx * gx).astype(np.float32)


def windowed_smoothness(rho: np.ndarray, *, window: int = 5, kappa: float = 1.0) -> np.ndarray:
    grad_sq = gradient_magnitude(rho) ** 2
    raw = 0.5 * float(kappa) * grad_sq
    w = int(window)
    if w <= 1:
        return np.asarray(raw, dtype=np.float32)
    smoothed = ndimage.uniform_filter(np.asarray(raw, dtype=np.float64), size=w, mode="nearest")
    return smoothed.astype(np.float32)


def percentile_rank_in_mask(volume: np.ndarray, mask: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
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


def classify_build_zones(
    reliability_score: np.ndarray,
    mask: np.ndarray,
    *,
    build_pct: float = 66.67,
    caution_pct: float = 33.33,
) -> np.ndarray:
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


def bbox_from_mask(mask: np.ndarray, *, pad: int = 0) -> VolumeBbox:
    coords = np.argwhere(mask)
    z0, y0, x0 = coords.min(axis=0)
    z1, y1, x1 = coords.max(axis=0) + 1
    nz, ny, nx = mask.shape
    p = int(pad)
    return VolumeBbox(
        z0=max(0, int(z0) - p),
        z1=min(nz, int(z1) + p),
        y0=max(0, int(y0) - p),
        y1=min(ny, int(y1) + p),
        x0=max(0, int(x0) - p),
        x1=min(nx, int(x1) + p),
    )


def embed_array(
    full_shape: tuple[int, int, int],
    bbox: VolumeBbox,
    cropped: np.ndarray,
    *,
    fill: float | int = 0,
    dtype: type | None = None,
) -> np.ndarray:
    dt = dtype or cropped.dtype
    out = np.full(full_shape, fill, dtype=dt)
    out[bbox.slices] = np.asarray(cropped, dtype=dt)
    return out


def format_bbox_log(bbox: VolumeBbox, full_shape: tuple[int, int, int], *, pad: int) -> str:
    full_n = int(np.prod(full_shape))
    frac = 100.0 * bbox.n_voxels / max(1, full_n)
    return (
        f"bbox {bbox.shape[0]}×{bbox.shape[1]}×{bbox.shape[2]} "
        f"of {full_shape[0]}×{full_shape[1]}×{full_shape[2]} "
        f"({bbox.n_voxels:,}/{full_n:,} voxels, {frac:.1f}%, pad={pad})"
    )


def compute_reliability(
    reference: np.ndarray,
    half1: np.ndarray,
    half2: np.ndarray,
    *,
    contour: float,
    window: int = 5,
    crop_to_contour: bool = True,
) -> ReliabilityResult:
    """Match cryoem_mrc ``reliability_driver`` (avg_half ρ, contour crop, percentile rank)."""
    ref = np.asarray(reference, dtype=np.float32)
    h1 = np.asarray(half1, dtype=np.float32)
    h2 = np.asarray(half2, dtype=np.float32)
    if ref.shape != h1.shape or ref.shape != h2.shape:
        raise ValueError(
            f"Grid shape mismatch: reference {ref.shape}, half1 {h1.shape}, half2 {h2.shape}."
        )

    mask = build_contour_mask(ref, contour)
    if not mask.any():
        raise ValueError(f"Contour {contour:g} gives an empty mask on the reference map.")

    rho = zscore_halfmap_average(h1, h2)
    pad = pad_voxels_for_filters(window=window)
    crop_log: str | None = None

    if crop_to_contour:
        bbox = bbox_from_mask(mask, pad=pad)
        crop_log = format_bbox_log(bbox, ref.shape, pad=pad)
        sl = bbox.slices
        work_mask = mask[sl]
        smooth = windowed_smoothness(rho[sl], window=window)
        score = percentile_rank_in_mask(smooth, work_mask)
        zones = classify_build_zones(score, work_mask)
        reliability_score = embed_array(ref.shape, bbox, score, dtype=np.float32)
        build_zone = embed_array(ref.shape, bbox, zones, dtype=np.uint8)
    else:
        smooth = windowed_smoothness(rho, window=window)
        reliability_score = percentile_rank_in_mask(smooth, mask)
        build_zone = classify_build_zones(reliability_score, mask)

    zone_counts = {int(z): int((build_zone[mask] == z).sum()) for z in (0, 1, 2)}
    return ReliabilityResult(
        reliability_score=reliability_score,
        build_zone=build_zone,
        mask_voxels=int(mask.sum()),
        zone_counts=zone_counts,
        crop_log=crop_log,
    )
