"""Guinier B-factor estimation and B-factor sharpening (Relion / LocBFactor convention).

Estimates effective temperature factors from log-amplitude vs :math:`s^2` falloff
on unsharpened maps, then applies amplitude boosts in Fourier space. This is the
**sharpening** B used in cryo-EM post-processing — distinct from deposited atomic
B-factors in mmCIF models.

Convention (Kaur/Vargas Nat Commun 2021; Relion postprocess):
  log A(s) ≈ intercept + slope * s²   with   B = 4 * slope
  Sharpening multiplier: exp(-B * s² / 4)

LocBFactor uses spiral-phase amplitudes; here we use standard |FFT| shell
averages as a Python benchmark proxy. External LocBFactor MRC maps can be
ingested via :func:`load_external_bfactor_map`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import fft

from .local_fsc import _build_radial_shell_indices, _build_window

R_MIN_A_DEFAULT = 15.0


@dataclass(frozen=True)
class GuinierFit:
    """Linear Guinier fit in a resolution band [r_min_a, r_max_a]."""

    b_factor: float
    slope: float
    intercept: float
    n_shells: int
    r_squared: float


@dataclass(frozen=True)
class GuinierSharpeningEstimate:
    """Guinier B on avg-of-halves vs deposited primary (effective sharpening proxy)."""

    b_avg_guinier: float
    b_primary_guinier: float
    b_sharpening_delta: float
    b_avg_r_squared: float
    b_primary_r_squared: float

    @property
    def reported_style_sharpening_b(self) -> float:
        """Literature convention: negative B means sharpening was applied."""
        return self.b_sharpening_delta


def _mean_voxel_size(voxel_size_zyx: tuple[float, float, float] | float) -> float:
    if isinstance(voxel_size_zyx, (int, float)):
        return float(voxel_size_zyx)
    return float(np.mean(voxel_size_zyx))


def _shell_s_squared(
    shell_r: np.ndarray,
    *,
    n_ref: float,
    voxel_size_a: float,
) -> np.ndarray:
    """Spatial frequency squared (Å⁻²) for integer FFT shell indices."""
    s = shell_r.astype(np.float64) / (float(n_ref) * float(voxel_size_a))
    return s * s


def fit_guinier_b_shells(
    shell_r: np.ndarray,
    shell_amp: np.ndarray,
    *,
    voxel_size_a: float,
    n_ref: float,
    r_min_a: float,
    r_max_a: float,
) -> GuinierFit:
    """
    Fit log(amplitude) vs s² between 1/r_min_a and 1/r_max_a (Å⁻¹).

    ``r_min_a`` is the coarse end of the band (e.g. 15 Å); ``r_max_a`` is the
    fine end (global or local resolution in Å).
    """
    if r_min_a <= 0 or r_max_a <= 0:
        raise ValueError("r_min_a and r_max_a must be positive")
    if r_max_a >= r_min_a:
        raise ValueError("r_max_a must be finer (smaller Å) than r_min_a")

    s2 = _shell_s_squared(shell_r, n_ref=n_ref, voxel_size_a=voxel_size_a)
    s = np.sqrt(np.maximum(s2, 0.0))
    s_lo = 1.0 / float(r_min_a)
    s_hi = 1.0 / float(r_max_a)

    amp = np.asarray(shell_amp, dtype=np.float64)
    use = (
        (shell_r > 0)
        & np.isfinite(amp)
        & (amp > 0)
        & (s >= s_lo)
        & (s <= s_hi)
    )
    n = int(use.sum())
    if n < 3:
        return GuinierFit(
            b_factor=float("nan"),
            slope=float("nan"),
            intercept=float("nan"),
            n_shells=n,
            r_squared=float("nan"),
        )

    xs = s2[use]
    ys = np.log(amp[use])
    slope, intercept = np.polyfit(xs, ys, 1)
    y_hat = slope * xs + intercept
    ss_res = float(np.sum((ys - y_hat) ** 2))
    ss_tot = float(np.sum((ys - float(np.mean(ys))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return GuinierFit(
        b_factor=float(4.0 * slope),
        slope=float(slope),
        intercept=float(intercept),
        n_shells=n,
        r_squared=r2,
    )


def _radial_shell_amplitudes(
    volume: np.ndarray,
    *,
    shell_idx: np.ndarray | None = None,
    n_shells: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Mean |FFT| per integer shell for a cubic or general 3D volume."""
    vol = np.asarray(volume, dtype=np.float64)
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {vol.shape}")
    nz, ny, nx = vol.shape
    if shell_idx is None or n_shells is None:
        if nz != ny or ny != nx:
            raise ValueError("Anisotropic grids require cubic patches for shell indexing")
        shell_idx, n_shells = _build_radial_shell_indices(nz)

    ft = fft.rfftn(vol)
    amp = np.abs(ft).ravel()
    counts = np.bincount(shell_idx, minlength=n_shells).astype(np.float64)
    sums = np.bincount(shell_idx, weights=amp, minlength=n_shells)
    shell_amp = sums / np.maximum(counts, 1.0)
    shell_r = np.arange(n_shells, dtype=np.float64)
    return shell_r, shell_amp, nz


def estimate_global_guinier_b(
    volume: np.ndarray,
    voxel_size_zyx: tuple[float, float, float],
    *,
    r_min_a: float = R_MIN_A_DEFAULT,
    r_max_a: float,
    mask: np.ndarray | None = None,
) -> GuinierFit:
    """Whole-map radial Guinier B on an unsharpened volume (avg-of-halves)."""
    vol = np.asarray(volume, dtype=np.float64)
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if m.shape != vol.shape:
            raise ValueError("mask shape must match volume")
        vol = vol * m.astype(np.float64)

    nz, ny, nx = vol.shape
    n_ref = float(np.mean([nz, ny, nx]))
    d = _mean_voxel_size(voxel_size_zyx)

    kz_1d = np.fft.fftfreq(nz) * nz
    ky_1d = np.fft.fftfreq(ny) * ny
    kx_1d = np.fft.rfftfreq(nx) * nx
    kz, ky, kx = np.meshgrid(kz_1d, ky_1d, kx_1d, indexing="ij")
    shell_r = np.floor(np.sqrt(kz * kz + ky * ky + kx * kx)).astype(np.int32)
    n_shells = int(shell_r.max()) + 1
    shell_idx = shell_r.ravel()

    shell_r_out, shell_amp, _ = _radial_shell_amplitudes(
        vol, shell_idx=shell_idx, n_shells=n_shells
    )
    return fit_guinier_b_shells(
        shell_r_out,
        shell_amp,
        voxel_size_a=d,
        n_ref=n_ref,
        r_min_a=r_min_a,
        r_max_a=r_max_a,
    )


def compare_guinier_b_avg_vs_primary(
    avg_volume: np.ndarray,
    primary_volume: np.ndarray,
    voxel_size_zyx: tuple[float, float, float],
    *,
    r_min_a: float = R_MIN_A_DEFAULT,
    r_max_a: float,
    mask: np.ndarray | None = None,
) -> GuinierSharpeningEstimate:
    """
    Compare radial Guinier B on avg-of-halves vs the deposited primary map.

    ``b_sharpening_delta`` = B_primary − B_avg. When the depositor sharpened the
    map beyond the half-map average, this is typically negative (Relion-style
    reported sharpening B).
    """
    fit_avg = estimate_global_guinier_b(
        avg_volume, voxel_size_zyx, r_min_a=r_min_a, r_max_a=r_max_a, mask=mask
    )
    fit_pri = estimate_global_guinier_b(
        primary_volume, voxel_size_zyx, r_min_a=r_min_a, r_max_a=r_max_a, mask=mask
    )
    delta = float(fit_pri.b_factor - fit_avg.b_factor)
    return GuinierSharpeningEstimate(
        b_avg_guinier=float(fit_avg.b_factor),
        b_primary_guinier=float(fit_pri.b_factor),
        b_sharpening_delta=delta,
        b_avg_r_squared=float(fit_avg.r_squared),
        b_primary_r_squared=float(fit_pri.r_squared),
    )


def _guinier_b_from_patch(
    patch: np.ndarray,
    *,
    voxel_size_a: float,
    r_min_a: float,
    r_max_a: float,
) -> float:
    p = int(patch.shape[0])
    shell_idx, n_shells = _build_radial_shell_indices(p)
    shell_r, shell_amp, n_ref = _radial_shell_amplitudes(
        patch, shell_idx=shell_idx, n_shells=n_shells
    )
    fit = fit_guinier_b_shells(
        shell_r,
        shell_amp,
        voxel_size_a=voxel_size_a,
        n_ref=float(n_ref),
        r_min_a=r_min_a,
        r_max_a=r_max_a,
    )
    return fit.b_factor


def estimate_local_guinier_b_map(
    volume: np.ndarray,
    *,
    voxel_size_zyx: tuple[float, float, float],
    r_min_a: float,
    r_max_a: float | np.ndarray,
    patch_size: int = 17,
    stride: int = 8,
    mask: np.ndarray | None = None,
    window: str = "hann",
    r_max_mode: Literal["global", "locres"] = "global",
) -> np.ndarray:
    """
    Windowed local Guinier B map (LocBFactor-style proxy).

    ``r_max_a`` is either a scalar (global-resolution band everywhere) or a 3D
    local-resolution field (BlocRes Å) sampled at each patch center when
    ``r_max_mode='locres'``.
    """
    vol = np.asarray(volume, dtype=np.float64)
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D volume, got {vol.shape}")
    if patch_size % 2 == 0 or patch_size < 3:
        raise ValueError("patch_size must be odd and >= 3")
    if stride < 1:
        raise ValueError("stride must be >= 1")

    d = _mean_voxel_size(voxel_size_zyx)
    half = patch_size // 2
    window3d = _build_window(patch_size, window)

    rmax_field: np.ndarray | None
    if isinstance(r_max_a, np.ndarray):
        rmax_field = np.asarray(r_max_a, dtype=np.float64)
        if rmax_field.shape != vol.shape:
            raise ValueError("r_max_a field must match volume shape")
    else:
        rmax_field = None
        r_max_scalar = float(r_max_a)

    mask_b = None if mask is None else np.asarray(mask, dtype=bool)
    b_accum = np.zeros(vol.shape, dtype=np.float64)
    weight = np.zeros(vol.shape, dtype=np.float64)

    for cz in range(half, vol.shape[0] - half, stride):
        for cy in range(half, vol.shape[1] - half, stride):
            for cx in range(half, vol.shape[2] - half, stride):
                if mask_b is not None and not mask_b[cz, cy, cx]:
                    continue
                if rmax_field is not None:
                    r_hi = float(rmax_field[cz, cy, cx])
                    if not np.isfinite(r_hi) or r_hi <= 0:
                        continue
                    if r_hi >= r_min_a:
                        continue
                else:
                    r_hi = r_max_scalar

                z0, z1 = cz - half, cz + half + 1
                y0, y1 = cy - half, cy + half + 1
                x0, x1 = cx - half, cx + half + 1
                patch = vol[z0:z1, y0:y1, x0:x1] * window3d
                b_loc = _guinier_b_from_patch(
                    patch,
                    voxel_size_a=d,
                    r_min_a=r_min_a,
                    r_max_a=r_hi,
                )
                if not np.isfinite(b_loc):
                    continue
                b_accum[z0:z1, y0:y1, x0:x1] += b_loc * window3d
                weight[z0:z1, y0:y1, x0:x1] += window3d

    out = np.full(vol.shape, np.nan, dtype=np.float32)
    valid = weight > 0
    out[valid] = (b_accum[valid] / weight[valid]).astype(np.float32, copy=False)
    _ = r_max_mode  # caller documents which R_max policy was used
    return out


def _fourier_boost_grid(
    shape_zyx: tuple[int, int, int],
    voxel_size_zyx: tuple[float, float, float],
    b_factor: float | np.ndarray,
    *,
    r_min_a: float,
    r_max_a: float | np.ndarray,
) -> np.ndarray:
    """Per-Fourier-voxel boost exp(-B * s² / 4), optionally band-limited."""
    nz, ny, nx = shape_zyx
    kz = np.fft.fftfreq(nz)[:, None, None] / (nz * voxel_size_zyx[0])
    ky = np.fft.fftfreq(ny)[None, :, None] / (ny * voxel_size_zyx[1])
    kx = np.fft.rfftfreq(nx)[None, None, :] / (nx * voxel_size_zyx[2])
    s2 = kz * kz + ky * ky + kx * kx
    s = np.sqrt(np.maximum(s2, 0.0))

    if np.isscalar(b_factor):
        boost = np.exp(-float(b_factor) * s2 / 4.0)
    else:
        raise ValueError("Spatially varying B sharpening uses patch overlap path")

    s_lo = 1.0 / float(r_min_a)
    s_hi = 1.0 / float(r_max_a)
    band = (s >= s_lo) & (s <= s_hi)
    return np.where(band, boost, 1.0).astype(np.float64)


def apply_global_bfactor_sharpen(
    volume: np.ndarray,
    voxel_size_zyx: tuple[float, float, float],
    b_factor: float,
    *,
    r_min_a: float = R_MIN_A_DEFAULT,
    r_max_a: float,
) -> np.ndarray:
    """Apply depositor-style global B-factor sharpening in Fourier space."""
    vol = np.asarray(volume, dtype=np.float64)
    boost = _fourier_boost_grid(
        vol.shape,
        voxel_size_zyx,
        b_factor,
        r_min_a=r_min_a,
        r_max_a=r_max_a,
    )
    ft = fft.rfftn(vol)
    return fft.irfftn(ft * boost, s=vol.shape).astype(np.float32, copy=False)


def apply_local_bfactor_sharpen(
    volume: np.ndarray,
    b_map: np.ndarray,
    *,
    voxel_size_zyx: tuple[float, float, float],
    r_min_a: float = R_MIN_A_DEFAULT,
    r_max_a: float | np.ndarray,
    patch_size: int = 17,
    stride: int = 8,
    mask: np.ndarray | None = None,
    window: str = "hann",
) -> np.ndarray:
    """
    Patch-wise local B sharpening (LocBSharpen-style proxy).

    Each patch is sharpened with the local B at its center, then Hann-weighted
    into an overlap-add accumulator.
    """
    vol = np.asarray(volume, dtype=np.float64)
    bmap = np.asarray(b_map, dtype=np.float64)
    if vol.shape != bmap.shape:
        raise ValueError("b_map must match volume shape")

    d = _mean_voxel_size(voxel_size_zyx)
    half = patch_size // 2
    window3d = _build_window(patch_size, window)
    mask_b = None if mask is None else np.asarray(mask, dtype=bool)

    rmax_field: np.ndarray | None
    if isinstance(r_max_a, np.ndarray):
        rmax_field = np.asarray(r_max_a, dtype=np.float64)
    else:
        rmax_field = None
        r_max_scalar = float(r_max_a)

    out = np.zeros(vol.shape, dtype=np.float64)
    weight = np.zeros(vol.shape, dtype=np.float64)

    for cz in range(half, vol.shape[0] - half, stride):
        for cy in range(half, vol.shape[1] - half, stride):
            for cx in range(half, vol.shape[2] - half, stride):
                if mask_b is not None and not mask_b[cz, cy, cx]:
                    continue
                b_loc = float(bmap[cz, cy, cx])
                if not np.isfinite(b_loc):
                    continue
                r_hi = float(rmax_field[cz, cy, cx]) if rmax_field is not None else r_max_scalar
                if not np.isfinite(r_hi) or r_hi <= 0:
                    continue

                z0, z1 = cz - half, cz + half + 1
                y0, y1 = cy - half, cy + half + 1
                x0, x1 = cx - half, cx + half + 1
                patch = vol[z0:z1, y0:y1, x0:x1] * window3d
                boost = _fourier_boost_grid(
                    patch.shape,
                    voxel_size_zyx,
                    b_loc,
                    r_min_a=r_min_a,
                    r_max_a=r_hi,
                )
                sharp = fft.irfftn(fft.rfftn(patch) * boost, s=patch.shape)
                out[z0:z1, y0:y1, x0:x1] += sharp * window3d
                weight[z0:z1, y0:y1, x0:x1] += window3d

    valid = weight > 0
    out[valid] /= weight[valid]
    out[~valid] = vol[~valid]
    return out.astype(np.float32, copy=False)


def masked_map_ccc(
    a: np.ndarray,
    b: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Masked Pearson correlation (map CCC) between two volumes."""
    m = np.asarray(mask, dtype=bool) & np.isfinite(a) & np.isfinite(b)
    if int(m.sum()) < 100:
        return float("nan")
    x = np.asarray(a, dtype=np.float64)[m]
    y = np.asarray(b, dtype=np.float64)[m]
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    if denom <= 0:
        return float("nan")
    return float(np.sum(x * y) / denom)


def summarize_b_map(
    b_map: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    """Median / IQR of a B map inside a mask."""
    m = np.asarray(mask, dtype=bool) & np.isfinite(b_map)
    vals = np.asarray(b_map, dtype=np.float64)[m]
    if vals.size < 10:
        return {
            "median": float("nan"),
            "iqr": float("nan"),
            "std": float("nan"),
            "n_voxels": float(vals.size),
        }
    q25, q75 = np.percentile(vals, [25, 75])
    return {
        "median": float(np.median(vals)),
        "iqr": float(q75 - q25),
        "std": float(np.std(vals)),
        "n_voxels": float(vals.size),
    }


def load_external_bfactor_map(path: str) -> np.ndarray:
    """Load a per-voxel B map (e.g. LocBFactor MATLAB output) from MRC."""
    from .map_grid import load_map_grid

    grid = load_map_grid(path, dtype=np.float32)
    return np.asarray(grid.data, dtype=np.float32)


def apply_synthetic_bfactor_blur(
    volume: np.ndarray,
    voxel_size_zyx: tuple[float, float, float],
    b_factor: float,
    *,
    r_min_a: float = R_MIN_A_DEFAULT,
    r_max_a: float = 4.0,
) -> np.ndarray:
    """Blur a volume with exp(+B*s²/4) for synthetic recovery tests (B > 0 blurs)."""
    vol = np.asarray(volume, dtype=np.float64)
    boost = _fourier_boost_grid(
        vol.shape,
        voxel_size_zyx,
        -float(b_factor),
        r_min_a=r_min_a,
        r_max_a=r_max_a,
    )
    ft = fft.rfftn(vol)
    return fft.irfftn(ft / np.maximum(boost, 1e-12), s=vol.shape)
