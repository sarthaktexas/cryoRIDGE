"""Density-derived kinetic/curvature maps for the T/V block-structure experiment.

Motivation (docs/ALTERNATIVE_APPROACHES.md)
-------------------------------------------

The deployed reliability pipeline uses windowed gradient energy from
globally z-scored half-map average density. A reviewer-facing
worry is that any second-derivative *T* (the Laplacian, ``tr(H)``) and a
curvature-based *V* (built from the Hessian) are *two contractions of the same
Hessian* and therefore not information-independent. The proposed fix is to define
the kinetic term as the **von Weizsäcker kinetic-energy density** ``∝ |∇ρ|²`` (a
first-derivative object) instead of the Laplacian, which is genuinely orthogonal
in information content to a second-derivative *V*.

This module computes, from a single density map ``ρ`` (no model, no half-maps):

- ``T_laplacian``       signed discrete Laplacian ``∇²ρ`` (= ``tr(H)``)
- ``T_laplacian_abs``   ``|∇²ρ|`` — curvature *strength* (Laplacian is negative at peaks)
- ``T_vonweizsacker``   ``|∇ρ|²`` — von Weizsäcker kinetic-energy density (∝, no 1/ρ)
- ``V_curvature``       ``‖H‖_F²`` = Σ_ij (∂²ρ/∂x_i∂x_j)² — squared local curvature

Scientific prediction being tested
----------------------------------

Across the cohort, sampled at Cα and correlated (Spearman) against deposited
B-factors and BlocRes local resolution, we expect a **block structure**:

- *T* (both forms) loads on **local resolution** (high-k weighting / sharpness),
- *V* (squared curvature) loads on **B-factor** (curvature ∝ 1/σ² ∝ 1/B),
- off-diagonal couplings (T↔B, V↔resolution) are comparatively weak, and
- the von Weizsäcker *T* (``|∇ρ|²``) **de-correlates from V** more than the
  Laplacian *T* does — because ``|∇ρ|²`` is a first-derivative object whereas the
  Laplacian and ``V`` are both contractions of the Hessian.

All per-map summaries use Spearman ρ, which is invariant to monotonic per-map
rescaling — so cross-cohort intensity-normalization differences do not bias the
block-structure test.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage, stats

from cryoem_mrc.analysis import build_contour_mask
from cryoem_mrc.local_resolution import aggregate_locres_to_ca, locres_blocres_path
from cryoem_mrc.map_grid import load_map_grid
from cryoem_mrc.repo_paths import COHORT_MANIFEST
from cryoem_mrc.structure_validation import (
    iter_ca_residues,
    load_cohort_manifest_row,
    physical_xyz_to_voxel_indices,
    sample_volume_at_ca,
)

TV_FEATURE_KEYS: tuple[str, ...] = (
    "T_laplacian_abs",
    "T_vonweizsacker",
    "V_curvature",
)

TV_TARGET_KEYS: tuple[str, ...] = ("b_factor", "local_resolution")




def _tv_block(block: np.ndarray, spacing: tuple[float, float, float]) -> dict[str, np.ndarray]:
    """Compute the four T/V scalar maps on a (possibly padded) Z-slab."""
    sz, sy, sx = spacing
    gz, gy, gx = np.gradient(block, sz, sy, sx)

    grad_sq = gz * gz + gy * gy + gx * gx  # |∇ρ|² (von Weizsäcker kinetic density, ∝)

    gzz, gzy, gzx = np.gradient(gz, sz, sy, sx)
    gyz, gyy, gyx = np.gradient(gy, sz, sy, sx)
    gxz, gxy, gxx = np.gradient(gx, sz, sy, sx)

    laplacian = gzz + gyy + gxx  # ∇²ρ = tr(H)

    # Frobenius² of the symmetrized Hessian: Σ_ij H_ij².
    h01 = 0.5 * (gzy + gyz)
    h02 = 0.5 * (gzx + gxz)
    h12 = 0.5 * (gyx + gxy)
    frob_sq = (
        gzz * gzz + gyy * gyy + gxx * gxx + 2.0 * (h01 * h01 + h02 * h02 + h12 * h12)
    )

    return {
        "T_laplacian": laplacian,
        "T_laplacian_abs": np.abs(laplacian),
        "T_vonweizsacker": grad_sq,
        "V_curvature": frob_sq,
    }


def density_tv_curvature_maps(
    rho: np.ndarray,
    *,
    spacing_zyx: tuple[float, float, float] | None = None,
    chunk_z: int | None = 64,
) -> dict[str, np.ndarray]:
    """
    Compute the von-Weizsäcker / Laplacian / Hessian-curvature scalar maps on ``ρ``.

    Parameters
    ----------
    rho
        Density volume ``(Z, Y, X)``.
    spacing_zyx
        Physical voxel spacing in Å ``(z, y, x)`` passed to the finite-difference
        derivatives. Defaults to unit spacing; for isotropic voxels the choice is
        irrelevant to per-map Spearman correlations (it rescales every map by the
        same constant), but using the true spacing keeps the maps physically
        dimensioned (Å⁻¹ etc.) for figures.
    chunk_z
        Z-slab height for bounded peak memory. ``None`` processes the full volume.

    Returns
    -------
    dict
        Keys ``T_laplacian``, ``T_laplacian_abs``, ``T_vonweizsacker``,
        ``V_curvature`` — each aligned with ``rho``.
    """
    v = np.asarray(rho, dtype=np.float64)
    if v.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {v.shape}")
    spacing = (1.0, 1.0, 1.0) if spacing_zyx is None else tuple(float(s) for s in spacing_zyx)
    out_dtype = np.asarray(rho).dtype

    if chunk_z is None:
        raw = _tv_block(v, spacing)
        return {k: arr.astype(out_dtype, copy=False) for k, arr in raw.items()}

    nz, ny, nx = v.shape
    pad = 3  # two-level central differences need a 3-voxel halo
    tmpl = _tv_block(v[: min(nz, chunk_z + 2 * pad)], spacing)
    out: dict[str, np.ndarray] = {k: np.empty((nz, ny, nx), dtype=out_dtype) for k in tmpl}

    z0 = 0
    while z0 < nz:
        z1 = min(nz, z0 + chunk_z)
        za = max(0, z0 - pad)
        zb = min(nz, z1 + pad)
        part = _tv_block(v[za:zb], spacing)
        off = z0 - za
        take = z1 - z0
        for k in out:
            out[k][z0:z1] = part[k][off : off + take].astype(out_dtype, copy=False)
        z0 = z1

    return out


# Per-map Cα sampling against B-factor and local resolution


def compute_map_tv_table(
    emdb_id: str,
    *,
    manifest: Path = COHORT_MANIFEST,
    sphere_radius_a: float = 2.0,
    chunk_z: int | None = 64,
    use_voxel_spacing: bool = True,
) -> pd.DataFrame:
    """
    Per-residue T/V curvature features at Cα for one cohort entry.

    Builds the density-derived maps, samples them in a sphere around each Cα,
    and joins deposited B-factors plus BlocRes local resolution (when present).

    Returns a tidy DataFrame; ``local_resolution`` is NaN when no BlocRes map
    exists for the entry.
    """
    emdb_id = str(emdb_id).strip()
    row = load_cohort_manifest_row(manifest, emdb_id)
    ref_path = Path(row["reference_mrc"])
    pdb_raw = row.get("flexibility_path_or_pdb", "").strip()
    pdb_path = Path(pdb_raw) if pdb_raw else None
    contour = float(row["contour"])

    if not ref_path.is_file():
        raise FileNotFoundError(f"EMD-{emdb_id} missing reference: {ref_path}")
    if pdb_path is None or not pdb_path.is_file():
        raise FileNotFoundError(f"EMD-{emdb_id} missing structure: {pdb_path}")

    grid = load_map_grid(ref_path, dtype=np.float32)
    rho = np.asarray(grid.data, dtype=np.float32)
    mask = build_contour_mask(rho, contour)

    spacing = grid.voxel_size_zyx if use_voxel_spacing else None
    maps = density_tv_curvature_maps(rho, spacing_zyx=spacing, chunk_z=chunk_z)

    residues = iter_ca_residues(pdb_path)
    data: dict[str, object] = {
        "emdb_id": emdb_id,
        "chain": [r.chain for r in residues],
        "seq_num": [r.seq_num for r in residues],
        "res_name": [r.res_name for r in residues],
        "b_factor": np.array([r.b_iso for r in residues], dtype=np.float64),
    }
    for key in ("T_laplacian", "T_laplacian_abs", "T_vonweizsacker", "V_curvature"):
        data[key] = sample_volume_at_ca(maps[key], grid, residues, sphere_radius_a=sphere_radius_a)

    in_mask = np.empty(len(residues), dtype=bool)
    for i, res in enumerate(residues):
        iz, iy, ix = physical_xyz_to_voxel_indices(res.x, res.y, res.z, grid)
        in_mask[i] = (
            0 <= iz < mask.shape[0]
            and 0 <= iy < mask.shape[1]
            and 0 <= ix < mask.shape[2]
            and bool(mask[iz, iy, ix])
        )
    data["in_contour_mask"] = in_mask

    df = pd.DataFrame(data)
    df["local_resolution"] = np.nan

    locres_path = locres_blocres_path(emdb_id)
    # Skip the join when either side is empty (e.g. a model with no Cα, as on
    # EMD-33736): an empty residue table has a float64 ``chain`` key that cannot be
    # merged against the object key, and there is nothing to join anyway. Leaving
    # ``local_resolution`` as NaN lets downstream correlations skip the map cleanly.
    if locres_path.is_file() and not df.empty:
        loc_df = aggregate_locres_to_ca(
            locres_path,
            pdb_path,
            radius_angstrom=sphere_radius_a,
            reference_path=ref_path,
        ).rename(columns={"local_resolution_mean": "local_resolution"})
        if not loc_df.empty:
            df = df.drop(columns=["local_resolution"]).merge(
                loc_df[["chain", "seq_num", "local_resolution"]],
                on=["chain", "seq_num"],
                how="left",
            )

    return df


# Block-structure correlations


@dataclass
class TvBlockResult:
    """Per-map Spearman summary for the T/V block-structure test."""

    emdb_id: str
    n_in_mask: int
    feature_vs_target: dict[str, float]  # "{feature}__{target}" -> rho
    feature_vs_feature: dict[str, float]  # "{feat_i}__{feat_j}" -> rho

    def flat_record(self) -> dict[str, object]:
        rec: dict[str, object] = {"emdb_id": self.emdb_id, "n_in_mask": self.n_in_mask}
        rec.update(self.feature_vs_target)
        rec.update(self.feature_vs_feature)
        return rec


def _spearman(x: np.ndarray, y: np.ndarray, *, min_pairs: int = 10) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) < min_pairs:
        return float("nan")
    xf, yf = x[m], y[m]
    if xf.std() == 0 or yf.std() == 0:
        return float("nan")
    return float(stats.spearmanr(xf, yf).statistic)


def tv_block_correlations(
    df: pd.DataFrame,
    *,
    emdb_id: str,
    feature_keys: tuple[str, ...] = TV_FEATURE_KEYS,
    target_keys: tuple[str, ...] = TV_TARGET_KEYS,
    in_mask_only: bool = True,
    min_pairs: int = 10,
) -> TvBlockResult:
    """
    Per-map Spearman ρ of each T/V feature against each target (B, local res) plus
    feature-feature couplings (to test von Weizsäcker T de-correlating from V).
    """
    use = df
    if in_mask_only and "in_contour_mask" in df.columns:
        use = df[df["in_contour_mask"].astype(bool)]

    feature_vs_target: dict[str, float] = {}
    for feat in feature_keys:
        for tgt in target_keys:
            key = f"rho__{feat}__vs__{tgt}"
            if feat in use.columns and tgt in use.columns:
                feature_vs_target[key] = _spearman(
                    use[feat].to_numpy(dtype=np.float64),
                    use[tgt].to_numpy(dtype=np.float64),
                    min_pairs=min_pairs,
                )
            else:
                feature_vs_target[key] = float("nan")

    feature_vs_feature: dict[str, float] = {}
    for i, fi in enumerate(feature_keys):
        for fj in feature_keys[i + 1 :]:
            key = f"rho__{fi}__vs__{fj}"
            if fi in use.columns and fj in use.columns:
                feature_vs_feature[key] = _spearman(
                    use[fi].to_numpy(dtype=np.float64),
                    use[fj].to_numpy(dtype=np.float64),
                    min_pairs=min_pairs,
                )
            else:
                feature_vs_feature[key] = float("nan")

    n_in_mask = int(len(use))
    return TvBlockResult(
        emdb_id=str(emdb_id),
        n_in_mask=n_in_mask,
        feature_vs_target=feature_vs_target,
        feature_vs_feature=feature_vs_feature,
    )


__all__ = [
    "TV_FEATURE_KEYS",
    "TV_TARGET_KEYS",
    "TvBlockResult",
    "compute_map_tv_table",
    "density_tv_curvature_maps",
    "tv_block_correlations",
]
