"""Residue-level B-factor vs map reliability (thesis external validation)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy import stats

from cryoem_mrc.halfmap_metrics import load_windowed_halfmap_correlation
from cryoem_mrc.map_grid import load_map_grid
from cryoem_mrc.repo_paths import COHORT_MANIFEST, find_features_npz, halfmap_metrics_npz
from cryoem_mrc.structure_validation import (
    ResidueValidationRow,
    _b_iso_is_uniform,
    build_residue_validation_table,
    default_reliability_out_dir,
    iter_ca_residues,
    load_cohort_manifest_row,
    write_residue_validation_csv,
)

from thesis.reliability_volumes import load_reliability_mrc_pair, recompute_lh_volumes


@dataclass
class BfactorValidationStats:
    emdb_id: str
    n_residues: int
    n_in_mask: int
    spearman_b_vs_reliability: float
    spearman_b_vs_H_repro: float
    spearman_b_vs_build_zone: float
    partial_b_vs_reliability_given_variance: float = float("nan")
    median_b_by_zone: dict[int, float] = field(default_factory=dict)
    notes: str = ""


def _partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    xr, yr, zr = stats.rankdata(x), stats.rankdata(y), stats.rankdata(z)
    r_xy = np.corrcoef(xr, yr)[0, 1]
    r_xz = np.corrcoef(xr, zr)[0, 1]
    r_yz = np.corrcoef(yr, zr)[0, 1]
    d = (1.0 - r_xz * r_xz) * (1.0 - r_yz * r_yz)
    if d <= 0:
        return float("nan")
    return float((r_xy - r_xz * r_yz) / np.sqrt(d))


def compute_bfactor_validation_stats(
    rows: Sequence[ResidueValidationRow],
    *,
    emdb_id: str,
    in_mask_only: bool = True,
) -> BfactorValidationStats:
    use = [r for r in rows if (r.in_contour_mask if in_mask_only else True)]
    n_all = len(rows)
    n_use = len(use)
    if n_use < 10:
        return BfactorValidationStats(
            emdb_id=emdb_id,
            n_residues=n_all,
            n_in_mask=n_use,
            spearman_b_vs_reliability=float("nan"),
            spearman_b_vs_H_repro=float("nan"),
            spearman_b_vs_build_zone=float("nan"),
            notes="too few residues for correlation",
        )

    b = np.array([r.b_iso for r in use], dtype=np.float64)
    rel = np.array([r.reliability_score for r in use], dtype=np.float64)
    h = np.array([r.reliability_H_repro for r in use], dtype=np.float64)
    zones = np.array([r.build_zone for r in use], dtype=np.float64)
    var = np.array([r.local_variance for r in use], dtype=np.float64)

    med_by_zone: dict[int, float] = {}
    for z in (0, 1, 2):
        zb = b[zones == z]
        if zb.size:
            med_by_zone[z] = float(np.median(zb))

    if _b_iso_is_uniform(b):
        return BfactorValidationStats(
            emdb_id=emdb_id,
            n_residues=n_all,
            n_in_mask=n_use,
            spearman_b_vs_reliability=float("nan"),
            spearman_b_vs_H_repro=float("nan"),
            spearman_b_vs_build_zone=float("nan"),
            partial_b_vs_reliability_given_variance=float("nan"),
            median_b_by_zone=med_by_zone,
            notes=(
                "Uniform deposited B-factors (B_iso has zero variance in mask); "
                "Spearman correlations skipped."
            ),
        )

    rho_rel, _ = stats.spearmanr(b, rel)
    rho_h, _ = stats.spearmanr(b, h)
    rho_z, _ = stats.spearmanr(b, zones)

    partial = float("nan")
    if np.isfinite(var).sum() >= 10:
        ok = np.isfinite(var)
        partial = _partial_spearman(b[ok], rel[ok], var[ok])

    return BfactorValidationStats(
        emdb_id=emdb_id,
        n_residues=n_all,
        n_in_mask=n_use,
        spearman_b_vs_reliability=float(rho_rel),
        spearman_b_vs_H_repro=float(rho_h),
        spearman_b_vs_build_zone=float(rho_z),
        partial_b_vs_reliability_given_variance=partial,
        median_b_by_zone=med_by_zone,
    )


def run_emdb_bfactor_validation(
    emd_id: str,
    *,
    manifest: Path = COHORT_MANIFEST,
    reliability_dir: Path | None = None,
    reference: Path | None = None,
    pdb: Path | None = None,
    contour: float | None = None,
    halfmap_npz: Path | None = None,
    features_npz: Path | None = None,
    window_radius: int = 0,
    require_b_factor_source: bool = True,
) -> tuple[int, list[ResidueValidationRow], BfactorValidationStats | None, Path]:
    row = load_cohort_manifest_row(manifest, emd_id)
    if require_b_factor_source and row.get("flexibility_source", "").strip() != "b_factor" and pdb is None:
        return 0, [], None, default_reliability_out_dir(emd_id)

    ref_path = reference or Path(row["reference_mrc"])
    pdb_path = pdb or Path(row["flexibility_path_or_pdb"])
    contour_val = contour if contour is not None else float(row["contour"])
    out_dir = default_reliability_out_dir(emd_id, reliability_dir)

    for label, p in (("reference", ref_path), ("pdb", pdb_path)):
        if not p.exists():
            raise FileNotFoundError(f"EMD-{emd_id} missing {label}: {p}")

    reliability_score, build_zone = load_reliability_mrc_pair(emd_id, out_dir=out_dir)
    reliability_H_repro, _v = recompute_lh_volumes(
        emd_id,
        reference_path=ref_path,
        half1_path=Path(row["half1_path"]),
        half2_path=Path(row["half2_path"]),
        contour=contour_val,
    )

    if halfmap_npz is None:
        candidate = halfmap_metrics_npz(emd_id)
        halfmap_npz = candidate if candidate.exists() else None

    residues = iter_ca_residues(pdb_path)
    grid = load_map_grid(ref_path, dtype=np.float32)
    reference_density = np.asarray(grid.data, dtype=np.float32)

    if reliability_score.shape != reference_density.shape:
        raise ValueError(
            f"EMD-{emd_id}: reliability shape {reliability_score.shape} != reference {reference_density.shape}"
        )

    cc = None
    if halfmap_npz is not None and halfmap_npz.exists():
        with np.load(halfmap_npz, allow_pickle=False) as hm:
            cc = load_windowed_halfmap_correlation(hm)
    local_var = None
    if features_npz is None:
        features_npz = find_features_npz(ref_path.parent, emd_id, contour_val)
    if features_npz is not None and features_npz.exists():
        with np.load(features_npz, allow_pickle=False) as feat:
            local_var = np.asarray(feat["local_variance"], dtype=np.float32)

    rows = build_residue_validation_table(
        residues,
        grid=grid,
        reference_density=reference_density,
        contour=contour_val,
        reliability_score=reliability_score,
        reliability_H_repro=reliability_H_repro,
        build_zone=build_zone,
        windowed_halfmap_correlation=cc,
        local_variance=local_var,
        window_radius=window_radius,
    )
    stats = compute_bfactor_validation_stats(rows, emdb_id=emd_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_residue_validation_csv(out_dir / "residue_validation.csv", rows)
    write_bfactor_validation_md(
        out_dir / "B_FACTOR_VALIDATION.md",
        stats,
        pdb_path=pdb_path,
        contour=contour_val,
        sampling="nearest voxel" if window_radius <= 0 else f"{2 * window_radius + 1}³ window mean",
    )
    return 0, rows, stats, out_dir


def write_bfactor_validation_md(
    path: Path,
    stats: BfactorValidationStats,
    *,
    pdb_path: Path,
    contour: float,
    sampling: str = "nearest voxel",
) -> None:
    zlabels = {0: "omit", 1: "caution", 2: "build"}
    zone_lines = "\n".join(
        f"| {zlabels.get(z, z)} | {stats.median_b_by_zone.get(z, float('nan')):.1f} |"
        for z in (0, 1, 2)
        if z in stats.median_b_by_zone
    )
    caveat_line = f"\n\n**Caveat:** {stats.notes}" if stats.notes else ""
    text = f"""# B-factor external validation — EMD-{stats.emdb_id}

Exploratory comparison of **deposited model B-factors** vs **map reliability** (H_repro / build zones).
This does **not** claim H_repro measures molecular flexibility — B_iso reflects refinement displacement
and local order, while reliability_score reflects half-map agreement inside the contour mask.

**Model:** `{pdb_path}`  
**Mask:** deposited reference ρ ≥ {contour} (Cα: {sampling})  
**Residues:** {stats.n_residues:,} Cα total; **{stats.n_in_mask:,}** inside contour mask

---

## Spearman correlations (in-mask Cα)

| Comparison | ρ |
|------------|--:|
| B_iso vs reliability_score | {stats.spearman_b_vs_reliability:+.3f} |
| B_iso vs reliability_H_repro | {stats.spearman_b_vs_H_repro:+.3f} |
| B_iso vs build_zone (0/1/2) | {stats.spearman_b_vs_build_zone:+.3f} |
| Partial: B vs reliability \\| local_variance | {stats.partial_b_vs_reliability_given_variance:+.3f} |

**Sign note:** Higher B_iso ↔ more displacement; higher reliability_score ↔ more reliable map.
A **negative** ρ(B, reliability) is the naive expectation if both proxy local order.{caveat_line}

---

## Median B_iso by build zone

| Zone | Median B_iso |
|------|-------------:|
{zone_lines}

---

## Files

| File | Description |
|------|-------------|
| `residue_validation.csv` | Per-residue table |
"""
    path.write_text(text)
