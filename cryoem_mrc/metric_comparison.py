"""Cross-metric residue tables and correlations for cohort validation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

from .analysis import build_contour_mask
from .half_map_repro import WINDOWED_HALFMAP_CORRELATION_KEY, load_windowed_halfmap_correlation
from .local_resolution import (
    RESMAP_UNRESOLVED_SENTINEL_A,
    aggregate_locres_to_ca,
    locres_blocres_path,
    locres_resmap_path,
)
from .map_grid import load_map_grid
from .repo_paths import COHORT_MANIFEST, find_features_npz, halfmap_metrics_npz, resolve_halfmap_reliability_dir
from .structure_validation import (
    build_residue_validation_table,
    iter_ca_residues,
    load_cohort_manifest_row,
    sample_volume_at_ca,
)

logger = logging.getLogger(__name__)

# Cross-metric coupling uses one LH-pipeline column: constraint V (placement axis).
# reliability_score is the in-mask percentile rank of H_repro (= T + V); H_repro and
# its rank are Spearman-redundant with each other and overlap the same Hamiltonian
# family as V — omit them here to avoid duplicate rows on the heatmap.
METRIC_COLUMNS = (
    "v_metric",
    "b_factor",
    WINDOWED_HALFMAP_CORRELATION_KEY,
    "local_variance",
    "local_resolution",
)

LocresSource = Literal["blocres", "resmap"]


def metric_comparison_dirname(locres_source: LocresSource = "blocres") -> str:
    """Per-map export folder name under ``outputs/emd_<ID>/``."""
    if locres_source == "blocres":
        return "metric_comparison"
    return f"metric_comparison_{locres_source}"


def _locres_path(emdb_id: str, locres_source: LocresSource) -> Path:
    if locres_source == "resmap":
        return locres_resmap_path(emdb_id)
    return locres_blocres_path(emdb_id)


def load_all_metrics(
    emdb_id: str,
    *,
    manifest: Path = COHORT_MANIFEST,
    sphere_radius_a: float = 2.0,
    locres_source: LocresSource = "blocres",
) -> pd.DataFrame:
    """
    Per-residue metrics for one EMDB entry.

    ``local_resolution`` is filled from BlocRes (``locres_blocres.mrc``, contour-masked
    at run time) or ResMap (``resmap/resmap.mrc``, auto-masked at run time). Cα
    correlations use ``in_contour_mask`` for both so comparisons stay in the depositor
    contour even when ResMap computed values more broadly.
    """
    emdb_id = str(emdb_id).strip()
    row = load_cohort_manifest_row(manifest, emdb_id)
    ref_path = Path(row["reference_mrc"])
    pdb_raw = row.get("flexibility_path_or_pdb", "").strip()
    pdb_path = Path(pdb_raw) if pdb_raw else None
    contour = float(row["contour"])

    out_dir = resolve_halfmap_reliability_dir(emdb_id)
    npz_path = out_dir / "reliability.npz"

    if not ref_path.is_file():
        raise FileNotFoundError(f"EMD-{emdb_id} missing reference: {ref_path}")
    if pdb_path is None or not pdb_path.is_file():
        raise FileNotFoundError(f"EMD-{emdb_id} missing structure: {pdb_path}")
    if not npz_path.is_file():
        raise FileNotFoundError(f"EMD-{emdb_id} missing reliability.npz: {npz_path}")

    grid = load_map_grid(ref_path, dtype=np.float32)
    reference_density = np.asarray(grid.data, dtype=np.float32)

    with np.load(npz_path, allow_pickle=False) as d:
        reliability_score = np.asarray(d["reliability_score"], dtype=np.float32)
        reliability_H_repro = np.asarray(d["reliability_H_repro"], dtype=np.float32)
        build_zone = np.asarray(d["build_zone"], dtype=np.uint8)
        v_metric_vol = np.asarray(
            d.get("reliability_smoothness", d.get("reliability_constraint_V")),
            dtype=np.float32,
        )

    halfmap_npz = halfmap_metrics_npz(emdb_id)
    cc = None
    if halfmap_npz.is_file():
        with np.load(halfmap_npz, allow_pickle=False) as hm:
            cc = load_windowed_halfmap_correlation(hm)

    features_npz = find_features_npz(ref_path.parent, emdb_id, contour)
    local_var = None
    if features_npz is not None and features_npz.is_file():
        with np.load(features_npz, allow_pickle=False) as feat:
            local_var = np.asarray(feat["local_variance"], dtype=np.float32)

    residues = iter_ca_residues(pdb_path)
    rows = build_residue_validation_table(
        residues,
        grid=grid,
        reference_density=reference_density,
        contour=contour,
        reliability_score=reliability_score,
        reliability_H_repro=reliability_H_repro,
        build_zone=build_zone,
        windowed_halfmap_correlation=cc,
        local_variance=local_var,
        window_radius=0,
    )
    v_at_ca = sample_volume_at_ca(
        v_metric_vol,
        grid,
        residues,
        sphere_radius_a=sphere_radius_a,
    )

    df = pd.DataFrame(
        {
            "emdb_id": emdb_id,
            "chain": [r.chain for r in rows],
            "seq_num": [r.seq_num for r in rows],
            "seq_icode": [r.seq_icode for r in rows],
            "res_name": [r.res_name for r in rows],
            "v_metric": v_at_ca,
            "reliability_score": [r.reliability_score for r in rows],
            "reliability_H_repro": [r.reliability_H_repro for r in rows],
            "b_factor": [r.b_iso for r in rows],
            WINDOWED_HALFMAP_CORRELATION_KEY: [
                r.windowed_halfmap_correlation for r in rows
            ],
            "local_variance": [r.local_variance for r in rows],
            "build_zone": [r.build_zone for r in rows],
            "in_contour_mask": [r.in_contour_mask for r in rows],
            "local_resolution": np.nan,
        }
    )

    locres_path = _locres_path(emdb_id, locres_source)
    if locres_path.is_file():
        try:
            agg_kw: dict[str, object] = {
                "radius_angstrom": sphere_radius_a,
                "reference_path": ref_path,
            }
            if locres_source == "resmap":
                agg_kw["exclude_at_or_above"] = RESMAP_UNRESOLVED_SENTINEL_A
            loc_df = aggregate_locres_to_ca(
                locres_path,
                pdb_path,
                **agg_kw,
            )
            loc_df = loc_df.rename(columns={"local_resolution_mean": "local_resolution"})
            df = df.drop(columns=["local_resolution"]).merge(
                loc_df[["chain", "seq_num", "local_resolution"]],
                on=["chain", "seq_num"],
                how="left",
            )
        except Exception as exc:
            logger.warning(
                "EMD-%s: failed to aggregate %s (%s): %s",
                emdb_id,
                locres_path,
                locres_source,
                exc,
            )
    else:
        logger.warning(
            "EMD-%s: no %s map at %s; local_resolution left as NaN "
            "(run scripts/run_resmap_align_to_reference.py for ResMap)",
            emdb_id,
            locres_source,
            locres_path,
        )

    return df


def compute_cross_metric_correlations(
    df: pd.DataFrame,
    *,
    columns: tuple[str, ...] = METRIC_COLUMNS,
    min_pairs: int = 30,
    mask_column: str = "in_contour_mask",
) -> pd.DataFrame:
    """
    Pairwise Spearman ρ between numeric metric columns (in-mask residues by default).
    """
    use = df
    if mask_column in df.columns:
        use = df[df[mask_column].astype(bool)]

    avail = [c for c in columns if c in use.columns]
    numeric = use[avail].apply(pd.to_numeric, errors="coerce")
    n = len(avail)
    rho = np.full((n, n), np.nan, dtype=np.float64)
    pval = np.full((n, n), np.nan, dtype=np.float64)

    for i, ci in enumerate(avail):
        for j, cj in enumerate(avail):
            if j < i:
                rho[i, j] = rho[j, i]
                pval[i, j] = pval[j, i]
                continue
            m = numeric[ci].notna() & numeric[cj].notna()
            if m.sum() < min_pairs:
                continue
            r, p = stats.spearmanr(numeric.loc[m, ci], numeric.loc[m, cj])
            rho[i, j] = float(r)
            pval[i, j] = float(p)

    out = pd.DataFrame(rho, index=avail, columns=avail)
    out.attrs["p_values"] = pd.DataFrame(pval, index=avail, columns=avail)
    if mask_column in use.columns:
        out.attrs["n_residues"] = int(use[mask_column].astype(bool).sum())
    else:
        out.attrs["n_residues"] = len(use)
    return out
