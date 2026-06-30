"""Builder-omission ROC: classify built vs sequence-gap proxy sites inside the contour mask.

Class 0 (negative): deposited Cα inside the contour mask (expert built).
Class 1 (positive): linearly interpolated backbone positions for missing sequence
indices flanked by built residues on the same chain, restricted to in-mask voxels.

This is an *operational* label (builder abandonment), not independent ground truth.
"""

from __future__ import annotations

import csv
import gc
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

_REPO_ROOT = Path(__file__).resolve().parent.parent

import numpy as np
import pandas as pd

from cryoem_mrc.halfmap_metrics import WINDOWED_HALFMAP_CORRELATION_KEY, load_windowed_halfmap_correlation
from thesis.incremental_prediction import load_metrics_dataframe
from cryoem_mrc.local_resolution import RESMAP_UNRESOLVED_SENTINEL_A, locres_resmap_path
from cryoem_mrc.manifest_policy import row_ca_metrics_eligible
from cryoem_mrc.map_grid import load_map_grid, resample_volume_onto_grid
from thesis.placement_utility import (
    PLACEMENT_Q_ROC_PREDICTORS,
    PREDICTOR_LABELS,
    PredictorId,
    _predictor_scores,
    _roc_points_from_scores,
    enrich_with_resmap_locres,
    placement_roc_positive_mask,
    rank_auc,
)
from cryoem_mrc.repo_paths import COHORT_MANIFEST, emd_output_dir, find_features_npz, halfmap_metrics_npz, resolve_halfmap_reliability_dir
from cryoem_mrc.structure_validation import (
    CaResidue,
    build_contour_mask,
    iter_ca_residues,
    load_cohort_manifest_row,
    sample_volume_at_ca,
)

BUILDER_OMISSION_MAX_GAP_DEFAULT = 20
BUILDER_OMISSION_MIN_SITES_DEFAULT = 30
BUILDER_OMISSION_MIN_CLASS_DEFAULT = 10


@dataclass(frozen=True)
class SequenceGap:
    """Missing sequence indices on one chain between two flanking Cα residues."""

    chain: str
    left: CaResidue
    right: CaResidue
    missing_seq_nums: tuple[int, ...]


@dataclass(frozen=True)
class BuilderOmissionMapStats:
    emdb_id: str
    n_built: int
    n_omission: int
    n_gaps: int
    frac_omission_resmap_finite: float
    frac_built_resmap_finite: float
    median_v_built: float
    median_v_omission: float


def iter_builder_omission_maps(manifest: Path = COHORT_MANIFEST) -> list[str]:
    """EMDB IDs with a deposited model suitable for builder-omission ROC."""
    if not manifest.is_file():
        return []
    ids: list[str] = []
    with manifest.open(newline="") as f:
        for row in csv.DictReader(f):
            eid = str(row.get("emdb_id", "")).strip()
            if not eid:
                continue
            src = str(row.get("flexibility_source", "")).strip()
            if src in {"excluded", "optional"}:
                continue
            if not row_ca_metrics_eligible(row):
                continue
            ids.append(eid)
    return ids


def enumerate_sequence_gaps(
    residues: Sequence[CaResidue],
    *,
    max_gap_length: int = BUILDER_OMISSION_MAX_GAP_DEFAULT,
) -> list[SequenceGap]:
    """Internal sequence gaps per chain: missing indices between min and max built seq_num."""
    by_chain: dict[str, list[CaResidue]] = {}
    for res in residues:
        by_chain.setdefault(res.chain, []).append(res)

    gaps: list[SequenceGap] = []
    for chain, chain_res in by_chain.items():
        # One representative Cα per integer seq_num (first altloc already stripped).
        by_seq: dict[int, CaResidue] = {}
        for res in chain_res:
            if res.seq_num not in by_seq:
                by_seq[res.seq_num] = res
        if len(by_seq) < 2:
            continue
        ordered = sorted(by_seq.items(), key=lambda item: item[0])
        for (s1, left), (s2, right) in zip(ordered, ordered[1:]):
            if s2 - s1 <= 1:
                continue
            missing = tuple(
                s for s in range(s1 + 1, s2) if (s2 - s1 - 1) <= max_gap_length
            )
            if not missing:
                continue
            if len(missing) > max_gap_length:
                missing = missing[:max_gap_length]
            gaps.append(
                SequenceGap(
                    chain=chain,
                    left=left,
                    right=right,
                    missing_seq_nums=missing,
                )
            )
    return gaps


def interpolate_gap_residue(
    gap: SequenceGap,
    seq_num: int,
) -> CaResidue:
    """Linear Cα interpolation between flanking deposited residues."""
    left = gap.left
    right = gap.right
    span = float(right.seq_num - left.seq_num)
    frac = float(seq_num - left.seq_num) / span
    return CaResidue(
        chain=gap.chain,
        seq_num=seq_num,
        seq_icode="",
        res_name="---",
        x=(1.0 - frac) * left.x + frac * right.x,
        y=(1.0 - frac) * left.y + frac * right.y,
        z=(1.0 - frac) * left.z + frac * right.z,
        b_iso=float("nan"),
        auth_chain=left.auth_chain or left.chain,
        auth_seq_num=seq_num,
    )


def _load_resmap_on_reference(
    emdb_id: str,
    reference_path: Path,
) -> np.ndarray | None:
    path = locres_resmap_path(emdb_id)
    if not path.is_file():
        return None
    ref_grid = load_map_grid(reference_path, dtype=np.float32)
    loc_grid = load_map_grid(path, dtype=np.float64)
    if loc_grid.shape_zyx != ref_grid.shape_zyx:
        return resample_volume_onto_grid(loc_grid, ref_grid).astype(np.float64)
    return np.asarray(loc_grid.data, dtype=np.float64)


def _sanitize_resmap(values: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64)
    bad = ~np.isfinite(out) | (out >= RESMAP_UNRESOLVED_SENTINEL_A)
    out = out.copy()
    out[bad] = np.nan
    return out


def tpr_at_fpr(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    target_fpr: float = 0.10,
) -> float:
    """TPR at the score threshold nearest ``target_fpr`` (higher score = more likely positive)."""
    y = y_true.astype(bool)
    s = np.asarray(scores, dtype=np.float64)
    m = np.isfinite(s)
    y = y[m]
    s = s[m]
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    fpr, tpr, _ = _roc_points_from_scores(y, s)
    if not fpr:
        return float("nan")
    fpr_a = np.asarray(fpr, dtype=np.float64)
    tpr_a = np.asarray(tpr, dtype=np.float64)
    idx = int(np.argmin(np.abs(fpr_a - target_fpr)))
    return float(tpr_a[idx])


def _find_reliability_npz(emdb_id: str) -> Path | None:
    """Locate cached half-map reliability bundle (canonical or legacy halfmap-qc path)."""
    candidates = (
        resolve_halfmap_reliability_dir(emdb_id) / "reliability.npz",
        emd_output_dir(emdb_id) / "halfmap-qc" / "reliability" / "reliability.npz",
    )
    for path in candidates:
        if path.is_file():
            return path
    return None


def _load_reliability_volumes(
    emdb_id: str,
    *,
    reference_path: Path,
    half1_path: Path,
    half2_path: Path,
    contour: float,
    allow_recompute: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Load reliability, build-zone, and V volumes without reloading half-maps when cached."""
    npz_path = _find_reliability_npz(emdb_id)
    if npz_path is not None:
        with np.load(npz_path, allow_pickle=False) as d:
            rel = np.asarray(d["reliability_score"], dtype=np.float32)
            zone = np.rint(np.asarray(d["build_zone"], dtype=np.float32)).astype(np.uint8)
            v_key = "reliability_smoothness"
            if v_key not in d:
                return None
            v = np.asarray(d[v_key], dtype=np.float32)
        return rel, zone, v

    if not allow_recompute:
        return None

    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from cryoem_mrc.reliability_volumes import load_reliability_mrc_pair, recompute_reliability_volumes

    try:
        reliability_score, build_zone = load_reliability_mrc_pair(emdb_id)
        _, v_metric_vol = recompute_reliability_volumes(
            emdb_id,
            reference_path=reference_path,
            half1_path=half1_path,
            half2_path=half2_path,
            contour=contour,
        )
    except (FileNotFoundError, ValueError):
        return None
    return reliability_score, build_zone, v_metric_vol


def _attach_coordinates_from_pdb(df: pd.DataFrame, pdb_path: Path) -> pd.DataFrame:
    """Join mmCIF Cα coordinates onto a per-residue metrics table."""
    lookup = {(r.chain, r.seq_num): r for r in iter_ca_residues(pdb_path)}
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    for row in df.itertuples(index=False):
        res = lookup.get((str(row.chain), int(row.seq_num)))
        if res is None:
            xs.append(float("nan"))
            ys.append(float("nan"))
            zs.append(float("nan"))
        else:
            xs.append(res.x)
            ys.append(res.y)
            zs.append(res.z)
    out = df.copy()
    out["x"] = xs
    out["y"] = ys
    out["z"] = zs
    return out[out[["x", "y", "z"]].notna().all(axis=1)].copy()


def _dataframe_to_ca_residues(df: pd.DataFrame) -> list[CaResidue]:
    residues: list[CaResidue] = []
    for row in df.itertuples(index=False):
        residues.append(
            CaResidue(
                chain=str(row.chain),
                seq_num=int(row.seq_num),
                seq_icode=str(getattr(row, "seq_icode", "") or ""),
                res_name=str(getattr(row, "res_name", "UNK") or "UNK"),
                x=float(row.x),
                y=float(row.y),
                z=float(row.z),
                b_iso=float(getattr(row, "b_iso", float("nan"))),
                auth_chain=str(getattr(row, "auth_chain", row.chain) or row.chain),
                auth_seq_num=int(getattr(row, "auth_seq_num", row.seq_num) or row.seq_num),
            )
        )
    return residues


def _resample_volume_metrics_at_rows(
    df: pd.DataFrame,
    *,
    grid,
    sphere_radius_a: float,
    reliability_score: np.ndarray,
    build_zone: np.ndarray,
    v_metric_vol: np.ndarray,
    cc_vol: np.ndarray | None,
    var_vol: np.ndarray | None,
    resmap_vol: np.ndarray | None,
) -> pd.DataFrame:
    """Overwrite map-sampled columns so built and omission sites share one volume source."""
    out = df.copy()
    residues = _dataframe_to_ca_residues(out)
    out["reliability_score"] = sample_volume_at_ca(
        reliability_score, grid, residues, sphere_radius_a=sphere_radius_a
    )
    out["v_metric"] = sample_volume_at_ca(
        v_metric_vol, grid, residues, sphere_radius_a=sphere_radius_a
    )
    out["build_zone"] = np.rint(
        sample_volume_at_ca(
            build_zone.astype(np.float64),
            grid,
            residues,
            sphere_radius_a=sphere_radius_a,
        )
    ).astype(int)
    if cc_vol is not None:
        out[WINDOWED_HALFMAP_CORRELATION_KEY] = sample_volume_at_ca(
            cc_vol, grid, residues, sphere_radius_a=sphere_radius_a
        )
    if var_vol is not None:
        out["local_variance"] = sample_volume_at_ca(
            var_vol, grid, residues, sphere_radius_a=sphere_radius_a
        )
    if resmap_vol is not None:
        out["local_resolution_resmap"] = _sanitize_resmap(
            sample_volume_at_ca(
                resmap_vol, grid, residues, sphere_radius_a=sphere_radius_a
            )
        )
    return out


def build_builder_omission_frame(
    emdb_id: str,
    *,
    manifest: Path = COHORT_MANIFEST,
    sphere_radius_a: float = 2.0,
    max_gap_length: int = BUILDER_OMISSION_MAX_GAP_DEFAULT,
    min_built: int = BUILDER_OMISSION_MIN_CLASS_DEFAULT,
    min_omission: int = BUILDER_OMISSION_MIN_CLASS_DEFAULT,
    allow_lh_recompute: bool = False,
) -> tuple[pd.DataFrame, BuilderOmissionMapStats] | None:
    """
    Combine in-mask built Cα rows with in-mask interpolated gap-proxy omission sites.

    Returns ``None`` when the map lacks metrics, a structure, or enough class balance.
    """
    emdb_id = str(emdb_id).strip()
    metrics = load_metrics_dataframe(
        emdb_id, manifest=manifest, sphere_radius_a=sphere_radius_a
    )
    if metrics is None or metrics.empty:
        return None

    try:
        row = load_cohort_manifest_row(manifest, emdb_id)
    except KeyError:
        return None

    pdb_raw = str(row.get("flexibility_path_or_pdb", "")).strip()
    if not pdb_raw:
        return None
    pdb_path = Path(pdb_raw)
    if not pdb_path.is_file():
        return None

    ref_path = Path(row["reference_mrc"])
    contour = float(row["contour"])

    built = metrics.copy()
    if "in_contour_mask" in built.columns:
        built = built[built["in_contour_mask"].astype(bool)].copy()
    if built.empty:
        return None

    built["builder_omission"] = False
    if "emdb_id" not in built.columns:
        built["emdb_id"] = emdb_id

    built = enrich_with_resmap_locres(built, emdb_id)
    built = _attach_coordinates_from_pdb(built, pdb_path)
    if built.empty:
        return None

    gaps = enumerate_sequence_gaps(iter_ca_residues(pdb_path), max_gap_length=max_gap_length)
    if not gaps:
        return None

    volumes = _load_reliability_volumes(
        emdb_id,
        reference_path=ref_path,
        half1_path=Path(row["half1_path"]),
        half2_path=Path(row["half2_path"]),
        contour=contour,
        allow_recompute=allow_lh_recompute,
    )
    if volumes is None:
        return None
    reliability_score, build_zone, v_metric_vol = volumes

    grid = load_map_grid(ref_path, dtype=np.float32)
    reference_density = np.asarray(grid.data, dtype=np.float32)
    mask_vol = build_contour_mask(reference_density, contour).astype(np.float64)

    halfmap_npz = halfmap_metrics_npz(emdb_id)
    cc_vol = None
    if halfmap_npz.is_file():
        with np.load(halfmap_npz, allow_pickle=False) as hm:
            cc_vol = load_windowed_halfmap_correlation(hm)

    var_vol = None
    features_npz = find_features_npz(ref_path.parent, emdb_id, contour)
    if features_npz is not None and features_npz.is_file():
        with np.load(features_npz, allow_pickle=False) as feat:
            if "local_variance" in feat:
                var_vol = np.asarray(feat["local_variance"], dtype=np.float32)

    resmap_vol = _load_resmap_on_reference(emdb_id, ref_path)

    omission_residues: list[CaResidue] = []
    for gap in gaps:
        for seq_num in gap.missing_seq_nums:
            omission_residues.append(interpolate_gap_residue(gap, seq_num))
    if not omission_residues:
        return None

    in_mask = sample_volume_at_ca(
        mask_vol, grid, omission_residues, sphere_radius_a=sphere_radius_a
    )
    keep = in_mask >= 0.5
    omission_residues = [r for r, ok in zip(omission_residues, keep) if ok]
    if not omission_residues:
        return None

    rel_s = sample_volume_at_ca(
        reliability_score, grid, omission_residues, sphere_radius_a=sphere_radius_a
    )
    v_s = sample_volume_at_ca(
        v_metric_vol, grid, omission_residues, sphere_radius_a=sphere_radius_a
    )
    zone_s = sample_volume_at_ca(
        build_zone.astype(np.float64),
        grid,
        omission_residues,
        sphere_radius_a=sphere_radius_a,
    )
    cc_s = (
        sample_volume_at_ca(cc_vol, grid, omission_residues, sphere_radius_a=sphere_radius_a)
        if cc_vol is not None
        else np.full(len(omission_residues), np.nan)
    )
    var_s = (
        sample_volume_at_ca(var_vol, grid, omission_residues, sphere_radius_a=sphere_radius_a)
        if var_vol is not None
        else np.full(len(omission_residues), np.nan)
    )
    resmap_s = (
        _sanitize_resmap(
            sample_volume_at_ca(
                resmap_vol, grid, omission_residues, sphere_radius_a=sphere_radius_a
            )
        )
        if resmap_vol is not None
        else np.full(len(omission_residues), np.nan)
    )

    omission_df = pd.DataFrame(
        {
            "emdb_id": emdb_id,
            "chain": [r.chain for r in omission_residues],
            "seq_num": [r.seq_num for r in omission_residues],
            "seq_icode": [r.seq_icode for r in omission_residues],
            "res_name": [r.res_name for r in omission_residues],
            "reliability_score": rel_s,
            "v_metric": v_s,
            "build_zone": np.rint(zone_s).astype(int),
            WINDOWED_HALFMAP_CORRELATION_KEY: cc_s,
            "local_variance": var_s,
            "local_resolution": np.nan,
            "local_resolution_resmap": resmap_s,
            "in_contour_mask": True,
            "builder_omission": True,
            "site_kind": "omission",
        }
    )

    if int((~built["builder_omission"]).sum()) < min_built:
        return None
    if len(omission_df) < min_omission:
        return None

    built = _resample_volume_metrics_at_rows(
        built,
        grid=grid,
        sphere_radius_a=sphere_radius_a,
        reliability_score=reliability_score,
        build_zone=build_zone,
        v_metric_vol=v_metric_vol,
        cc_vol=cc_vol,
        var_vol=var_vol,
        resmap_vol=resmap_vol,
    )
    built["site_kind"] = "built"
    combined = pd.concat([built, omission_df], ignore_index=True, sort=False)

    stats = BuilderOmissionMapStats(
        emdb_id=emdb_id,
        n_built=int((~combined["builder_omission"]).sum()),
        n_omission=int(combined["builder_omission"].sum()),
        n_gaps=len(gaps),
        frac_omission_resmap_finite=_finite_frac(
            combined.loc[combined["builder_omission"], "local_resolution_resmap"]
        ),
        frac_built_resmap_finite=_finite_frac(
            combined.loc[~combined["builder_omission"], "local_resolution_resmap"]
        ),
        median_v_built=_median_finite(
            combined.loc[~combined["builder_omission"], "v_metric"]
        ),
        median_v_omission=_median_finite(
            combined.loc[combined["builder_omission"], "v_metric"]
        ),
    )
    return combined, stats


def _finite_frac(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce")
    if vals.empty:
        return float("nan")
    return float(vals.notna().sum() / len(vals))


def _median_finite(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce").to_numpy()
    finite = vals[np.isfinite(vals)]
    return float(np.median(finite)) if finite.size else float("nan")


def load_per_map_frames_for_builder_omission_roc(
    *,
    manifest: Path = COHORT_MANIFEST,
    sphere_radius_a: float = 2.0,
    max_gap_length: int = BUILDER_OMISSION_MAX_GAP_DEFAULT,
    min_total: int = BUILDER_OMISSION_MIN_SITES_DEFAULT,
    emdb_ids: Sequence[str] | None = None,
    allow_lh_recompute: bool = False,
) -> tuple[list[tuple[str, pd.DataFrame, float]], list[BuilderOmissionMapStats]]:
    """Load builder-omission frames for all eligible manifest rows."""
    frames: list[tuple[str, pd.DataFrame, float]] = []
    stats_rows: list[BuilderOmissionMapStats] = []
    candidates = list(emdb_ids) if emdb_ids is not None else iter_builder_omission_maps(manifest)
    for emdb_id in candidates:
        try:
            row = load_cohort_manifest_row(manifest, emdb_id)
            gres = float(row["global_resolution_a"]) if row.get("global_resolution_a") else float("nan")
        except (KeyError, ValueError):
            gres = float("nan")
        built = build_builder_omission_frame(
            emdb_id,
            manifest=manifest,
            sphere_radius_a=sphere_radius_a,
            max_gap_length=max_gap_length,
            allow_lh_recompute=allow_lh_recompute,
        )
        if built is None:
            continue
        df, stats = built
        if len(df) < min_total:
            continue
        frames.append((emdb_id, df, gres))
        stats_rows.append(stats)
        del df
        gc.collect()
    return frames, stats_rows


def summarize_builder_omission_roc_per_map(
    per_map_frames: Sequence[tuple[str, pd.DataFrame]],
    *,
    predictors: Sequence[PredictorId] = PLACEMENT_Q_ROC_PREDICTORS,
    tpr_fpr_target: float = 0.10,
) -> list[dict[str, object]]:
    """Per-map ROC rows for builder-omission ground truth."""
    rows: list[dict[str, object]] = []
    for emdb_id, df in per_map_frames:
        pos_m, positive = placement_roc_positive_mask(df, ground_truth="builder_omission")
        resmap_col = pd.to_numeric(df.get("local_resolution_resmap", np.nan), errors="coerce")
        resmap_defined = pos_m & resmap_col.notna().to_numpy()
        for pid in predictors:
            scores = _predictor_scores(df)[pid]
            m = pos_m & np.isfinite(scores)
            if int(m.sum()) < BUILDER_OMISSION_MIN_SITES_DEFAULT:
                continue
            y = positive[m]
            if int(y.sum()) == 0 or int((~y).sum()) == 0:
                continue
            auc = rank_auc(y, scores[m])
            tpr = tpr_at_fpr(y, scores[m], target_fpr=tpr_fpr_target)
            auc_resmap_defined = float("nan")
            m_rd = resmap_defined & np.isfinite(scores)
            if pid == "resmap_locres_worse_than_median" and int(m_rd.sum()) >= BUILDER_OMISSION_MIN_SITES_DEFAULT:
                y_rd = positive[m_rd]
                if int(y_rd.sum()) > 0 and int((~y_rd).sum()) > 0:
                    auc_resmap_defined = rank_auc(y_rd, scores[m_rd])
            rows.append(
                {
                    "emdb_id": str(emdb_id),
                    "ground_truth": "builder_omission",
                    "predictor": pid,
                    "label": PREDICTOR_LABELS[pid],
                    "auc": float(auc),
                    "auc_resmap_defined_sites": float(auc_resmap_defined),
                    "tpr_at_10pct_fpr": float(tpr),
                    "n_sites": int(m.sum()),
                    "n_omission": int(y.sum()),
                    "n_built": int((~y).sum()),
                    "frac_omission_resmap_finite": _finite_frac(
                        df.loc[df["builder_omission"], "local_resolution_resmap"]
                    ),
                }
            )
    return rows
