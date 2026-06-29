"""Operational utility analyses for pre-model placement guidance vs Q-scores.

Tier-1 Structure-paper analyses: low-Q enrichment, head-to-head flag rules,
calibration of reliability vs Q, mis-ranking under BlocRes, and per-map rank
recovery ρ(proxy, Q) compared across pre-model readouts.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .halfmap_metrics import WINDOWED_HALFMAP_CORRELATION_KEY
from .cohort_resolution import COHORT_RESOLUTION_BINS, median_rho_by_resolution_bin
from .incremental_prediction import (
    TARGET_Q,
    iter_eligible_emdb_ids,
    load_metrics_dataframe,
    load_qscore_target,
)
from .repo_paths import COHORT_MANIFEST, OUTPUTS_ROOT, emd_output_dir, resolve_halfmap_reliability_dir
from .structure_validation import load_cohort_manifest_row

from .qscore_cohort import QSCORE_PANEL_EXCLUDE, filter_emdb_ids, qscore_exclude_ids

# Maps excluded from ResMap parallel LOMO after auto-mask QC (sentinel / coverage / flat range).
RESMAP_QC_EXCLUDE = frozenset(
    {"9156", "24120", "11149", "33736", "52525", "41596", "11638"}
)

LocresMethodLomoPredictor = Literal[
    "blocres_locres_inmap_median",
    "blocres_locres_vs_global",
    "resmap_locres_inmap_median",
    "resmap_locres_vs_global",
    "monores_locres_inmap_median",
    "monores_locres_vs_global",
    "omit_zone",
]

LOCRES_METHOD_LOMO_LABELS: dict[LocresMethodLomoPredictor, str] = {
    "blocres_locres_inmap_median": "BlocRes worse than in-map median (Å)",
    "blocres_locres_vs_global": "BlocRes worse than deposited global resolution (Å)",
    "resmap_locres_inmap_median": "ResMap worse than in-map median (Å)",
    "resmap_locres_vs_global": "ResMap worse than deposited global resolution (Å)",
    "monores_locres_inmap_median": "MonoRes worse than in-map median (Å)",
    "monores_locres_vs_global": "MonoRes worse than deposited global resolution (Å)",
    "omit_zone": "Omit build zone (reliability tercile)",
}

LOCRES_METHOD_LOMO_PREDICTORS: tuple[LocresMethodLomoPredictor, ...] = (
    "blocres_locres_inmap_median",
    "blocres_locres_vs_global",
    "resmap_locres_inmap_median",
    "resmap_locres_vs_global",
    "monores_locres_inmap_median",
    "monores_locres_vs_global",
    "omit_zone",
)

PredictorId = Literal[
    "omit_zone",
    "reliability_below_0_33",
    "cc_below_0_5",
    "locres_worse_than_median",
    "variance_above_median",
]

PREDICTOR_LABELS: dict[PredictorId, str] = {
    "omit_zone": "Omit build zone (tercile)",
    "reliability_below_0_33": "Reliability score < 0.33",
    "cc_below_0_5": "Windowed half-map CC < 0.5",
    "locres_worse_than_median": "BlocRes worse than in-map median (Å)",
    "variance_above_median": "Local variance above in-map median",
}

# Rank-recovery bar charts compare proxies on a common axis: higher ⇒ better Q coupling.
# BlocRes is stored in Å (larger = blurrier), so raw ρ(Q, locres) is usually negative
# when the median-split flag (loc > in-map median) tracks low Q.
RANK_RECOVERY_PROXY_KEYS: tuple[str, ...] = (
    "spearman_q_vs_reliability",
    "spearman_q_vs_cc",
    "spearman_q_vs_locres",
    "spearman_q_vs_variance",
    "spearman_q_vs_v",
)

RANK_RECOVERY_PROXY_LABELS: dict[str, str] = {
    "spearman_q_vs_reliability": "reliability",
    "spearman_q_vs_cc": "windowed CC",
    "spearman_q_vs_locres": "BlocRes (sharpness)",
    "spearman_q_vs_variance": "variance",
    "spearman_q_vs_v": "constraint V",
}

RANK_RECOVERY_Q_COUPLING_SIGN: dict[str, float] = {
    "spearman_q_vs_reliability": 1.0,
    "spearman_q_vs_cc": 1.0,
    "spearman_q_vs_locres": -1.0,
    "spearman_q_vs_variance": 1.0,
    "spearman_q_vs_v": 1.0,
}


def aligned_rank_recovery_rho(raw_rho: float, proxy_key: str) -> float:
    """Sign-align ρ(Q, proxy) so larger bars mean stronger placement-consistent coupling."""
    if not np.isfinite(raw_rho):
        return float("nan")
    return float(RANK_RECOVERY_Q_COUPLING_SIGN.get(proxy_key, 1.0) * raw_rho)


@dataclass(frozen=True)
class LowQEnrichmentRow:
    emdb_id: str
    display_name: str
    global_resolution_a: float
    n_in_mask: int
    n_low_q: int
    q_threshold: float
    frac_low_q: float
    frac_low_q_in_omit_zone: float
    frac_low_q_reliability_below: float
    frac_low_q_cc_below: float
    frac_low_q_locres_worse_than_median: float
    frac_low_q_variance_above_median: float
    omit_zone_baseline: float


@dataclass(frozen=True)
class PredictorUtilityRow:
    predictor: PredictorId
    n_maps: int
    n_residues_pooled: int
    n_low_q_pooled: int
    median_frac_low_q_flagged: float
    pooled_frac_low_q_flagged: float
    pooled_sensitivity: float
    pooled_specificity: float
    pooled_balanced_accuracy: float
    median_map_balanced_accuracy: float
    median_map_auc: float
    median_map_spearman_vs_q: float


@dataclass(frozen=True)
class RankRecoveryRow:
    emdb_id: str
    global_resolution_a: float
    n_in_mask: int
    spearman_q_vs_reliability: float
    spearman_q_vs_cc: float
    spearman_q_vs_locres: float
    spearman_q_vs_variance: float
    spearman_q_vs_v: float


def median_aligned_rank_recovery(
    rows: Sequence[RankRecoveryRow],
    proxy_key: str,
) -> float:
    vals = [
        aligned_rank_recovery_rho(getattr(r, proxy_key), proxy_key)
        for r in rows
        if np.isfinite(getattr(r, proxy_key))
    ]
    return float(np.median(vals)) if vals else float("nan")


@dataclass(frozen=True)
class MisrankingRow:
    emdb_id: str
    global_resolution_a: float
    n_in_mask: int
    frac_sharp_locres_low_q_tercile: float
    frac_omit_zone_low_q_tercile: float
    frac_cc_above_0_7_low_q_tercile: float


@dataclass(frozen=True)
class CalibrationBin:
    reliability_bin_lo: float
    reliability_bin_hi: float
    n_residues: int
    mean_q: float
    median_q: float


@dataclass(frozen=True)
class PlacementUtilitySummary:
    q_threshold: float
    enrichment_rows: tuple[LowQEnrichmentRow, ...]
    predictor_rows: tuple[PredictorUtilityRow, ...]
    rank_recovery_rows: tuple[RankRecoveryRow, ...]
    misranking_rows: tuple[MisrankingRow, ...]
    calibration_bins: tuple[CalibrationBin, ...]
    resolution_bins: dict[str, float] = field(default_factory=dict)


def _global_resolution(manifest_row: dict[str, str]) -> float:
    raw = manifest_row.get("global_resolution_a", "").strip()
    if not raw:
        return float("nan")
    try:
        return float(raw)
    except ValueError:
        return float("nan")


def load_map_with_qscore(
    emdb_id: str,
    *,
    manifest: Path = COHORT_MANIFEST,
    sphere_radius_a: float = 2.0,
) -> pd.DataFrame | None:
    """In-mask per-residue metrics merged with Q-scores when available."""
    metrics = load_metrics_dataframe(
        emdb_id, manifest=manifest, sphere_radius_a=sphere_radius_a
    )
    if metrics is None:
        return None
    merged = load_qscore_target(metrics, emdb_id)
    if merged is None:
        return None
    if "in_contour_mask" in merged.columns:
        merged = merged[merged["in_contour_mask"].astype(bool)].copy()
    merged["emdb_id"] = str(emdb_id)
    return merged


def iter_qscore_maps(
    *,
    manifest: Path = COHORT_MANIFEST,
    exclude: frozenset[str] | None = None,
    core: bool = False,
) -> list[str]:
    exclude_set = exclude if exclude is not None else qscore_exclude_ids(core=core)
    ids = iter_eligible_emdb_ids(TARGET_Q, manifest=manifest, qscore_exclude=frozenset())
    return filter_emdb_ids(ids, core=core) if core else [i for i in ids if i not in exclude_set]


def _finite_spearman(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 10:
        return float("nan")
    xm = x[m]
    ym = y[m]
    if np.nanstd(xm) == 0 or np.nanstd(ym) == 0:
        return float("nan")
    rho, _ = stats.spearmanr(xm, ym)
    return float(rho)


def rank_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """ROC AUC; higher ``scores`` ⇒ more likely ``y_true == 1``."""
    y = y_true.astype(bool)
    s = np.asarray(scores, dtype=np.float64)
    m = np.isfinite(s)
    y = y[m]
    s = s[m]
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = stats.rankdata(s)
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y = y_true.astype(bool)
    p = y_pred.astype(bool)
    m = np.isfinite(y_true)  # both are bool arrays; all finite
    y = y[m]
    p = p[m]
    tp = int((p & y).sum())
    tn = int((~p & ~y).sum())
    fp = int((p & ~y).sum())
    fn = int((~p & y).sum())
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    if not np.isfinite(sens) or not np.isfinite(spec):
        return float("nan")
    return float(0.5 * (sens + spec))


def _predictor_flags(df: pd.DataFrame) -> dict[PredictorId, np.ndarray]:
    rel = pd.to_numeric(df["reliability_score"], errors="coerce").to_numpy()
    cc = pd.to_numeric(df[WINDOWED_HALFMAP_CORRELATION_KEY], errors="coerce").to_numpy()
    loc = pd.to_numeric(df["local_resolution"], errors="coerce").to_numpy()
    var = pd.to_numeric(df["local_variance"], errors="coerce").to_numpy()
    zone = pd.to_numeric(df["build_zone"], errors="coerce").to_numpy(dtype=np.int32)

    loc_med = np.nanmedian(loc) if np.isfinite(loc).any() else float("nan")
    var_med = np.nanmedian(var) if np.isfinite(var).any() else float("nan")

    # BlocRes Å increases with blur; flag residues worse than the in-map median.
    return {
        "omit_zone": zone == 0,
        "reliability_below_0_33": rel < 0.33,
        "cc_below_0_5": cc < 0.5,
        "locres_worse_than_median": loc > loc_med if np.isfinite(loc_med) else np.zeros(len(df), bool),
        "variance_above_median": var > var_med if np.isfinite(var_med) else np.zeros(len(df), bool),
    }


def _predictor_scores(df: pd.DataFrame) -> dict[PredictorId, np.ndarray]:
    """Continuous scores where higher ⇒ more likely low Q (for AUC)."""
    rel = pd.to_numeric(df["reliability_score"], errors="coerce").to_numpy()
    cc = pd.to_numeric(df[WINDOWED_HALFMAP_CORRELATION_KEY], errors="coerce").to_numpy()
    loc = pd.to_numeric(df["local_resolution"], errors="coerce").to_numpy()
    var = pd.to_numeric(df["local_variance"], errors="coerce").to_numpy()
    zone = pd.to_numeric(df["build_zone"], errors="coerce").to_numpy()

    loc_score = loc.copy()
    cc_score = -cc.copy()
    var_score = var.copy()
    rel_score = 1.0 - rel
    zone_score = np.where(zone == 0, 1.0, np.where(zone == 1, 0.5, 0.0))

    return {
        "omit_zone": zone_score,
        "reliability_below_0_33": rel_score,
        "cc_below_0_5": cc_score,
        "locres_worse_than_median": loc_score,
        "variance_above_median": var_score,
    }


def compute_low_q_enrichment_row(
    df: pd.DataFrame,
    *,
    emdb_id: str,
    display_name: str = "",
    global_resolution_a: float = float("nan"),
    q_threshold: float = 0.5,
) -> LowQEnrichmentRow | None:
    q = pd.to_numeric(df["q_score"], errors="coerce").to_numpy()
    m = np.isfinite(q)
    n = int(m.sum())
    if n == 0:
        return None

    low = q < q_threshold
    n_low = int(low.sum())
    flags = _predictor_flags(df)

    def frac_flag(name: PredictorId) -> float:
        if n_low == 0:
            return float("nan")
        return float(flags[name][m][low[m]].mean())

    omit_base = float((pd.to_numeric(df["build_zone"], errors="coerce") == 0)[m].mean())

    return LowQEnrichmentRow(
        emdb_id=str(emdb_id),
        display_name=display_name,
        global_resolution_a=global_resolution_a,
        n_in_mask=n,
        n_low_q=n_low,
        q_threshold=q_threshold,
        frac_low_q=float(low[m].mean()),
        frac_low_q_in_omit_zone=frac_flag("omit_zone"),
        frac_low_q_reliability_below=frac_flag("reliability_below_0_33"),
        frac_low_q_cc_below=frac_flag("cc_below_0_5"),
        frac_low_q_locres_worse_than_median=frac_flag("locres_worse_than_median"),
        frac_low_q_variance_above_median=frac_flag("variance_above_median"),
        omit_zone_baseline=omit_base,
    )


def compute_misranking_row(
    df: pd.DataFrame,
    *,
    emdb_id: str,
    global_resolution_a: float = float("nan"),
    q_tercile: Literal["bottom"] = "bottom",
) -> MisrankingRow | None:
    q = pd.to_numeric(df["q_score"], errors="coerce").to_numpy()
    loc = pd.to_numeric(df["local_resolution"], errors="coerce").to_numpy()
    cc = pd.to_numeric(df[WINDOWED_HALFMAP_CORRELATION_KEY], errors="coerce").to_numpy()
    zone = pd.to_numeric(df["build_zone"], errors="coerce").to_numpy()

    m = np.isfinite(q)
    if m.sum() < 30:
        return None

    q_m = q[m]
    t1 = np.percentile(q_m, 100 / 3)
    low_q = np.zeros_like(q, dtype=bool)
    low_q[m] = q_m <= t1

    loc_m = loc[m]
    sharp_locres = np.zeros_like(loc, dtype=bool)
    if np.isfinite(loc_m).sum() >= 10:
        sharp_locres[m] = loc_m <= np.nanmedian(loc_m)

    cc_m = cc[m]
    high_cc = np.zeros_like(cc, dtype=bool)
    if np.isfinite(cc_m).sum() >= 10:
        high_cc[m] = cc_m >= 0.7

    n = int(m.sum())

    def frac(cond: np.ndarray) -> float:
        n_hit = int(low_q.sum())
        if n_hit == 0:
            return float("nan")
        return float(cond[low_q].mean())

    return MisrankingRow(
        emdb_id=str(emdb_id),
        global_resolution_a=global_resolution_a,
        n_in_mask=n,
        frac_sharp_locres_low_q_tercile=frac(sharp_locres),
        frac_omit_zone_low_q_tercile=frac(zone == 0),
        frac_cc_above_0_7_low_q_tercile=frac(high_cc),
    )


def compute_rank_recovery_row(
    df: pd.DataFrame,
    *,
    emdb_id: str,
    global_resolution_a: float = float("nan"),
) -> RankRecoveryRow | None:
    q = pd.to_numeric(df["q_score"], errors="coerce").to_numpy()
    m = np.isfinite(q)
    if m.sum() < 30:
        return None

    rel = pd.to_numeric(df["reliability_score"], errors="coerce").to_numpy()
    cc = pd.to_numeric(df[WINDOWED_HALFMAP_CORRELATION_KEY], errors="coerce").to_numpy()
    loc = pd.to_numeric(df["local_resolution"], errors="coerce").to_numpy()
    var = pd.to_numeric(df["local_variance"], errors="coerce").to_numpy()
    v = pd.to_numeric(df.get("v_metric", np.nan), errors="coerce").to_numpy()

    return RankRecoveryRow(
        emdb_id=str(emdb_id),
        global_resolution_a=global_resolution_a,
        n_in_mask=int(m.sum()),
        spearman_q_vs_reliability=_finite_spearman(q, rel),
        spearman_q_vs_cc=_finite_spearman(q, cc),
        spearman_q_vs_locres=_finite_spearman(q, loc),
        spearman_q_vs_variance=_finite_spearman(q, var),
        spearman_q_vs_v=_finite_spearman(q, v),
    )


def compute_calibration_bins(
    frames: Sequence[pd.DataFrame],
    *,
    n_bins: int = 10,
) -> tuple[CalibrationBin, ...]:
    rel_all: list[float] = []
    q_all: list[float] = []
    for df in frames:
        rel = pd.to_numeric(df["reliability_score"], errors="coerce").to_numpy()
        q = pd.to_numeric(df["q_score"], errors="coerce").to_numpy()
        m = np.isfinite(rel) & np.isfinite(q)
        rel_all.extend(rel[m].tolist())
        q_all.extend(q[m].tolist())

    if not rel_all:
        return ()

    rel_a = np.asarray(rel_all, dtype=np.float64)
    q_a = np.asarray(q_all, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[CalibrationBin] = []
    for i in range(n_bins):
        lo, hi = float(edges[i]), float(edges[i + 1])
        if i < n_bins - 1:
            mask = (rel_a >= lo) & (rel_a < hi)
        else:
            mask = (rel_a >= lo) & (rel_a <= hi)
        if not mask.any():
            continue
        bins.append(
            CalibrationBin(
                reliability_bin_lo=lo,
                reliability_bin_hi=hi,
                n_residues=int(mask.sum()),
                mean_q=float(q_a[mask].mean()),
                median_q=float(np.median(q_a[mask])),
            )
        )
    return tuple(bins)


def _summarize_predictor(
    predictor: PredictorId,
    per_map_frames: Sequence[tuple[str, pd.DataFrame]],
    *,
    q_threshold: float,
) -> PredictorUtilityRow:
    flags_list: list[np.ndarray] = []
    scores_list: list[np.ndarray] = []
    low_q_list: list[np.ndarray] = []
    frac_flagged_per_map: list[float] = []
    ba_per_map: list[float] = []
    auc_per_map: list[float] = []
    rho_per_map: list[float] = []

    for _emd, df in per_map_frames:
        q = pd.to_numeric(df["q_score"], errors="coerce").to_numpy()
        m = np.isfinite(q)
        if m.sum() < 10:
            continue
        low = q < q_threshold
        flags = _predictor_flags(df)[predictor]
        scores = _predictor_scores(df)[predictor]

        flags_list.append(flags[m])
        scores_list.append(scores[m])
        low_q_list.append(low[m])

        n_low = int(low[m].sum())
        if n_low > 0:
            frac_flagged_per_map.append(float(flags[m][low[m]].mean()))
        ba_per_map.append(balanced_accuracy(low[m], flags[m]))
        auc_per_map.append(rank_auc(low[m], scores[m]))
        rho_per_map.append(_finite_spearman(q[m], scores[m]))

    if not low_q_list:
        nan = float("nan")
        return PredictorUtilityRow(
            predictor=predictor,
            n_maps=0,
            n_residues_pooled=0,
            n_low_q_pooled=0,
            median_frac_low_q_flagged=nan,
            pooled_frac_low_q_flagged=nan,
            pooled_sensitivity=nan,
            pooled_specificity=nan,
            pooled_balanced_accuracy=nan,
            median_map_balanced_accuracy=nan,
            median_map_auc=nan,
            median_map_spearman_vs_q=nan,
        )

    flags_p = np.concatenate(flags_list)
    low_p = np.concatenate(low_q_list)
    n_low_p = int(low_p.sum())
    pooled_frac = float(flags_p[low_p].mean()) if n_low_p else float("nan")

    tp = int((flags_p & low_p).sum())
    fn = int((~flags_p & low_p).sum())
    fp = int((flags_p & ~low_p).sum())
    tn = int((~flags_p & ~low_p).sum())
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    pooled_ba = float(0.5 * (sens + spec)) if np.isfinite(sens) and np.isfinite(spec) else float("nan")

    return PredictorUtilityRow(
        predictor=predictor,
        n_maps=len(frac_flagged_per_map),
        n_residues_pooled=int(len(low_p)),
        n_low_q_pooled=n_low_p,
        median_frac_low_q_flagged=float(np.nanmedian(frac_flagged_per_map)),
        pooled_frac_low_q_flagged=pooled_frac,
        pooled_sensitivity=float(sens),
        pooled_specificity=float(spec),
        pooled_balanced_accuracy=pooled_ba,
        median_map_balanced_accuracy=float(np.nanmedian(ba_per_map)),
        median_map_auc=float(np.nanmedian(auc_per_map)),
        median_map_spearman_vs_q=float(np.nanmedian(rho_per_map)),
    )


def run_placement_utility_analysis(
    *,
    manifest: Path = COHORT_MANIFEST,
    q_threshold: float = 0.5,
    sphere_radius_a: float = 2.0,
    exclude: frozenset[str] | None = None,
) -> PlacementUtilitySummary:
    """Run cohort placement-utility analyses for maps with Q-score validation."""
    emdb_ids = iter_qscore_maps(manifest=manifest, exclude=exclude)
    enrichment: list[LowQEnrichmentRow] = []
    rank_rows: list[RankRecoveryRow] = []
    misrank: list[MisrankingRow] = []
    per_map_frames: list[tuple[str, pd.DataFrame]] = []
    all_frames: list[pd.DataFrame] = []

    for emdb_id in emdb_ids:
        try:
            row = load_cohort_manifest_row(manifest, emdb_id)
        except KeyError:
            continue
        display_name = row.get("display_name", "").strip()
        gres = _global_resolution(row)

        df = load_map_with_qscore(
            emdb_id, manifest=manifest, sphere_radius_a=sphere_radius_a
        )
        if df is None or df.empty:
            continue

        per_map_frames.append((emdb_id, df))
        all_frames.append(df)

        enr = compute_low_q_enrichment_row(
            df,
            emdb_id=emdb_id,
            display_name=display_name,
            global_resolution_a=gres,
            q_threshold=q_threshold,
        )
        if enr is not None:
            enrichment.append(enr)

        rr = compute_rank_recovery_row(df, emdb_id=emdb_id, global_resolution_a=gres)
        if rr is not None:
            rank_rows.append(rr)

        mr = compute_misranking_row(df, emdb_id=emdb_id, global_resolution_a=gres)
        if mr is not None:
            misrank.append(mr)

    predictors: list[PredictorUtilityRow] = []
    for pid in PREDICTOR_LABELS:
        predictors.append(
            _summarize_predictor(pid, per_map_frames, q_threshold=q_threshold)
        )

    cal = compute_calibration_bins(all_frames)

    # Resolution-bin medians for ρ(Q, reliability) — matches cohort figure bins.
    res_bins: dict[str, float] = {}
    if rank_rows:
        arr = [
            (r.global_resolution_a, r.spearman_q_vs_reliability)
            for r in rank_rows
            if np.isfinite(r.global_resolution_a) and np.isfinite(r.spearman_q_vs_reliability)
        ]
        if arr:
            res_bins = median_rho_by_resolution_bin(
                arr,
                prefix="median_spearman",
                metric="q_reliability",
            )

    return PlacementUtilitySummary(
        q_threshold=q_threshold,
        enrichment_rows=tuple(enrichment),
        predictor_rows=tuple(predictors),
        rank_recovery_rows=tuple(rank_rows),
        misranking_rows=tuple(misrank),
        calibration_bins=cal,
        resolution_bins=res_bins,
    )


def write_placement_utility_csvs(
    summary: PlacementUtilitySummary,
    out_dir: Path,
) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    p_enr = out_dir / "placement_low_q_enrichment.csv"
    with p_enr.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "emdb_id",
                "display_name",
                "global_resolution_a",
                "n_in_mask",
                "n_low_q",
                "q_threshold",
                "frac_low_q",
                "frac_low_q_in_omit_zone",
                "frac_low_q_reliability_below",
                "frac_low_q_cc_below",
                "frac_low_q_locres_worse_than_median",
                "frac_low_q_variance_above_median",
                "omit_zone_baseline",
            ],
        )
        w.writeheader()
        for r in summary.enrichment_rows:
            w.writerow(
                {
                    "emdb_id": r.emdb_id,
                    "display_name": r.display_name,
                    "global_resolution_a": f"{r.global_resolution_a:.2f}"
                    if np.isfinite(r.global_resolution_a)
                    else "",
                    "n_in_mask": r.n_in_mask,
                    "n_low_q": r.n_low_q,
                    "q_threshold": f"{r.q_threshold:.2f}",
                    "frac_low_q": f"{r.frac_low_q:.4f}",
                    "frac_low_q_in_omit_zone": f"{r.frac_low_q_in_omit_zone:.4f}",
                    "frac_low_q_reliability_below": f"{r.frac_low_q_reliability_below:.4f}",
                    "frac_low_q_cc_below": f"{r.frac_low_q_cc_below:.4f}",
                    "frac_low_q_locres_worse_than_median": f"{r.frac_low_q_locres_worse_than_median:.4f}",
                    "frac_low_q_variance_above_median": f"{r.frac_low_q_variance_above_median:.4f}",
                    "omit_zone_baseline": f"{r.omit_zone_baseline:.4f}",
                }
            )
    paths["enrichment"] = p_enr

    p_pred = out_dir / "placement_predictor_head_to_head.csv"
    with p_pred.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "predictor",
                "label",
                "n_maps",
                "n_residues_pooled",
                "n_low_q_pooled",
                "median_frac_low_q_flagged",
                "pooled_frac_low_q_flagged",
                "pooled_sensitivity",
                "pooled_specificity",
                "pooled_balanced_accuracy",
                "median_map_balanced_accuracy",
                "median_map_auc",
                "median_map_spearman_vs_q",
            ],
        )
        w.writeheader()
        for r in summary.predictor_rows:
            w.writerow(
                {
                    "predictor": r.predictor,
                    "label": PREDICTOR_LABELS[r.predictor],
                    "n_maps": r.n_maps,
                    "n_residues_pooled": r.n_residues_pooled,
                    "n_low_q_pooled": r.n_low_q_pooled,
                    "median_frac_low_q_flagged": f"{r.median_frac_low_q_flagged:.4f}",
                    "pooled_frac_low_q_flagged": f"{r.pooled_frac_low_q_flagged:.4f}",
                    "pooled_sensitivity": f"{r.pooled_sensitivity:.4f}",
                    "pooled_specificity": f"{r.pooled_specificity:.4f}",
                    "pooled_balanced_accuracy": f"{r.pooled_balanced_accuracy:.4f}",
                    "median_map_balanced_accuracy": f"{r.median_map_balanced_accuracy:.4f}",
                    "median_map_auc": f"{r.median_map_auc:.4f}",
                    "median_map_spearman_vs_q": f"{r.median_map_spearman_vs_q:.4f}",
                }
            )
    paths["predictors"] = p_pred

    p_rr = out_dir / "placement_rank_recovery.csv"
    with p_rr.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "emdb_id",
                "global_resolution_a",
                "n_in_mask",
                "spearman_q_vs_reliability",
                "spearman_q_vs_cc",
                "spearman_q_vs_locres",
                "spearman_q_vs_variance",
                "spearman_q_vs_v",
            ],
        )
        w.writeheader()
        for r in summary.rank_recovery_rows:
            w.writerow(
                {
                    "emdb_id": r.emdb_id,
                    "global_resolution_a": f"{r.global_resolution_a:.2f}"
                    if np.isfinite(r.global_resolution_a)
                    else "",
                    "n_in_mask": r.n_in_mask,
                    "spearman_q_vs_reliability": f"{r.spearman_q_vs_reliability:.4f}",
                    "spearman_q_vs_cc": f"{r.spearman_q_vs_cc:.4f}",
                    "spearman_q_vs_locres": f"{r.spearman_q_vs_locres:.4f}",
                    "spearman_q_vs_variance": f"{r.spearman_q_vs_variance:.4f}",
                    "spearman_q_vs_v": f"{r.spearman_q_vs_v:.4f}",
                }
            )
    paths["rank_recovery"] = p_rr

    p_mr = out_dir / "placement_misranking.csv"
    with p_mr.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "emdb_id",
                "global_resolution_a",
                "n_in_mask",
                "frac_sharp_locres_low_q_tercile",
                "frac_omit_zone_low_q_tercile",
                "frac_cc_above_0_7_low_q_tercile",
            ],
        )
        w.writeheader()
        for r in summary.misranking_rows:
            w.writerow(
                {
                    "emdb_id": r.emdb_id,
                    "global_resolution_a": f"{r.global_resolution_a:.2f}"
                    if np.isfinite(r.global_resolution_a)
                    else "",
                    "n_in_mask": r.n_in_mask,
                    "frac_sharp_locres_low_q_tercile": f"{r.frac_sharp_locres_low_q_tercile:.4f}",
                    "frac_omit_zone_low_q_tercile": f"{r.frac_omit_zone_low_q_tercile:.4f}",
                    "frac_cc_above_0_7_low_q_tercile": f"{r.frac_cc_above_0_7_low_q_tercile:.4f}",
                }
            )
    paths["misranking"] = p_mr

    p_cal = out_dir / "placement_q_calibration_bins.csv"
    with p_cal.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "reliability_bin_lo",
                "reliability_bin_hi",
                "n_residues",
                "mean_q",
                "median_q",
            ],
        )
        w.writeheader()
        for b in summary.calibration_bins:
            w.writerow(
                {
                    "reliability_bin_lo": f"{b.reliability_bin_lo:.2f}",
                    "reliability_bin_hi": f"{b.reliability_bin_hi:.2f}",
                    "n_residues": b.n_residues,
                    "mean_q": f"{b.mean_q:.4f}",
                    "median_q": f"{b.median_q:.4f}",
                }
            )
    paths["calibration"] = p_cal

    return paths


def write_placement_utility_markdown(
    summary: PlacementUtilitySummary,
    path: Path,
) -> Path:
    """Human-readable summary for thesis / Structure paper supplement."""
    path = Path(path)
    lines: list[str] = [
        "# Placement utility analysis (Q-score operational validation)",
        "",
        f"Low-Q definition: Q-score < **{summary.q_threshold:.2f}** at in-mask Cα.",
        f"Maps analyzed: **{len(summary.enrichment_rows)}** with Q-score validation.",
        "",
        "## Tier 1 — Low-Q enrichment",
        "",
        "Among residues with Q below threshold, fraction flagged by each pre-model readout.",
        "Omit-zone baseline ≈ 0.33 by construction (tercile). Enrichment above baseline ⇒ utility.",
        "",
    ]

    if summary.enrichment_rows:
        def med(attr: str) -> float:
            vals = [getattr(r, attr) for r in summary.enrichment_rows]
            vals = [v for v in vals if np.isfinite(v)]
            return float(np.median(vals)) if vals else float("nan")

        lines.extend(
            [
                "| Readout | Median frac. of low-Q residues flagged |",
                "|---------|----------------------------------------|",
                f"| Omit zone | {med('frac_low_q_in_omit_zone'):.3f} |",
                f"| Reliability < 0.33 | {med('frac_low_q_reliability_below'):.3f} |",
                f"| CC < 0.5 | {med('frac_low_q_cc_below'):.3f} |",
                f"| BlocRes worse than median | {med('frac_low_q_locres_worse_than_median'):.3f} |",
                f"| Variance above median | {med('frac_low_q_variance_above_median'):.3f} |",
                f"| Omit-zone baseline (all Cα) | {med('omit_zone_baseline'):.3f} |",
                "",
            ]
        )

    lines.extend(["## Tier 1 — Head-to-head predictors (pooled cohort)", ""])
    if summary.predictor_rows:
        lines.extend(
            [
                "| Predictor | Pooled frac. low-Q flagged | Pooled BA | Median map AUC |",
                "|-----------|----------------------------|-----------|----------------|",
            ]
        )
        for r in summary.predictor_rows:
            lines.append(
                f"| {PREDICTOR_LABELS[r.predictor]} | "
                f"{r.pooled_frac_low_q_flagged:.3f} | "
                f"{r.pooled_balanced_accuracy:.3f} | "
                f"{r.median_map_auc:.3f} |"
            )
        lines.append("")

    lines.extend(["## Tier 2 — Rank recovery ρ(Q, proxy) medians", ""])
    if summary.rank_recovery_rows:
        def med_rr(attr: str) -> float:
            return median_aligned_rank_recovery(summary.rank_recovery_rows, attr)

        lines.extend(
            [
                f"- ρ(Q, reliability): **{med_rr('spearman_q_vs_reliability'):.3f}**",
                f"- ρ(Q, windowed CC): {med_rr('spearman_q_vs_cc'):.3f}",
                f"- ρ(Q, BlocRes sharpness): {med_rr('spearman_q_vs_locres'):.3f} "
                f"(raw ρ vs Å: {float(np.median([r.spearman_q_vs_locres for r in summary.rank_recovery_rows if np.isfinite(r.spearman_q_vs_locres)])):.3f})",
                f"- ρ(Q, local variance): {med_rr('spearman_q_vs_variance'):.3f}",
                f"- ρ(Q, constraint V): {med_rr('spearman_q_vs_v'):.3f}",
                "",
                "_BlocRes bars use sign-aligned coupling (negate raw ρ vs Å); "
                "median-split flags use loc > in-map median as low-confidence._",
                "",
            ]
        )
        for k, v in sorted(summary.resolution_bins.items()):
            label = k
            for b in COHORT_RESOLUTION_BINS:
                if b.key in k:
                    label = f"{b.label} (median ρ(Q, reliability))"
                    break
            lines.append(f"- {label}: **{v:.3f}**")
        lines.append("")

    lines.extend(["## Tier 1 — Mis-ranking (bottom Q tercile)", ""])
    if summary.misranking_rows:
        def med_mr(attr: str) -> float:
            vals = [getattr(r, attr) for r in summary.misranking_rows]
            vals = [v for v in vals if np.isfinite(v)]
            return float(np.median(vals)) if vals else float("nan")

        lines.extend(
            [
                f"- Fraction with **sharp BlocRes** (≤ median Å) but bottom-Q tercile: "
                f"{med_mr('frac_sharp_locres_low_q_tercile'):.3f}",
                f"- Fraction in **omit zone** among bottom-Q tercile: "
                f"{med_mr('frac_omit_zone_low_q_tercile'):.3f}",
                f"- Fraction with **CC ≥ 0.7** among bottom-Q tercile: "
                f"{med_mr('frac_cc_above_0_7_low_q_tercile'):.3f}",
                "",
            ]
        )

    lines.append("## Calibration")
    lines.append("")
    lines.append("Reliability score deciles vs mean Q — see `placement_q_calibration_bins.csv`.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path



TRAIN_DERIVED_PREDICTORS: frozenset[PredictorId] = frozenset(
    {"locres_worse_than_median", "variance_above_median"}
)

MAIN_ROC_PREDICTORS: tuple[PredictorId, ...] = (
    "reliability_below_0_33",
    "cc_below_0_5",
    "omit_zone",
    "locres_worse_than_median",
)


@dataclass(frozen=True)
class LomoPredictorFoldRow:
    held_out_emdb_id: str
    predictor: PredictorId
    global_resolution_a: float
    n_residues: int
    n_low_q: int
    balanced_accuracy: float
    auc: float
    spearman_q_vs_score: float
    frac_low_q_flagged: float
    train_locres_median: float = float("nan")
    train_variance_median: float = float("nan")


@dataclass(frozen=True)
class LomoPlacementSummary:
    q_threshold: float
    fold_rows: tuple[LomoPredictorFoldRow, ...]
    predictor_medians: dict[str, dict[str, float]]


@dataclass(frozen=True)
class LocresMethodLomoFoldRow:
    held_out_emdb_id: str
    predictor: LocresMethodLomoPredictor
    global_resolution_a: float
    n_residues: int
    n_low_q: int
    balanced_accuracy: float
    auc: float
    spearman_q_vs_score: float
    frac_low_q_flagged: float
    flag_threshold_a: float = float("nan")


@dataclass(frozen=True)
class LocresMethodLomoSummary:
    q_threshold: float
    exclude_emdb_ids: frozenset[str]
    fold_rows: tuple[LocresMethodLomoFoldRow, ...]
    predictor_medians: dict[str, dict[str, float]]


@dataclass(frozen=True)
class RocCurve:
    predictor: PredictorId
    fpr: tuple[float, ...]
    tpr: tuple[float, ...]
    auc: float


@dataclass(frozen=True)
class CohortRocSummary:
    """Per-map AUC summary plus ROC from the map nearest the cohort median AUC."""

    predictor: PredictorId
    median_auc: float
    representative_emdb_id: str
    representative_auc: float
    n_maps: int
    fpr: tuple[float, ...]
    tpr: tuple[float, ...]
    per_map_aucs: tuple[tuple[str, float], ...]


def finite_qv_emdb_ids() -> frozenset[str]:
    """EMDB IDs with finite in-mask ρ(Q, V) from ``qscore_correlations.csv``."""
    path = OUTPUTS_ROOT / "cohort_summary" / "qscore_correlations.csv"
    if not path.is_file():
        return frozenset()
    ids: set[str] = set()
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            eid = str(row.get("emdb_id", "")).strip()
            raw = str(row.get("spearman_q_vs_V", "")).strip()
            if not eid or eid in QSCORE_PANEL_EXCLUDE or raw in ("", "nan"):
                continue
            try:
                rho = float(raw)
            except ValueError:
                continue
            if np.isfinite(rho):
                ids.add(eid)
    return frozenset(ids)


def _roc_points_from_scores(y_true: np.ndarray, scores: np.ndarray) -> tuple[tuple[float, ...], tuple[float, ...], float]:
    """Return (fpr, tpr, auc) for binary labels and higher-is-risk scores."""
    y = y_true.astype(bool)
    s = np.asarray(scores, dtype=np.float64)
    m = np.isfinite(s)
    y = y[m]
    s = s[m]
    auc = rank_auc(y, s)
    order = np.argsort(-s)
    y_sorted = y[order]
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return (), (), float(auc)
    tps = np.cumsum(y_sorted).astype(np.float64)
    fps = np.cumsum(~y_sorted).astype(np.float64)
    tpr_pts = np.concatenate([[0.0], tps / n_pos])
    fpr_pts = np.concatenate([[0.0], fps / n_neg])
    return (
        tuple(float(x) for x in fpr_pts),
        tuple(float(x) for x in tpr_pts),
        float(auc),
    )


def single_map_roc_curve(
    df: pd.DataFrame,
    predictor: PredictorId,
    *,
    q_threshold: float,
) -> RocCurve:
    """ROC/AUC for one map's in-mask residues."""
    q = pd.to_numeric(df["q_score"], errors="coerce").to_numpy()
    m = np.isfinite(q)
    if int(m.sum()) < 5:
        return RocCurve(predictor=predictor, fpr=(), tpr=(), auc=float("nan"))
    y = (q < q_threshold)[m]
    s = _predictor_scores(df)[predictor][m]
    fpr, tpr, auc = _roc_points_from_scores(y, s)
    return RocCurve(predictor=predictor, fpr=fpr, tpr=tpr, auc=auc)


def cohort_representative_roc(
    per_map_frames: Sequence[tuple[str, pd.DataFrame]],
    predictor: PredictorId,
    *,
    q_threshold: float,
    eligible_emdb_ids: frozenset[str] | None = None,
) -> CohortRocSummary:
    """
    Per-map AUC on eligible maps; plot curve from the map whose AUC is nearest
    the cohort median (illustrative single-map ROC).
    """
    per_map_aucs: list[tuple[str, float]] = []
    curves: dict[str, RocCurve] = {}
    for emdb_id, df in per_map_frames:
        if eligible_emdb_ids is not None and str(emdb_id) not in eligible_emdb_ids:
            continue
        curve = single_map_roc_curve(df, predictor, q_threshold=q_threshold)
        if not np.isfinite(curve.auc):
            continue
        per_map_aucs.append((str(emdb_id), float(curve.auc)))
        curves[str(emdb_id)] = curve

    if not per_map_aucs:
        return CohortRocSummary(
            predictor=predictor,
            median_auc=float("nan"),
            representative_emdb_id="",
            representative_auc=float("nan"),
            n_maps=0,
            fpr=(),
            tpr=(),
            per_map_aucs=(),
        )

    auc_vals = np.array([a for _, a in per_map_aucs], dtype=np.float64)
    median_auc = float(np.median(auc_vals))
    rep_id, rep_auc = min(per_map_aucs, key=lambda item: abs(item[1] - median_auc))
    rep_curve = curves[rep_id]
    per_map_aucs.sort(key=lambda item: item[1])

    return CohortRocSummary(
        predictor=predictor,
        median_auc=median_auc,
        representative_emdb_id=rep_id,
        representative_auc=float(rep_auc),
        n_maps=len(per_map_aucs),
        fpr=rep_curve.fpr,
        tpr=rep_curve.tpr,
        per_map_aucs=tuple(per_map_aucs),
    )


def _train_medians(train_dfs: Sequence[pd.DataFrame]) -> tuple[float, float]:
    loc_parts: list[np.ndarray] = []
    var_parts: list[np.ndarray] = []
    for df in train_dfs:
        loc = pd.to_numeric(df["local_resolution"], errors="coerce").to_numpy()
        var = pd.to_numeric(df["local_variance"], errors="coerce").to_numpy()
        loc_parts.append(loc[np.isfinite(loc)])
        var_parts.append(var[np.isfinite(var)])
    loc_all = np.concatenate(loc_parts) if loc_parts else np.array([], dtype=np.float64)
    var_all = np.concatenate(var_parts) if var_parts else np.array([], dtype=np.float64)
    loc_med = float(np.median(loc_all)) if loc_all.size else float("nan")
    var_med = float(np.median(var_all)) if var_all.size else float("nan")
    return loc_med, var_med


def _lomo_predictor_flags(
    df: pd.DataFrame,
    predictor: PredictorId,
    *,
    train_locres_median: float = float("nan"),
    train_variance_median: float = float("nan"),
) -> np.ndarray:
    if predictor not in TRAIN_DERIVED_PREDICTORS:
        return _predictor_flags(df)[predictor]
    loc = pd.to_numeric(df["local_resolution"], errors="coerce").to_numpy()
    var = pd.to_numeric(df["local_variance"], errors="coerce").to_numpy()
    if predictor == "locres_worse_than_median":
        if not np.isfinite(train_locres_median):
            return np.zeros(len(df), dtype=bool)
        return loc > train_locres_median
    if not np.isfinite(train_variance_median):
        return np.zeros(len(df), dtype=bool)
    return var > train_variance_median


def _predictor_rank_proxy(df: pd.DataFrame, predictor: PredictorId) -> np.ndarray:
    """Continuous proxy where higher values should track higher Q (for Spearman ρ)."""
    rel = pd.to_numeric(df["reliability_score"], errors="coerce").to_numpy()
    cc = pd.to_numeric(df[WINDOWED_HALFMAP_CORRELATION_KEY], errors="coerce").to_numpy()
    loc = pd.to_numeric(df["local_resolution"], errors="coerce").to_numpy()
    var = pd.to_numeric(df["local_variance"], errors="coerce").to_numpy()
    zone = pd.to_numeric(df["build_zone"], errors="coerce").to_numpy()
    return {
        "omit_zone": zone,
        "reliability_below_0_33": rel,
        "cc_below_0_5": cc,
        "locres_worse_than_median": loc,
        "variance_above_median": var,
    }[predictor]


def evaluate_map_predictor(
    df: pd.DataFrame,
    predictor: PredictorId,
    *,
    q_threshold: float,
    train_locres_median: float = float("nan"),
    train_variance_median: float = float("nan"),
) -> tuple[float, float, float, float, int, int]:
    """Return BA, AUC, Spearman ρ, frac low-Q flagged, n_residues, n_low_q."""
    q = pd.to_numeric(df["q_score"], errors="coerce").to_numpy()
    m = np.isfinite(q)
    if int(m.sum()) < 10:
        nan = float("nan")
        return nan, nan, nan, nan, int(m.sum()), 0
    low = q < q_threshold
    flags = _lomo_predictor_flags(
        df,
        predictor,
        train_locres_median=train_locres_median,
        train_variance_median=train_variance_median,
    )
    scores = _predictor_scores(df)[predictor]
    n_low = int(low[m].sum())
    frac_flag = float(flags[m][low[m]].mean()) if n_low else float("nan")
    return (
        balanced_accuracy(low[m], flags[m]),
        rank_auc(low[m], scores[m]),
        _finite_spearman(q[m], _predictor_rank_proxy(df, predictor)[m]),
        frac_flag,
        int(m.sum()),
        n_low,
    )


def run_lomo_placement_validation(
    per_map_frames: Sequence[tuple[str, pd.DataFrame, float]],
    *,
    q_threshold: float = 0.5,
) -> LomoPlacementSummary:
    """Leave-one-map-out evaluation; train-derived medians for BlocRes/variance flags."""
    if len(per_map_frames) < 3:
        raise ValueError("need at least three maps for leave-one-map-out validation")

    fold_rows: list[LomoPredictorFoldRow] = []
    for held_out_id, test_df, gres in per_map_frames:
        train = [(eid, df) for eid, df, _ in per_map_frames if eid != held_out_id]
        train_dfs = [df for _, df in train]
        loc_med, var_med = _train_medians(train_dfs)
        for pid in PREDICTOR_LABELS:
            ba, auc, rho, frac_flag, n_res, n_low = evaluate_map_predictor(
                test_df,
                pid,
                q_threshold=q_threshold,
                train_locres_median=loc_med,
                train_variance_median=var_med,
            )
            fold_rows.append(
                LomoPredictorFoldRow(
                    held_out_emdb_id=str(held_out_id),
                    predictor=pid,
                    global_resolution_a=gres,
                    n_residues=n_res,
                    n_low_q=n_low,
                    balanced_accuracy=ba,
                    auc=auc,
                    spearman_q_vs_score=rho,
                    frac_low_q_flagged=frac_flag,
                    train_locres_median=loc_med,
                    train_variance_median=var_med,
                )
            )

    predictor_medians: dict[str, dict[str, float]] = {}
    for pid in PREDICTOR_LABELS:
        sub = [r for r in fold_rows if r.predictor == pid]
        for attr in ("balanced_accuracy", "auc", "spearman_q_vs_score", "frac_low_q_flagged"):
            vals = [getattr(r, attr) for r in sub if np.isfinite(getattr(r, attr))]
            predictor_medians.setdefault(pid, {})[f"median_{attr}"] = (
                float(np.median(vals)) if vals else float("nan")
            )

    return LomoPlacementSummary(
        q_threshold=q_threshold,
        fold_rows=tuple(fold_rows),
        predictor_medians=predictor_medians,
    )


def pooled_roc_curve(
    per_map_frames: Sequence[tuple[str, pd.DataFrame]],
    predictor: PredictorId,
    *,
    q_threshold: float,
) -> RocCurve:
    """Pooled cohort ROC for low-Q classification (y = Q < threshold)."""
    low_list: list[np.ndarray] = []
    score_list: list[np.ndarray] = []
    for _emd, df in per_map_frames:
        q = pd.to_numeric(df["q_score"], errors="coerce").to_numpy()
        m = np.isfinite(q)
        if m.sum() < 5:
            continue
        low_list.append((q < q_threshold)[m])
        score_list.append(_predictor_scores(df)[predictor][m])
    if not low_list:
        return RocCurve(predictor=predictor, fpr=(), tpr=(), auc=float("nan"))

    y = np.concatenate(low_list).astype(bool)
    s = np.concatenate(score_list)
    fpr, tpr, auc = _roc_points_from_scores(y, s)
    return RocCurve(predictor=predictor, fpr=fpr, tpr=tpr, auc=auc)


def write_roc_per_map_csv(
    summaries: Sequence[CohortRocSummary],
    out_dir: Path,
) -> Path:
    """Write per-map AUC rows for each main ROC predictor."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "placement_roc_per_map.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "predictor",
                "label",
                "emdb_id",
                "auc",
                "cohort_median_auc",
                "representative_emdb_id",
            ],
        )
        w.writeheader()
        for summary in summaries:
            for emdb_id, auc in summary.per_map_aucs:
                w.writerow(
                    {
                        "predictor": summary.predictor,
                        "label": PREDICTOR_LABELS[summary.predictor],
                        "emdb_id": emdb_id,
                        "auc": f"{auc:.4f}",
                        "cohort_median_auc": f"{summary.median_auc:.4f}"
                        if np.isfinite(summary.median_auc)
                        else "",
                        "representative_emdb_id": summary.representative_emdb_id,
                    }
                )
    return path


def write_roc_figma_json(
    summaries: Sequence[CohortRocSummary],
    path: Path,
    *,
    q_threshold: float,
    n_maps: int,
) -> Path:
    """Compact ROC curve bundle for the Figma placement plugin."""
    colors = ("#E8303A", "#4B6FD4", "#3BBF6A", "#BA3EC3")
    payload = {
        "q_threshold": q_threshold,
        "n_maps": n_maps,
        "curves": [
            {
                "predictor": summary.predictor,
                "label": PREDICTOR_LABELS[summary.predictor],
                "color": colors[i % len(colors)],
                "median_auc": round(summary.median_auc, 4)
                if np.isfinite(summary.median_auc)
                else None,
                "representative_emdb_id": summary.representative_emdb_id,
                "representative_auc": round(summary.representative_auc, 4)
                if np.isfinite(summary.representative_auc)
                else None,
                "fpr": [round(x, 5) for x in summary.fpr],
                "tpr": [round(y, 5) for y in summary.tpr],
            }
            for i, summary in enumerate(summaries)
            if summary.fpr
        ],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def load_per_map_frames_for_lomo(
    *,
    manifest: Path = COHORT_MANIFEST,
    sphere_radius_a: float = 2.0,
    exclude: frozenset[str] | None = None,
) -> list[tuple[str, pd.DataFrame, float]]:
    """Load (emdb_id, dataframe, global_resolution) for maps with Q-scores."""
    frames: list[tuple[str, pd.DataFrame, float]] = []
    for emdb_id in iter_qscore_maps(manifest=manifest, exclude=exclude):
        try:
            row = load_cohort_manifest_row(manifest, emdb_id)
        except KeyError:
            continue
        df = load_map_with_qscore(
            emdb_id, manifest=manifest, sphere_radius_a=sphere_radius_a
        )
        if df is None or df.empty:
            continue
        frames.append((emdb_id, df, _global_resolution(row)))
    return frames


def write_lomo_placement_csvs(
    summary: LomoPlacementSummary,
    out_dir: Path,
) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    p_folds = out_dir / "placement_lomo_folds.csv"
    with p_folds.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "held_out_emdb_id",
                "predictor",
                "label",
                "global_resolution_a",
                "n_residues",
                "n_low_q",
                "balanced_accuracy",
                "auc",
                "spearman_q_vs_score",
                "frac_low_q_flagged",
                "train_locres_median",
                "train_variance_median",
            ],
        )
        w.writeheader()
        for r in summary.fold_rows:
            w.writerow(
                {
                    "held_out_emdb_id": r.held_out_emdb_id,
                    "predictor": r.predictor,
                    "label": PREDICTOR_LABELS[r.predictor],
                    "global_resolution_a": f"{r.global_resolution_a:.2f}"
                    if np.isfinite(r.global_resolution_a)
                    else "",
                    "n_residues": r.n_residues,
                    "n_low_q": r.n_low_q,
                    "balanced_accuracy": f"{r.balanced_accuracy:.4f}",
                    "auc": f"{r.auc:.4f}",
                    "spearman_q_vs_score": f"{r.spearman_q_vs_score:.4f}",
                    "frac_low_q_flagged": f"{r.frac_low_q_flagged:.4f}",
                    "train_locres_median": f"{r.train_locres_median:.3f}"
                    if np.isfinite(r.train_locres_median)
                    else "",
                    "train_variance_median": f"{r.train_variance_median:.3f}"
                    if np.isfinite(r.train_variance_median)
                    else "",
                }
            )
    paths["folds"] = p_folds

    p_med = out_dir / "placement_lomo_medians.csv"
    with p_med.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "predictor",
                "label",
                "median_balanced_accuracy",
                "median_auc",
                "median_spearman_q_vs_score",
                "median_frac_low_q_flagged",
            ],
        )
        w.writeheader()
        for pid in PREDICTOR_LABELS:
            meds = summary.predictor_medians.get(pid, {})
            w.writerow(
                {
                    "predictor": pid,
                    "label": PREDICTOR_LABELS[pid],
                    "median_balanced_accuracy": f"{meds.get('median_balanced_accuracy', float('nan')):.4f}",
                    "median_auc": f"{meds.get('median_auc', float('nan')):.4f}",
                    "median_spearman_q_vs_score": f"{meds.get('median_spearman_q_vs_score', float('nan')):.4f}",
                    "median_frac_low_q_flagged": f"{meds.get('median_frac_low_q_flagged', float('nan')):.4f}",
                }
            )
    paths["medians"] = p_med
    return paths


def write_lomo_placement_markdown(summary: LomoPlacementSummary, path: Path) -> Path:
    path = Path(path)
    n_maps = len({r.held_out_emdb_id for r in summary.fold_rows})
    lines = [
        "# Semi-prospective placement validation (leave-one-map-out)",
        "",
        f"Low-Q definition: Q-score < **{summary.q_threshold:.2f}**.",
        f"Maps: **{n_maps}** held-out folds; BlocRes/variance thresholds fit on the other *N*−1 maps.",
        "",
        "## Median held-out metrics",
        "",
        "| Predictor | Median BA | Median AUC | Median ρ(Q, score) |",
        "|-----------|-----------|------------|---------------------|",
    ]
    for pid in PREDICTOR_LABELS:
        meds = summary.predictor_medians.get(pid, {})
        lines.append(
            f"| {PREDICTOR_LABELS[pid]} | "
            f"{meds.get('median_balanced_accuracy', float('nan')):.3f} | "
            f"{meds.get('median_auc', float('nan')):.3f} | "
            f"{meds.get('median_spearman_q_vs_score', float('nan')):.3f} |"
        )
    lines.extend(
        [
            "",
            "Fixed-threshold readouts (reliability < 0.33, CC < 0.5, omit zone) do not use training data;",
            "LOMO confirms per-map generalization rather than cohort pooling.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def enrich_with_resmap_locres(df: pd.DataFrame, emdb_id: str) -> pd.DataFrame:
    """Attach ``local_resolution_resmap`` from the ResMap metric-export table."""
    out = df.copy()
    path = emd_output_dir(emdb_id) / "metric_comparison_resmap" / "residue_metrics.csv"
    if not path.is_file():
        out["local_resolution_resmap"] = np.nan
        return out
    resmap = pd.read_csv(path)
    if "chain" not in resmap.columns or "seq_num" not in resmap.columns:
        out["local_resolution_resmap"] = np.nan
        return out
    loc = pd.to_numeric(
        resmap.get("local_resolution", resmap.get("local_resolution_mean", np.nan)),
        errors="coerce",
    )
    merge = resmap[["chain", "seq_num"]].assign(local_resolution_resmap=loc)
    for frame in (out, merge):
        frame["chain"] = frame["chain"].astype(str)
        frame["seq_num"] = pd.to_numeric(frame["seq_num"], errors="coerce")
    return out.merge(merge, on=["chain", "seq_num"], how="left")


def enrich_with_monores_locres(df: pd.DataFrame, emdb_id: str) -> pd.DataFrame:
    """Attach ``local_resolution_monores`` from the MonoRes metric-export table."""
    out = df.copy()
    path = emd_output_dir(emdb_id) / "metric_comparison_monores" / "residue_metrics.csv"
    if not path.is_file():
        out["local_resolution_monores"] = np.nan
        return out
    mono = pd.read_csv(path)
    if "chain" not in mono.columns or "seq_num" not in mono.columns:
        out["local_resolution_monores"] = np.nan
        return out
    loc = pd.to_numeric(
        mono.get("local_resolution", mono.get("local_resolution_mean", np.nan)),
        errors="coerce",
    )
    merge = mono[["chain", "seq_num"]].assign(local_resolution_monores=loc)
    for frame in (out, merge):
        frame["chain"] = frame["chain"].astype(str)
        frame["seq_num"] = pd.to_numeric(frame["seq_num"], errors="coerce")
    return out.merge(merge, on=["chain", "seq_num"], how="left")


def load_map_with_qscore_and_resmap(
    emdb_id: str,
    *,
    manifest: Path = COHORT_MANIFEST,
    sphere_radius_a: float = 2.0,
) -> pd.DataFrame | None:
    """In-mask BlocRes metrics + Q-scores + ResMap + MonoRes locres columns."""
    df = load_map_with_qscore(emdb_id, manifest=manifest, sphere_radius_a=sphere_radius_a)
    if df is None:
        return None
    df = enrich_with_resmap_locres(df, emdb_id)
    return enrich_with_monores_locres(df, emdb_id)


def load_per_map_frames_for_locres_lomo(
    *,
    manifest: Path = COHORT_MANIFEST,
    sphere_radius_a: float = 2.0,
    exclude: frozenset[str] | None = None,
) -> list[tuple[str, pd.DataFrame, float]]:
    """Load Q-score frames with BlocRes, ResMap, and MonoRes locres columns."""
    exclude = exclude or QSCORE_PANEL_EXCLUDE
    frames: list[tuple[str, pd.DataFrame, float]] = []
    for emdb_id in iter_qscore_maps(manifest=manifest, exclude=exclude):
        try:
            row = load_cohort_manifest_row(manifest, emdb_id)
        except KeyError:
            continue
        df = load_map_with_qscore_and_resmap(
            emdb_id, manifest=manifest, sphere_radius_a=sphere_radius_a
        )
        if df is None or df.empty:
            continue
        frames.append((emdb_id, df, _global_resolution(row)))
    return frames


def _inmap_locres_median(loc: np.ndarray) -> float:
    finite = loc[np.isfinite(loc)]
    return float(np.median(finite)) if finite.size else float("nan")


def _numeric_series(df: pd.DataFrame, column: str) -> np.ndarray:
    if column not in df.columns:
        return np.full(len(df), np.nan, dtype=np.float64)
    return pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=np.float64)


def _locres_method_fixed_flags(
    df: pd.DataFrame,
    predictor: LocresMethodLomoPredictor,
    *,
    global_resolution_a: float,
) -> tuple[np.ndarray, float]:
    """Fixed operational flag rules; return (flags, threshold used for locres rules)."""
    if predictor == "omit_zone":
        zone = pd.to_numeric(df["build_zone"], errors="coerce").to_numpy(dtype=np.int32)
        return zone == 0, float("nan")

    if predictor in ("blocres_locres_inmap_median", "blocres_locres_vs_global"):
        loc = _numeric_series(df, "local_resolution")
    elif predictor in ("monores_locres_inmap_median", "monores_locres_vs_global"):
        loc = _numeric_series(df, "local_resolution_monores")
    else:
        loc = _numeric_series(df, "local_resolution_resmap")

    if predictor.endswith("_inmap_median"):
        threshold = _inmap_locres_median(loc)
    else:
        threshold = float(global_resolution_a)

    if not np.isfinite(threshold):
        return np.zeros(len(df), dtype=bool), threshold
    return loc > threshold, threshold


def _locres_method_fixed_scores(
    df: pd.DataFrame,
    predictor: LocresMethodLomoPredictor,
) -> np.ndarray:
    """Continuous low-Q risk scores (higher ⇒ more likely Q < threshold)."""
    if predictor == "omit_zone":
        return _predictor_scores(df)["omit_zone"]
    if predictor.startswith("blocres_"):
        return _numeric_series(df, "local_resolution")
    if predictor.startswith("monores_"):
        return _numeric_series(df, "local_resolution_monores")
    return _numeric_series(df, "local_resolution_resmap")


def _locres_method_rank_proxy(
    df: pd.DataFrame,
    predictor: LocresMethodLomoPredictor,
) -> np.ndarray:
    """Continuous proxy where higher values should track higher Q (for Spearman ρ)."""
    if predictor == "omit_zone":
        return _predictor_rank_proxy(df, "omit_zone")
    if predictor.startswith("blocres_"):
        loc = _numeric_series(df, "local_resolution")
        return -loc
    if predictor.startswith("monores_"):
        loc = _numeric_series(df, "local_resolution_monores")
        return -loc
    loc = _numeric_series(df, "local_resolution_resmap")
    return -loc


def evaluate_locres_method_lomo_fold(
    df: pd.DataFrame,
    predictor: LocresMethodLomoPredictor,
    *,
    q_threshold: float,
    global_resolution_a: float,
    train_blocres_median: float = float("nan"),
    train_resmap_median: float = float("nan"),
    train_v_median: float = float("nan"),
) -> tuple[float, float, float, float, int, int, float]:
    """Return BA, AUC, Spearman ρ, frac low-Q flagged, n_residues, n_low_q, flag threshold."""
    del train_blocres_median, train_resmap_median, train_v_median
    q = pd.to_numeric(df["q_score"], errors="coerce").to_numpy(dtype=np.float64)
    m = np.isfinite(q)
    if int(m.sum()) < 10:
        nan = float("nan")
        return nan, nan, nan, nan, int(m.sum()), 0, nan

    low = q < q_threshold
    flags, threshold = _locres_method_fixed_flags(
        df, predictor, global_resolution_a=global_resolution_a
    )
    scores = _locres_method_fixed_scores(df, predictor)
    proxy = _locres_method_rank_proxy(df, predictor)
    n_low = int(low[m].sum())
    frac_flag = float(flags[m][low[m]].mean()) if n_low else float("nan")
    return (
        balanced_accuracy(low[m], flags[m]),
        rank_auc(low[m], scores[m]),
        _finite_spearman(q[m], proxy[m]),
        frac_flag,
        int(m.sum()),
        n_low,
        threshold,
    )


def run_locres_method_lomo_validation(
    per_map_frames: Sequence[tuple[str, pd.DataFrame, float]],
    *,
    q_threshold: float = 0.5,
    exclude_emdb_ids: frozenset[str] | None = None,
) -> LocresMethodLomoSummary:
    """Per-map low-Q utility with fixed operational rules (leave-one-map-out reporting).

    Locres flags: worse than **in-map median Å** or worse than **deposited global
    resolution** (two hypotheses). Omit-zone flag: ``build_zone == 0`` (~bottom
    reliability tercile). AUC uses continuous per-residue scores on each map.
    """
    exclude_emdb_ids = exclude_emdb_ids or frozenset()
    frames = [
        (eid, df, gres)
        for eid, df, gres in per_map_frames
        if str(eid) not in exclude_emdb_ids
    ]
    if len(frames) < 3:
        raise ValueError("need at least three maps for leave-one-map-out validation")

    fold_rows: list[LocresMethodLomoFoldRow] = []
    for held_out_id, test_df, gres in frames:
        for pid in LOCRES_METHOD_LOMO_PREDICTORS:
            ba, auc, rho, frac_flag, n_res, n_low, threshold = evaluate_locres_method_lomo_fold(
                test_df,
                pid,
                q_threshold=q_threshold,
                global_resolution_a=gres,
            )
            fold_rows.append(
                LocresMethodLomoFoldRow(
                    held_out_emdb_id=str(held_out_id),
                    predictor=pid,
                    global_resolution_a=gres,
                    n_residues=n_res,
                    n_low_q=n_low,
                    balanced_accuracy=ba,
                    auc=auc,
                    spearman_q_vs_score=rho,
                    frac_low_q_flagged=frac_flag,
                    flag_threshold_a=threshold,
                )
            )

    predictor_medians: dict[str, dict[str, float]] = {}
    for pid in LOCRES_METHOD_LOMO_PREDICTORS:
        sub = [r for r in fold_rows if r.predictor == pid]
        for attr in ("balanced_accuracy", "auc", "spearman_q_vs_score", "frac_low_q_flagged"):
            vals = [getattr(r, attr) for r in sub if np.isfinite(getattr(r, attr))]
            predictor_medians.setdefault(pid, {})[f"median_{attr}"] = (
                float(np.median(vals)) if vals else float("nan")
            )

    return LocresMethodLomoSummary(
        q_threshold=q_threshold,
        exclude_emdb_ids=exclude_emdb_ids,
        fold_rows=tuple(fold_rows),
        predictor_medians=predictor_medians,
    )


def write_locres_method_lomo_csvs(
    summary: LocresMethodLomoSummary,
    out_dir: Path,
    *,
    file_stem: str = "placement_locres_lomo",
) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    p_folds = out_dir / f"{file_stem}_folds.csv"
    with p_folds.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "held_out_emdb_id",
                "predictor",
                "label",
                "global_resolution_a",
                "n_residues",
                "n_low_q",
                "balanced_accuracy",
                "auc",
                "spearman_q_vs_score",
                "frac_low_q_flagged",
                "flag_threshold_a",
            ],
        )
        w.writeheader()
        for r in summary.fold_rows:
            w.writerow(
                {
                    "held_out_emdb_id": r.held_out_emdb_id,
                    "predictor": r.predictor,
                    "label": LOCRES_METHOD_LOMO_LABELS[r.predictor],
                    "global_resolution_a": f"{r.global_resolution_a:.2f}"
                    if np.isfinite(r.global_resolution_a)
                    else "",
                    "n_residues": r.n_residues,
                    "n_low_q": r.n_low_q,
                    "balanced_accuracy": f"{r.balanced_accuracy:.4f}",
                    "auc": f"{r.auc:.4f}",
                    "spearman_q_vs_score": f"{r.spearman_q_vs_score:.4f}",
                    "frac_low_q_flagged": f"{r.frac_low_q_flagged:.4f}",
                    "flag_threshold_a": f"{r.flag_threshold_a:.3f}"
                    if np.isfinite(r.flag_threshold_a)
                    else "",
                }
            )
    paths["folds"] = p_folds

    p_med = out_dir / f"{file_stem}_medians.csv"
    with p_med.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "predictor",
                "label",
                "median_balanced_accuracy",
                "median_auc",
                "median_spearman_q_vs_score",
                "median_frac_low_q_flagged",
            ],
        )
        w.writeheader()
        for pid in LOCRES_METHOD_LOMO_PREDICTORS:
            meds = summary.predictor_medians.get(pid, {})
            w.writerow(
                {
                    "predictor": pid,
                    "label": LOCRES_METHOD_LOMO_LABELS[pid],
                    "median_balanced_accuracy": f"{meds.get('median_balanced_accuracy', float('nan')):.4f}",
                    "median_auc": f"{meds.get('median_auc', float('nan')):.4f}",
                    "median_spearman_q_vs_score": f"{meds.get('median_spearman_q_vs_score', float('nan')):.4f}",
                    "median_frac_low_q_flagged": f"{meds.get('median_frac_low_q_flagged', float('nan')):.4f}",
                }
            )
    paths["medians"] = p_med
    return paths


def write_locres_method_lomo_markdown(summary: LocresMethodLomoSummary, path: Path) -> Path:
    path = Path(path)
    n_maps = len({r.held_out_emdb_id for r in summary.fold_rows})
    excluded = ", ".join(sorted(summary.exclude_emdb_ids)) or "(none)"
    lines = [
        "# Parallel per-map placement utility: BlocRes vs ResMap vs omit zone",
        "",
        f"Low-Q definition: Q-score < **{summary.q_threshold:.2f}**.",
        f"Maps: **{n_maps}** per-map evaluations.",
        f"Excluded EMD IDs: {excluded}.",
        "",
        "**Fixed flag rules** (not train-map medians):",
        "- BlocRes / ResMap: worse than in-map median Å **or** worse than deposited global resolution",
        "- Omit zone: ``build_zone == 0`` (bottom reliability tercile, ~33% of in-mask Cα)",
        "",
        "AUC uses continuous per-residue scores (locres Å; omit-zone risk score).",
        "",
        "## Median per-map metrics",
        "",
        "| Predictor | Median BA | Median AUC | Median ρ(Q, score) |",
        "|-----------|-----------|------------|---------------------|",
    ]
    for pid in LOCRES_METHOD_LOMO_PREDICTORS:
        meds = summary.predictor_medians.get(pid, {})
        lines.append(
            f"| {LOCRES_METHOD_LOMO_LABELS[pid]} | "
            f"{meds.get('median_balanced_accuracy', float('nan')):.3f} | "
            f"{meds.get('median_auc', float('nan')):.3f} | "
            f"{meds.get('median_spearman_q_vs_score', float('nan')):.3f} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
