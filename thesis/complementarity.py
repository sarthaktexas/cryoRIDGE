"""Gradient-family + Hessian geometry Q-score screening with rank-normalized LOMO."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from cryoem_mrc.halfmap_metrics import WINDOWED_HALFMAP_CORRELATION_KEY
from thesis.incremental_prediction import (
    MapPredictionFrame,
    ols_fit,
    ols_predict,
    ols_r2,
    percentile_rank,
    run_lomo_incremental_prediction,
)

BASELINE_RANK_COLUMNS: tuple[str, ...] = (
    "local_variance",
    WINDOWED_HALFMAP_CORRELATION_KEY,
    "local_resolution",
)

GRADIENT_FAMILY_COLUMNS: tuple[str, ...] = (
    "smoothness",
    "T_vonweizsacker",
    "V_curvature",
)

GEOMETRY_COLUMN = "neg_lam_min"

FOCUS_PREDICTORS: tuple[str, ...] = (*GRADIENT_FAMILY_COLUMNS, GEOMETRY_COLUMN)

LomoModelId = Literal[
    "baseline",
    "plus_neg_lam",
    "plus_grad_family",
    "plus_grad_neg_lam",
]

LOMO_MODEL_FEATURES: dict[LomoModelId, tuple[str, ...]] = {
    "baseline": BASELINE_RANK_COLUMNS,
    "plus_neg_lam": (*BASELINE_RANK_COLUMNS, GEOMETRY_COLUMN),
    "plus_grad_family": (*BASELINE_RANK_COLUMNS, *GRADIENT_FAMILY_COLUMNS),
    "plus_grad_neg_lam": (*BASELINE_RANK_COLUMNS, *GRADIENT_FAMILY_COLUMNS, GEOMETRY_COLUMN),
}


def masked_percentile_rank(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Percentile ranks in (0, 1] among ``mask & finite``; NaN elsewhere."""
    v = np.asarray(values, dtype=np.float64)
    m = np.asarray(mask, dtype=bool) & np.isfinite(v)
    out = np.full(v.shape, np.nan, dtype=np.float64)
    if int(m.sum()) < 2:
        return out
    out[m] = percentile_rank(v[m])
    return out


def spearman_vs_q(
    df: pd.DataFrame,
    predictors: Sequence[str],
    *,
    q_col: str = "q_score",
    mask_column: str = "in_contour_mask",
    min_pairs: int = 10,
) -> dict[str, float]:
    use = df[df[mask_column].astype(bool)].copy()
    q = pd.to_numeric(use[q_col], errors="coerce")
    out: dict[str, float] = {}
    for name in predictors:
        if name not in use.columns:
            out[name] = float("nan")
            continue
        x = pd.to_numeric(use[name], errors="coerce")
        m = q.notna() & x.notna()
        if int(m.sum()) < min_pairs:
            out[name] = float("nan")
            continue
        rho, _ = stats.spearmanr(q[m], x[m])
        out[name] = float(rho)
    return out


def pairwise_spearman_median(
    tables: Sequence[pd.DataFrame],
    columns: Sequence[str],
    *,
    mask_column: str = "in_contour_mask",
    min_pairs: int = 10,
) -> pd.DataFrame:
    """Median across maps of in-mask Spearman ρ between column pairs."""
    cols = [c for c in columns if any(c in t.columns for t in tables)]
    rhos: dict[tuple[str, str], list[float]] = {}
    for df in tables:
        use = df[df[mask_column].astype(bool)]
        numeric = use[cols].apply(pd.to_numeric, errors="coerce")
        for i, ci in enumerate(cols):
            for cj in cols[i + 1 :]:
                m = numeric[ci].notna() & numeric[cj].notna()
                if int(m.sum()) < min_pairs:
                    continue
                rho, _ = stats.spearmanr(numeric.loc[m, ci], numeric.loc[m, cj])
                rhos.setdefault((ci, cj), []).append(float(rho))

    rows: list[dict[str, object]] = []
    for (ci, cj), vals in sorted(rhos.items()):
        rows.append(
            {
                "feature_i": ci,
                "feature_j": cj,
                "median_rho": float(np.median(vals)),
                "n_maps": len(vals),
            }
        )
    return pd.DataFrame(rows)


def _rank_frame_for_model(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    target_col: str,
    mask_column: str = "in_contour_mask",
    min_residues: int = 30,
) -> tuple[np.ndarray, np.ndarray] | None:
    use = df[df[mask_column].astype(bool)].copy()
    need = [*feature_cols, target_col]
    missing = [c for c in need if c not in use.columns]
    if missing:
        return None
    sub = use[list(need)].apply(pd.to_numeric, errors="coerce")
    ok = sub.notna().all(axis=1)
    if int(ok.sum()) < min_residues:
        return None
    sub = sub.loc[ok]
    mask = np.ones(len(sub), dtype=bool)
    ranked_cols = []
    for col in feature_cols:
        ranked_cols.append(masked_percentile_rank(sub[col].to_numpy(dtype=np.float64), mask))
    X = np.column_stack(ranked_cols)
    y = masked_percentile_rank(sub[target_col].to_numpy(dtype=np.float64), mask)
    if not np.isfinite(y).all() or not np.isfinite(X).all():
        return None
    return X, y


def build_map_frame_for_model(
    df: pd.DataFrame,
    *,
    emdb_id: str,
    model: LomoModelId,
    target_col: str = "q_score",
    min_residues: int = 30,
) -> MapPredictionFrame | None:
    feat_cols = LOMO_MODEL_FEATURES[model]
    ranked = _rank_frame_for_model(
        df,
        feat_cols,
        target_col=target_col,
        min_residues=min_residues,
    )
    if ranked is None:
        return None
    X, y = ranked
    return MapPredictionFrame(
        emdb_id=str(emdb_id),
        X_baseline=X,
        X_full=X,
        y=y,
        n_residues=len(y),
    )


@dataclass(frozen=True)
class LomoModelSummary:
    model: LomoModelId
    n_maps: int
    median_r2: float
    median_delta_r2_vs_baseline: float
    n_positive_delta_r2: int
    sign_test_p_value: float


def run_lomo_model_screen(
    frames_by_model: dict[LomoModelId, list[MapPredictionFrame]],
    *,
    target: str = "q_score",
) -> list[LomoModelSummary]:
    """LOMO R² for each model; ΔR² relative to ``baseline``."""
    baseline_frames = frames_by_model.get("baseline", [])
    if len(baseline_frames) < 3:
        return []

    base_summary = run_lomo_incremental_prediction(baseline_frames, target=target)
    base_r2 = {f.emdb_id: f.r2_baseline for f in base_summary.fold_results}

    summaries: list[LomoModelSummary] = []
    for model, frames in frames_by_model.items():
        if len(frames) < 3:
            continue
        fold_r2: list[float] = []
        for held in frames:
            train = [f for f in frames if f.emdb_id != held.emdb_id]
            coef = ols_fit(np.vstack([f.X_baseline for f in train]), np.concatenate([f.y for f in train]))
            y_hat = ols_predict(coef, held.X_baseline)
            fold_r2.append(ols_r2(held.y, y_hat))

        deltas = []
        for held in frames:
            b = base_r2.get(held.emdb_id, float("nan"))
            idx = next(i for i, f in enumerate(frames) if f.emdb_id == held.emdb_id)
            if np.isfinite(b) and np.isfinite(fold_r2[idx]):
                deltas.append(fold_r2[idx] - b)

        n_pos = int(sum(d > 0 for d in deltas))
        n_f = len(deltas)
        p_sign = float(stats.binomtest(n_pos, n_f, p=0.5, alternative="two-sided").pvalue) if n_f else float("nan")
        summaries.append(
            LomoModelSummary(
                model=model,
                n_maps=len(frames),
                median_r2=float(np.median(fold_r2)),
                median_delta_r2_vs_baseline=float(np.median(deltas)) if deltas else float("nan"),
                n_positive_delta_r2=n_pos,
                sign_test_p_value=p_sign,
            )
        )
    return summaries
