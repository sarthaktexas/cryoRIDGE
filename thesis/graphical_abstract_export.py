"""Export cohort scatter + calibration data for the Figma graphical-abstract plugin."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from cryoem_mrc.cohort_labels import load_display_name_map
from cryoem_mrc.cohort_resolution import COHORT_RESOLUTION_BINS
from thesis.placement_utility import (
    QSCORE_PANEL_EXCLUDE,
    load_map_with_qscore,
    rank_auc,
)
from cryoem_mrc.repo_paths import ANCHOR_EMDB_ID, COHORT_MANIFEST, OUTPUTS_ROOT

PLUGIN_DIR = Path("figma-plugins/thesis-graphical-abstract")
COHORT_JSON = PLUGIN_DIR / "cohort_data.json"
UI_HTML = PLUGIN_DIR / "ui.html"
COHORT_DATA_MARKER_START = "<!-- COHORT_DATA_START -->"
COHORT_DATA_MARKER_END = "<!-- COHORT_DATA_END -->"

LOW_Q_THRESHOLD = 0.50
ATOMIC_REGIME_LO_A = 2.5
ATOMIC_REGIME_HI_A = 4.0
DEFAULT_MIN_IN_MASK = 500


@dataclass(frozen=True)
class GraphicalAbstractExport:
    """Serializable bundle for the Figma plugin (single exemplar map by default)."""

    generated_at: str
    emdb_id: str
    display_name: str
    global_resolution_a: float
    regime_label: str
    selection_note: str
    n_maps: int
    n_residues_pooled: int
    q_threshold: float
    spearman_q_vs_locres: float
    spearman_q_vs_reliability: float
    spearman_q_vs_v: float
    reliability_map_auc: float
    locres_map_auc: float
    locres_q_points: list[dict[str, float]]
    reliability_q_points: list[dict[str, float]]
    calibration_bins: list[dict[str, float]]
    predictor_auc: list[dict[str, float | str]]

    def to_dict(self) -> dict:
        return {
            "generated_at": self.generated_at,
            "scope": {
                "mode": "exemplar_map",
                "regime_label": self.regime_label,
                "resolution_lo_a": ATOMIC_REGIME_LO_A,
                "resolution_hi_a": ATOMIC_REGIME_HI_A,
                "emdb_id": self.emdb_id,
                "display_name": self.display_name,
                "global_resolution_a": round(self.global_resolution_a, 2),
                "selection_note": self.selection_note,
            },
            "n_maps": self.n_maps,
            "n_residues_pooled": self.n_residues_pooled,
            "q_threshold": self.q_threshold,
            "stats": {
                "spearman_q_vs_locres": round(self.spearman_q_vs_locres, 3),
                "spearman_q_vs_reliability": round(self.spearman_q_vs_reliability, 3),
                "spearman_q_vs_v": round(self.spearman_q_vs_v, 3),
                "reliability_median_auc": round(self.reliability_map_auc, 3),
                "locres_median_auc": round(self.locres_map_auc, 3),
            },
            "panels": {
                "locres_q": {
                    "title": "BlocRes local resolution vs Q-score",
                    "x_label": "BlocRes local resolution (Å)",
                    "y_label": "Q-score",
                    "spearman_rho": round(self.spearman_q_vs_locres, 3),
                    "points": self.locres_q_points,
                },
                "reliability_q": {
                    "title": "Reliability score vs Q-score",
                    "x_label": "Reliability score (percentile rank of V)",
                    "y_label": "Q-score",
                    "spearman_rho": round(self.spearman_q_vs_reliability, 3),
                    "points": self.reliability_q_points,
                },
                "calibration": {
                    "title": "Mean Q by reliability decile",
                    "x_label": "Reliability decile",
                    "y_label": "Mean Q-score",
                    "bins": self.calibration_bins,
                },
                "predictor_auc": {
                    "title": f"Low-Q detection AUC (Q < {self.q_threshold:.2f})",
                    "bars": self.predictor_auc,
                },
            },
        }


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


def _regime_label(lo: float, hi: float) -> str:
    for b in COHORT_RESOLUTION_BINS:
        if b.lo == lo and b.hi == hi:
            return b.label
    return f"{lo:.1f}–{hi:.1f} Å"


def _load_rank_recovery_rows() -> list[dict[str, str]]:
    path = OUTPUTS_ROOT / "cohort_summary" / "placement_rank_recovery.csv"
    if not path.is_file():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _emd_ids_from_qscore_cohort() -> list[str]:
    ids: list[str] = []
    csv_path = OUTPUTS_ROOT / "cohort_summary" / "qscore_correlations.csv"
    if not csv_path.is_file():
        return ids
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            eid = str(row["emdb_id"]).strip()
            if eid and eid not in QSCORE_PANEL_EXCLUDE:
                ids.append(eid)
    return ids


def _candidates_in_regime(
  *,
    resolution_lo: float,
    resolution_hi: float,
    min_in_mask: int,
) -> list[dict[str, float | str | int]]:
    """Maps in resolution bin with finite ρ(Q, reliability)."""
    qscore_ids = set(_emd_ids_from_qscore_cohort())
    out: list[dict[str, float | str | int]] = []
    for row in _load_rank_recovery_rows():
        eid = str(row["emdb_id"]).strip()
        if eid not in qscore_ids:
            continue
        res = float(row["global_resolution_a"])
        n_mask = int(row["n_in_mask"])
        rho_rel = float(row["spearman_q_vs_reliability"])
        if not (resolution_lo <= res < resolution_hi):
            continue
        if n_mask < min_in_mask:
            continue
        if not np.isfinite(rho_rel):
            continue
        out.append(
            {
                "emdb_id": eid,
                "global_resolution_a": res,
                "n_in_mask": n_mask,
                "spearman_q_vs_reliability": rho_rel,
                "spearman_q_vs_v": float(row["spearman_q_vs_v"]),
                "spearman_q_vs_locres": float(row["spearman_q_vs_locres"]),
            }
        )
    return out


def select_exemplar_map(
    *,
    emd_id: str | None = None,
    resolution_lo: float = ATOMIC_REGIME_LO_A,
    resolution_hi: float = ATOMIC_REGIME_HI_A,
    min_in_mask: int = DEFAULT_MIN_IN_MASK,
    prefer_anchor: bool = False,
) -> dict[str, float | str | int]:
    """
    Pick one map for the graphical abstract.

    Default: highest ρ(Q, reliability) in the atomic-building resolution bin.
  Use ``prefer_anchor=True`` or explicit ``emd_id`` for EMD-49450.
    """
    candidates = _candidates_in_regime(
        resolution_lo=resolution_lo,
        resolution_hi=resolution_hi,
        min_in_mask=min_in_mask,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No Q-score maps in {resolution_lo:.1f}–{resolution_hi:.1f} Å with "
            f"n_in_mask ≥ {min_in_mask}. Check placement_rank_recovery.csv."
        )

    if emd_id:
        for c in candidates:
            if str(c["emdb_id"]) == str(emd_id).strip():
                return c
        raise FileNotFoundError(
            f"EMD-{emd_id} not in atomic-building candidates "
            f"({resolution_lo:.1f}–{resolution_hi:.1f} Å, n≥{min_in_mask})."
        )

    if prefer_anchor:
        for c in candidates:
            if str(c["emdb_id"]) == ANCHOR_EMDB_ID:
                return c

    return max(candidates, key=lambda c: float(c["spearman_q_vs_reliability"]))


def _points_from_columns(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
) -> list[dict[str, float]]:
    sub = df[[x_col, y_col]].apply(pd.to_numeric, errors="coerce").dropna()
    return [
        {"x": round(float(row[x_col]), 4), "y": round(float(row[y_col]), 4)}
        for _, row in sub.iterrows()
    ]


def _subsample_df(df: pd.DataFrame, max_total: int, seed: int) -> pd.DataFrame:
    if len(df) <= max_total:
        return df.copy()
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), size=max_total, replace=False)
    return df.iloc[idx].copy()


def _calibration_bins(df: pd.DataFrame, n_bins: int = 10) -> list[dict[str, float]]:
    rel = pd.to_numeric(df["reliability_score"], errors="coerce")
    q = pd.to_numeric(df["q_score"], errors="coerce")
    ok = rel.notna() & q.notna()
    rel = rel[ok].to_numpy(dtype=np.float64)
    q = q[ok].to_numpy(dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[dict[str, float]] = []
    for i in range(n_bins):
        lo, hi = float(edges[i]), float(edges[i + 1])
        m = (rel >= lo) & (rel < hi) if i < n_bins - 1 else (rel >= lo) & (rel <= hi)
        if not m.any():
            continue
        bins.append(
            {
                "bin_lo": round(lo, 2),
                "bin_hi": round(hi, 2),
                "label": f"{i + 1}",
                "n_residues": int(m.sum()),
                "mean_q": round(float(np.mean(q[m])), 3),
                "median_q": round(float(np.median(q[m])), 3),
            }
        )
    return bins


def _predictor_auc_bars(df: pd.DataFrame) -> list[dict[str, float | str]]:
    q = pd.to_numeric(df["q_score"], errors="coerce").to_numpy(dtype=np.float64)
    rel = pd.to_numeric(df["reliability_score"], errors="coerce").to_numpy(dtype=np.float64)
    loc = pd.to_numeric(df["local_resolution"], errors="coerce").to_numpy(dtype=np.float64)
    v = pd.to_numeric(df.get("v_metric"), errors="coerce").to_numpy(dtype=np.float64)
    low_q = (q < LOW_Q_THRESHOLD).astype(np.int8)
    m = np.isfinite(q) & np.isfinite(rel) & np.isfinite(loc)
    q = q[m]
    low_q = low_q[m]
    rel = rel[m]
    loc = loc[m]
    v = v[m]

    rel_auc = rank_auc(low_q, -rel)
    loc_auc = rank_auc(low_q, loc)
    omit_auc = float("nan")
    if "build_zone" in df.columns:
        zone = pd.to_numeric(df.loc[m, "build_zone"], errors="coerce").to_numpy(dtype=np.float64)
        omit_auc = rank_auc(low_q, -zone)

    bars = [
        {"id": "reliability", "label": "Reliability < 0.33", "auc": round(rel_auc, 3)},
        {"id": "locres", "label": "BlocRes > median Å", "auc": round(loc_auc, 3)},
    ]
    if np.isfinite(omit_auc):
        bars.append({"id": "omit_zone", "label": "Omit build zone", "auc": round(omit_auc, 3)})
    return bars


def export_graphical_abstract_cohort_data(
    *,
    manifest: Path = COHORT_MANIFEST,
    emd_id: str | None = None,
    prefer_anchor: bool = False,
    resolution_lo: float = ATOMIC_REGIME_LO_A,
    resolution_hi: float = ATOMIC_REGIME_HI_A,
    min_in_mask: int = DEFAULT_MIN_IN_MASK,
    max_total: int = 1200,
    seed: int = 42,
) -> GraphicalAbstractExport:
    """Export one high-ρ exemplar map from the atomic-building resolution regime."""
    pick = select_exemplar_map(
        emd_id=emd_id,
        resolution_lo=resolution_lo,
        resolution_hi=resolution_hi,
        min_in_mask=min_in_mask,
        prefer_anchor=prefer_anchor,
    )
    exemplar_id = str(pick["emdb_id"])
    names = load_display_name_map(manifest)

    df = load_map_with_qscore(exemplar_id, manifest=manifest)
    if df is None:
        raise FileNotFoundError(f"Could not load per-residue metrics for EMD-{exemplar_id}")

    q = pd.to_numeric(df["q_score"], errors="coerce")
    rel = pd.to_numeric(df.get("reliability_score"), errors="coerce")
    loc = pd.to_numeric(df.get("local_resolution"), errors="coerce")
    ok = q.notna() & rel.notna() & loc.notna()
    use = df.loc[ok].copy()
    if len(use) < min_in_mask:
        raise FileNotFoundError(
            f"EMD-{exemplar_id} has only {len(use)} usable Cα after merging Q + metrics."
        )

    sampled = _subsample_df(use, max_total=max_total, seed=seed)
    locres_points = _points_from_columns(sampled, "local_resolution", "q_score")
    rel_points = _points_from_columns(sampled, "reliability_score", "q_score")
    calibration = _calibration_bins(use, n_bins=10)

    q_all = pd.to_numeric(use["q_score"], errors="coerce").to_numpy(dtype=np.float64)
    rel_all = pd.to_numeric(use["reliability_score"], errors="coerce").to_numpy(dtype=np.float64)
    loc_all = pd.to_numeric(use["local_resolution"], errors="coerce").to_numpy(dtype=np.float64)
    v_all = pd.to_numeric(use.get("v_metric"), errors="coerce").to_numpy(dtype=np.float64)

    rho_loc = _finite_spearman(loc_all, q_all)
    rho_rel = _finite_spearman(rel_all, q_all)
    rho_v = _finite_spearman(v_all, q_all)
    predictor_bars = _predictor_auc_bars(use)
    rel_auc = next((b["auc"] for b in predictor_bars if b["id"] == "reliability"), float("nan"))
    loc_auc = next((b["auc"] for b in predictor_bars if b["id"] == "locres"), float("nan"))

    display = names.get(exemplar_id, f"EMD-{exemplar_id}")
    regime = _regime_label(resolution_lo, resolution_hi)
    if emd_id:
        note = f"User-selected EMD-{exemplar_id} ({regime})"
    elif prefer_anchor:
        note = f"Thesis anchor EMD-{ANCHOR_EMDB_ID} ({regime})"
    else:
        note = (
            f"Highest ρ(Q, reliability) in {regime} with n≥{min_in_mask} "
            f"(ρ={float(pick['spearman_q_vs_reliability']):+.3f})"
        )

    return GraphicalAbstractExport(
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        emdb_id=exemplar_id,
        display_name=display,
        global_resolution_a=float(pick["global_resolution_a"]),
        regime_label=regime,
        selection_note=note,
        n_maps=1,
        n_residues_pooled=int(len(use)),
        q_threshold=LOW_Q_THRESHOLD,
        spearman_q_vs_locres=rho_loc,
        spearman_q_vs_reliability=rho_rel,
        spearman_q_vs_v=rho_v,
        reliability_map_auc=float(rel_auc),
        locres_map_auc=float(loc_auc),
        locres_q_points=locres_points,
        reliability_q_points=rel_points,
        calibration_bins=calibration,
        predictor_auc=predictor_bars,
    )


def _embed_cohort_data_in_ui(text: str, embedded: str) -> str:
    """Insert cohort JSON immediately after ``<body>`` (must load before UI script runs)."""
    if COHORT_DATA_MARKER_START in text and COHORT_DATA_MARKER_END in text:
        start = text.index(COHORT_DATA_MARKER_START)
        end = text.index(COHORT_DATA_MARKER_END) + len(COHORT_DATA_MARKER_END)
        while end < len(text) and text[end] in "\n\r":
            end += 1
        text = text[:start] + text[end:]

    body_tag = "<body>"
    idx = text.find(body_tag)
    if idx < 0:
        return text.replace("</body>", embedded + "\n</body>")
    insert_at = idx + len(body_tag)
    if insert_at < len(text) and text[insert_at] == "\n":
        insert_at += 1
    return text[:insert_at] + "\n" + embedded + "\n" + text[insert_at:]


def write_graphical_abstract_cohort_data(
    out_json: Path = COHORT_JSON,
    *,
    patch_ui: bool = True,
    **kwargs,
) -> Path:
    """Write ``cohort_data.json`` and optionally embed it in ``ui.html``."""
    export = export_graphical_abstract_cohort_data(**kwargs)
    payload = export.to_dict()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2) + "\n")

    if patch_ui and UI_HTML.is_file():
        embedded = (
            f"{COHORT_DATA_MARKER_START}\n"
            f'<script type="application/json" id="cohort-data">\n'
            f"{json.dumps(payload)}\n"
            f"</script>\n"
            f"{COHORT_DATA_MARKER_END}"
        )
        text = _embed_cohort_data_in_ui(UI_HTML.read_text(), embedded)
        UI_HTML.write_text(text)

    return out_json
