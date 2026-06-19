"""Export cohort Q vs V summary data for the Figma thesis-q-vs-v plugin."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import stats

from .cohort_labels import cohort_figure_label, load_display_name_map
from .cohort_resolution import (
    COHORT_RESOLUTION_BINS,
    cutoff_median_table,
    summarize_resolution_bins,
    sweep_resolution_bins,
)
from .placement_utility import QSCORE_PANEL_EXCLUDE
from .repo_paths import COHORT_MANIFEST, OUTPUTS_ROOT

PLUGIN_DIR = Path("figma-plugins/thesis-q-vs-v")
FIGMA_JSON = PLUGIN_DIR / "q_vs_v_data.json"
UI_HTML = PLUGIN_DIR / "ui.html"
DATA_MARKER_START = "<!-- Q_VS_V_DATA_START -->"
DATA_MARKER_END = "<!-- Q_VS_V_DATA_END -->"


@dataclass(frozen=True)
class QscoreCohortFigmaExport:
    """Serializable bundle for the cohort Q vs V figure (panels a + b)."""

    generated_at: str
    figure_title: str
    sensitivity_title: str
    n_structures: int
    median_rho: float
    resolution_min_a: float
    resolution_max_a: float
    spearman_rho_vs_resolution: float
    panel_a: dict
    panel_b: dict
    resolution_standard_bins: dict
    resolution_sweep: dict
    resolution_cutoff: dict

    def to_dict(self) -> dict:
        return {
            "generated_at": self.generated_at,
            "figure_title": self.figure_title,
            "sensitivity_title": self.sensitivity_title,
            "n_structures": self.n_structures,
            "stats": {
                "median_rho": round(self.median_rho, 3),
                "spearman_rho_vs_resolution": round(self.spearman_rho_vs_resolution, 3),
            },
            "resolution_range_a": [
                round(self.resolution_min_a, 2),
                round(self.resolution_max_a, 2),
            ],
            "panels": {
                "a": self.panel_a,
                "b": self.panel_b,
                "resolution_standard_bins": self.resolution_standard_bins,
                "resolution_sweep": self.resolution_sweep,
                "resolution_cutoff": self.resolution_cutoff,
            },
        }


def _load_qscore_correlations() -> list[dict[str, str]]:
    path = OUTPUTS_ROOT / "cohort_summary" / "qscore_correlations.csv"
    if not path.is_file():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _resolution_by_id(manifest: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    if not manifest.is_file():
        return out
    with manifest.open(newline="") as f:
        for row in csv.DictReader(f):
            eid = str(row.get("emdb_id", "")).strip()
            raw = row.get("global_resolution_a", "")
            if not eid or not raw:
                continue
            try:
                out[eid] = float(raw)
            except ValueError:
                continue
    return out


def _cohort_records(manifest: Path = COHORT_MANIFEST) -> list[dict]:
    """Per-structure rows for the cohort Q vs V figure (mirrors run_qscore_validation)."""
    rows = _load_qscore_correlations()
    if not rows:
        raise FileNotFoundError(
            "Missing outputs/cohort_summary/qscore_correlations.csv — "
            "run scripts/run_qscore_validation.py --cohort-summary first."
        )

    res_by_id = _resolution_by_id(manifest)
    name_by_id = load_display_name_map(manifest)
    recs: list[dict] = []

    for row in rows:
        raw = row.get("spearman_q_vs_V", "")
        if raw in ("", "nan"):
            continue
        rho = float(raw)
        if not np.isfinite(rho):
            continue
        eid = str(row["emdb_id"]).strip()
        if eid in QSCORE_PANEL_EXCLUDE:
            continue
        recs.append(
            {
                "emdb_id": eid,
                "label": cohort_figure_label(eid, names=name_by_id),
                "rho": rho,
                "n_in_mask": int(row.get("n_in_mask", 0) or 0),
                "resolution_a": float(res_by_id.get(eid, float("nan"))),
            }
        )

    if not recs:
        raise FileNotFoundError("No finite ρ(Q, V) rows in qscore_correlations.csv")
    recs.sort(key=lambda d: d["rho"])
    return recs


def export_q_vs_v_cohort_figma_data(
    *,
    manifest: Path = COHORT_MANIFEST,
) -> QscoreCohortFigmaExport:
    """Build cohort summary payload from ``qscore_correlations.csv``."""
    recs = _cohort_records(manifest)
    rhos = np.array([d["rho"] for d in recs], dtype=np.float64)
    res = np.array([d["resolution_a"] for d in recs], dtype=np.float64)
    median_rho = float(np.median(rhos))

    res_finite = res[np.isfinite(res)]
    vmin = float(res_finite.min()) if res_finite.size else 0.0
    vmax = float(res_finite.max()) if res_finite.size else 1.0

    m = np.isfinite(res)
    if m.sum() >= 3:
        rho_rr, _ = stats.spearmanr(res[m], rhos[m])
        coef = np.polyfit(res[m], rhos[m], 1)
        xline = np.linspace(float(res[m].min()), float(res[m].max()), 50)
        yline = np.polyval(coef, xline)
        trend = [
            {"x": round(float(x), 3), "y": round(float(y), 4)}
            for x, y in zip(xline, yline, strict=True)
        ]
        panel_b_title = f"ρ(Q,V) vs resolution (Spearman={float(rho_rr):+.2f})"
        spearman_rr = float(rho_rr)
    else:
        trend = []
        panel_b_title = "ρ(Q,V) vs resolution"
        spearman_rr = float("nan")

    bars = [
        {
            "emdb_id": d["emdb_id"],
            "label": d["label"],
            "rho": round(float(d["rho"]), 4),
            "resolution_a": round(float(d["resolution_a"]), 2)
            if np.isfinite(d["resolution_a"])
            else None,
            "n_in_mask": int(d["n_in_mask"]),
        }
        for d in recs
    ]

    scatter_pts = [
        {
            "resolution_a": round(float(d["resolution_a"]), 2),
            "rho": round(float(d["rho"]), 4),
            "emdb_id": d["emdb_id"],
        }
        for d in recs
        if np.isfinite(d["resolution_a"])
    ]

    panel_a = {
        "letter": "a",
        "kind": "barh",
        "title": f"Per-structure Q-score vs V (median ρ={median_rho:+.2f})",
        "x_label": "Spearman ρ(Q-score, V), in-mask Cα",
        "color_label": "Global resolution (Å)",
        "median_rho": round(median_rho, 3),
        "resolution_min_a": round(vmin, 2),
        "resolution_max_a": round(vmax, 2),
        "bars": bars,
    }

    panel_b = {
        "letter": "b",
        "kind": "scatter",
        "title": panel_b_title,
        "x_label": "Global resolution (Å)",
        "y_label": "Spearman ρ(Q-score, V)",
        "points": scatter_pts,
        "trend_line": trend,
        "cutoff_a": 4.0,
    }

    pairs = [(float(d["resolution_a"]), float(d["rho"])) for d in recs if np.isfinite(d["resolution_a"])]
    standard_rows = summarize_resolution_bins(pairs, bins=COHORT_RESOLUTION_BINS)
    standard_bins = [
        {
            "label": str(b["bin_label"]),
            "median_rho": round(float(b["median_rho"]), 4),
            "n": int(b["n"]),
        }
        for b in standard_rows
    ]

    fine_bins = sweep_resolution_bins(pairs, width=0.5, lo=2.0, hi=6.0)
    sweep_pts = [
        {
            "x": round((float(b["lo_a"]) + float(b["hi_a"])) / 2, 3),
            "y": round(float(b["median_rho"]), 4),
            "n": int(b["n"]),
            "label": str(b["bin_label"]),
        }
        for b in fine_bins
    ]

    resolution_sweep = {
        "letter": "b",
        "kind": "line",
        "title": "0.5 Å bin sweep (2–6 Å)",
        "x_label": "Global resolution (Å, bin center)",
        "y_label": "Median ρ(Q, V)",
        "bin_width_a": 0.5,
        "range_a": [2.0, 6.0],
        "cutoff_a": 4.0,
        "cutoff_label": "4 Å cutoff",
        "points": sweep_pts,
    }

    resolution_standard_bins = {
        "letter": "a",
        "kind": "bar",
        "title": "Standard resolution bins",
        "x_label": "Global resolution bin",
        "y_label": "Median Spearman ρ(Q, V)",
        "bars": standard_bins,
    }

    cutoff_rows = cutoff_median_table(pairs)
    series_le = [
        {
            "x": round(float(r["cutoff_a"]), 1),
            "y": round(float(r["median_rho_le_cutoff"]), 4),
            "n": int(r["n_le_cutoff"]),
        }
        for r in cutoff_rows
        if np.isfinite(float(r["median_rho_le_cutoff"]))
    ]
    series_gt = [
        {
            "x": round(float(r["cutoff_a"]), 1),
            "y": round(float(r["median_rho_gt_cutoff"]), 4),
            "n": int(r["n_gt_cutoff"]),
        }
        for r in cutoff_rows
        if np.isfinite(float(r["median_rho_gt_cutoff"]))
    ]

    resolution_cutoff = {
        "letter": "c",
        "kind": "cutoff",
        "title": "Where does signal drop?",
        "x_label": "Resolution ceiling (Å)",
        "y_label": "Median ρ(Q, V)",
        "cutoff_a": 4.0,
        "cutoff_label": "4 Å",
        "legend_le": "res ≤ cutoff",
        "legend_gt": "res > cutoff",
        "series_le": series_le,
        "series_gt": series_gt,
    }

    sensitivity_title = (
        f"ρ(Q, V) resolution sensitivity — n={len(recs)}, "
        f"cohort median={median_rho:+.2f}"
        + (f", ρ vs res={spearman_rr:+.2f}" if np.isfinite(spearman_rr) else "")
    )

    return QscoreCohortFigmaExport(
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        figure_title="Q-score vs constraint V — cohort summary",
        sensitivity_title=sensitivity_title,
        n_structures=len(recs),
        median_rho=median_rho,
        resolution_min_a=vmin,
        resolution_max_a=vmax,
        spearman_rho_vs_resolution=spearman_rr,
        panel_a=panel_a,
        panel_b=panel_b,
        resolution_standard_bins=resolution_standard_bins,
        resolution_sweep=resolution_sweep,
        resolution_cutoff=resolution_cutoff,
    )


def _embed_data_in_ui(text: str, embedded: str) -> str:
    """Insert JSON block immediately after ``<body>`` (or replace existing block)."""
    if DATA_MARKER_START in text and DATA_MARKER_END in text:
        start = text.index(DATA_MARKER_START)
        end = text.index(DATA_MARKER_END) + len(DATA_MARKER_END)
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


def write_q_vs_v_figma_data(
    out_json: Path = FIGMA_JSON,
    *,
    patch_ui: bool = True,
    manifest: Path = COHORT_MANIFEST,
) -> Path:
    """Write ``q_vs_v_data.json`` and optionally embed it in ``ui.html``."""
    export = export_q_vs_v_cohort_figma_data(manifest=manifest)
    payload = export.to_dict()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2) + "\n")

    if patch_ui and UI_HTML.is_file():
        embedded = (
            f"{DATA_MARKER_START}\n"
            f'<script type="application/json" id="q-vs-v-data">\n'
            f"{json.dumps(payload)}\n"
            f"</script>\n"
            f"{DATA_MARKER_END}"
        )
        text = _embed_data_in_ui(UI_HTML.read_text(), embedded)
        UI_HTML.write_text(text)

    return out_json
