"""Export placement utility figure data for the Figma thesis-placement plugin."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .placement_utility import (
    MAIN_ROC_PREDICTORS,
    PREDICTOR_LABELS,
    RANK_RECOVERY_PROXY_KEYS,
    RANK_RECOVERY_PROXY_LABELS,
    aligned_rank_recovery_rho,
    cohort_representative_roc,
    finite_qv_emdb_ids,
    load_per_map_frames_for_lomo,
)
from .repo_paths import OUTPUTS_ROOT

PLUGIN_DIR = Path("figma-plugins/thesis-placement")
FIGMA_JSON = PLUGIN_DIR / "placement_data.json"
UI_HTML = PLUGIN_DIR / "ui.html"
DATA_MARKER_START = "<!-- PLACEMENT_DATA_START -->"
DATA_MARKER_END = "<!-- PLACEMENT_DATA_END -->"

PREDICTOR_CSV = OUTPUTS_ROOT / "cohort_summary" / "placement_predictor_head_to_head.csv"
RANK_RECOVERY_CSV = OUTPUTS_ROOT / "cohort_summary" / "placement_rank_recovery.csv"
ROC_FIGMA_JSON = OUTPUTS_ROOT / "cohort_summary" / "placement_roc_figma.json"
UTILITY_JSON = OUTPUTS_ROOT / "cohort_summary" / "placement_utility.json"

CATEGORICAL_HEX = ("#E8303A", "#4B6FD4", "#3BBF6A", "#BA3EC3", "#F5C518")
ROC_COLORS = ("#E8303A", "#4B6FD4", "#3BBF6A", "#BA3EC3")

RANK_RECOVERY_PROXIES = tuple(
    (key, RANK_RECOVERY_PROXY_LABELS[key]) for key in RANK_RECOVERY_PROXY_KEYS
)


@dataclass(frozen=True)
class PlacementFigmaExport:
    """Serializable bundle for placement head-to-head, rank recovery, and ROC figures."""

    generated_at: str
    figure_title: str
    q_threshold: float
    n_maps: int
    n_roc_maps: int
    head_to_head: dict
    rank_recovery: dict
    low_q_roc: dict

    def to_dict(self) -> dict:
        return {
            "generated_at": self.generated_at,
            "figure_title": self.figure_title,
            "q_threshold": self.q_threshold,
            "n_maps": self.n_maps,
            "n_roc_maps": self.n_roc_maps,
            "panels": {
                "head_to_head": self.head_to_head,
                "rank_recovery": self.rank_recovery,
                "low_q_roc": self.low_q_roc,
            },
        }


def _load_predictor_rows() -> list[dict[str, str]]:
    if not PREDICTOR_CSV.is_file():
        return []
    with PREDICTOR_CSV.open(newline="") as f:
        return list(csv.DictReader(f))


def _load_rank_recovery_rows() -> list[dict[str, str]]:
    if not RANK_RECOVERY_CSV.is_file():
        return []
    with RANK_RECOVERY_CSV.open(newline="") as f:
        return list(csv.DictReader(f))


def _load_q_threshold() -> float:
    if UTILITY_JSON.is_file():
        try:
            meta = json.loads(UTILITY_JSON.read_text())
            return float(meta.get("q_threshold", 0.5))
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    return 0.5


def _load_roc_panel(q_threshold: float) -> dict:
    if ROC_FIGMA_JSON.is_file():
        raw = json.loads(ROC_FIGMA_JSON.read_text())
        return {
            "kind": "roc",
            "title": (
                f"Low-Q ROC — representative map per predictor "
                f"(n={raw.get('n_maps', '?')} maps with finite ρ(Q, V), "
                f"Q < {q_threshold:.1f})"
            ),
            "x_label": "False positive rate",
            "y_label": "True positive rate (Q < threshold)",
            "curves": raw.get("curves", []),
        }

    qv_ids = finite_qv_emdb_ids()
    if not qv_ids:
        return {"kind": "roc", "title": "Low-Q ROC", "curves": []}

    frames = load_per_map_frames_for_lomo()
    per_map = [(eid, df) for eid, df, _ in frames if eid in qv_ids]
    curves: list[dict] = []
    for i, pid in enumerate(MAIN_ROC_PREDICTORS):
        summary = cohort_representative_roc(
            per_map,
            pid,
            q_threshold=q_threshold,
            eligible_emdb_ids=qv_ids,
        )
        if not summary.fpr:
            continue
        curves.append(
            {
                "predictor": pid,
                "label": PREDICTOR_LABELS[pid],
                "color": ROC_COLORS[i % len(ROC_COLORS)],
                "median_auc": round(summary.median_auc, 4),
                "representative_emdb_id": summary.representative_emdb_id,
                "representative_auc": round(summary.representative_auc, 4),
                "fpr": [round(x, 5) for x in summary.fpr],
                "tpr": [round(y, 5) for y in summary.tpr],
            }
        )

    return {
        "kind": "roc",
        "title": (
            f"Low-Q ROC — representative map per predictor "
            f"(n={len(qv_ids)} maps with finite ρ(Q, V), Q < {q_threshold:.1f})"
        ),
        "x_label": "False positive rate",
        "y_label": "True positive rate (Q < threshold)",
        "curves": curves,
    }


def export_placement_figma_data() -> PlacementFigmaExport:
    """Build placement utility payload from cohort summary CSVs."""
    pred_rows = _load_predictor_rows()
    rr_rows = _load_rank_recovery_rows()
    if not pred_rows:
        raise FileNotFoundError(
            f"Missing {PREDICTOR_CSV} — run scripts/run_placement_utility_analysis.py first."
        )
    if not rr_rows:
        raise FileNotFoundError(
            f"Missing {RANK_RECOVERY_CSV} — run scripts/run_placement_utility_analysis.py first."
        )

    q_threshold = _load_q_threshold()
    n_maps = int(pred_rows[0].get("n_maps", len(rr_rows)) or len(rr_rows))
    qv_ids = finite_qv_emdb_ids()
    n_roc_maps = len(qv_ids)

    predictors: list[dict] = []
    for idx, row in enumerate(pred_rows):
        color = CATEGORICAL_HEX[idx % len(CATEGORICAL_HEX)]
        predictors.append(
            {
                "predictor": row["predictor"],
                "label": row["label"],
                "color": color,
                "frac_low_q_flagged": round(float(row["pooled_frac_low_q_flagged"]), 4),
                "balanced_accuracy": round(float(row["pooled_balanced_accuracy"]), 4),
                "median_auc": round(float(row["median_map_auc"]), 4),
            }
        )

    head_to_head = {
        "kind": "head_to_head_triple",
        "title": "Head-to-head pre-model readouts (Q-score ground truth)",
        "predictors": predictors,
        "panels": [
            {
                "title": "Enrichment",
                "x_label": "Pooled frac. low-Q flagged",
                "metric": "frac_low_q_flagged",
            },
            {
                "title": "Classification (Q threshold)",
                "x_label": "Pooled balanced accuracy",
                "metric": "balanced_accuracy",
            },
            {
                "title": "Low-Q AUC",
                "x_label": "Median per-map AUC",
                "metric": "median_auc",
            },
        ],
    }

    bars: list[dict] = []
    for idx, (col, label) in enumerate(RANK_RECOVERY_PROXIES):
        vals = []
        for row in rr_rows:
            raw = row.get(col, "")
            if raw in ("", "nan"):
                continue
            v = float(raw)
            if np.isfinite(v):
                vals.append(aligned_rank_recovery_rho(v, col))
        med = float(np.median(vals)) if vals else float("nan")
        bars.append(
            {
                "key": col,
                "label": label,
                "median_rho": round(med, 4) if np.isfinite(med) else None,
                "color": CATEGORICAL_HEX[idx % len(CATEGORICAL_HEX)],
            }
        )

    rank_recovery = {
        "kind": "bar",
        "title": "Rank recovery: which pre-model readout tracks Q?",
        "y_label": "Median per-map ρ(Q, proxy), sign-aligned",
        "bars": bars,
    }

    low_q_roc = _load_roc_panel(q_threshold)

    return PlacementFigmaExport(
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        figure_title="Placement utility — Q-score validation",
        q_threshold=q_threshold,
        n_maps=n_maps,
        n_roc_maps=n_roc_maps,
        head_to_head=head_to_head,
        rank_recovery=rank_recovery,
        low_q_roc=low_q_roc,
    )


def _embed_data_in_ui(text: str, embedded: str) -> str:
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


def write_placement_figma_data(
    out_json: Path = FIGMA_JSON,
    *,
    patch_ui: bool = True,
) -> Path:
    """Write ``placement_data.json`` and optionally embed it in ``ui.html``."""
    export = export_placement_figma_data()
    payload = export.to_dict()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2) + "\n")

    if patch_ui and UI_HTML.is_file():
        embedded = (
            f"{DATA_MARKER_START}\n"
            f'<script type="application/json" id="placement-data">\n'
            f"{json.dumps(payload)}\n"
            f"</script>\n"
            f"{DATA_MARKER_END}"
        )
        text = _embed_data_in_ui(UI_HTML.read_text(), embedded)
        UI_HTML.write_text(text)

    return out_json
