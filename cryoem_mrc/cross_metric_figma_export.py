"""Export cohort cross-metric figure data for the Figma thesis-cross-metric plugin."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .cohort_labels import cohort_figure_label, load_display_name_map
from .half_map_repro import WINDOWED_HALFMAP_CORRELATION_KEY, WINDOWED_HALFMAP_CORRELATION_LABEL
from .metric_comparison import METRIC_COLUMNS
from .repo_paths import COHORT_MANIFEST, emd_output_dir

PLUGIN_DIR = Path("figma-plugins/thesis-cross-metric")
FIGMA_JSON = PLUGIN_DIR / "cross_metric_data.json"
UI_HTML = PLUGIN_DIR / "ui.html"
DATA_MARKER_START = "<!-- CROSS_METRIC_DATA_START -->"
DATA_MARKER_END = "<!-- CROSS_METRIC_DATA_END -->"

METRIC_LABELS = {
    "v_metric": "V",
    "b_factor": "B_iso",
    WINDOWED_HALFMAP_CORRELATION_KEY: WINDOWED_HALFMAP_CORRELATION_LABEL.title(),
    "local_variance": "Local variance",
    "local_resolution": "BlocRes locres",
}

LOCres_PAIR_KEYS = (
    ("v_metric", "local_resolution"),
    ("b_factor", "local_resolution"),
    (WINDOWED_HALFMAP_CORRELATION_KEY, "local_resolution"),
    ("local_variance", "local_resolution"),
)

LOCres_PAIR_LABELS = (
    "V vs locres",
    "B vs locres",
    "windowed corr vs locres",
    "Var vs locres",
)

CATEGORICAL_HEX = ("#E8303A", "#4B6FD4", "#3BBF6A", "#BA3EC3")


@dataclass(frozen=True)
class CrossMetricFigmaExport:
    """Serializable bundle for cohort cross-metric figures."""

    generated_at: str
    figure_title: str
    n_structures: int
    median_heatmap: dict
    locres_pairs: dict

    def to_dict(self) -> dict:
        return {
            "generated_at": self.generated_at,
            "figure_title": self.figure_title,
            "n_structures": self.n_structures,
            "panels": {
                "median_heatmap": self.median_heatmap,
                "locres_pairs": self.locres_pairs,
            },
        }


def _eligible_ids(manifest: Path) -> list[str]:
    ids: list[str] = []
    with manifest.open(newline="") as f:
        for row in csv.DictReader(f):
            eid = str(row["emdb_id"]).strip()
            corr = emd_output_dir(eid) / "metric_comparison" / "cross_metric_correlations.csv"
            if corr.is_file():
                ids.append(eid)
    return ids


def _read_corr(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    return df


def _collect_median_matrix(ids: list[str]) -> tuple[np.ndarray, list[str]]:
    cols = list(METRIC_COLUMNS)
    stacks: dict[tuple[str, str], list[float]] = {}
    for eid in ids:
        corr = _read_corr(emd_output_dir(eid) / "metric_comparison" / "cross_metric_correlations.csv")
        for i, ci in enumerate(cols):
            if ci not in corr.index:
                continue
            for j, cj in enumerate(cols):
                if j < i or cj not in corr.columns:
                    continue
                val = float(corr.loc[ci, cj])
                if np.isfinite(val):
                    stacks.setdefault((ci, cj), []).append(val)

    mat = np.full((len(cols), len(cols)), np.nan, dtype=np.float64)
    for i, ci in enumerate(cols):
        for j, cj in enumerate(cols):
            if j < i:
                mat[i, j] = mat[j, i]
                continue
            vals = stacks.get((ci, cj), [])
            mat[i, j] = float(np.median(vals)) if vals else float("nan")
    return mat, cols


def _collect_locres_pairs(ids: list[str], *, manifest: Path) -> list[dict]:
    names = load_display_name_map(manifest)
    recs: list[dict] = []
    for eid in ids:
        corr = _read_corr(emd_output_dir(eid) / "metric_comparison" / "cross_metric_correlations.csv")
        rec: dict = {"emdb_id": eid, "label": cohort_figure_label(eid, names=names)}
        for a, b in LOCres_PAIR_KEYS:
            key = f"{a}|{b}"
            if a in corr.index and b in corr.columns:
                rec[key] = float(corr.loc[a, b])
            else:
                rec[key] = float("nan")
        recs.append(rec)
    return recs


def export_cross_metric_figma_data(*, manifest: Path = COHORT_MANIFEST) -> CrossMetricFigmaExport:
    """Build cross-metric cohort payload from per-map correlation CSVs."""
    ids = _eligible_ids(manifest)
    if len(ids) < 3:
        raise FileNotFoundError(
            "Fewer than three maps with outputs/emd_<ID>/metric_comparison/"
            "cross_metric_correlations.csv — run scripts/run_metric_comparison_export.py first."
        )

    mat, cols = _collect_median_matrix(ids)
    labels = [METRIC_LABELS.get(c, c) for c in cols]
    cells: list[dict] = []
    for i in range(len(cols)):
        for j in range(len(cols)):
            val = float(mat[i, j])
            if not np.isfinite(val):
                continue
            cells.append(
                {
                    "row": i,
                    "col": j,
                    "value": round(val, 4),
                    "label": f"{val:+.2f}",
                }
            )

    median_heatmap = {
        "kind": "heatmap",
        "title": "Cross-metric coupling — cohort median",
        "colorbar_label": "Median Spearman ρ (in-mask Cα)",
        "vmin": -1.0,
        "vmax": 1.0,
        "row_labels": labels,
        "col_labels": labels,
        "cells": cells,
    }

    recs = _collect_locres_pairs(ids, manifest=manifest)
    usable = [r for r in recs if np.isfinite(float(r.get("v_metric|local_resolution", float("nan"))))]
    usable.sort(key=lambda d: float(d["v_metric|local_resolution"]))

    series: list[dict] = []
    for idx, ((a, b), short_label, color) in enumerate(
        zip(LOCres_PAIR_KEYS, LOCres_PAIR_LABELS, CATEGORICAL_HEX, strict=True)
    ):
        key = f"{a}|{b}"
        series.append(
            {
                "key": key,
                "label": short_label,
                "color": color,
                "offset_index": idx - 1.5,
                "values": [
                    {
                        "emdb_id": r["emdb_id"],
                        "label": r["label"],
                        "rho": round(float(r[key]), 4) if np.isfinite(float(r[key])) else None,
                    }
                    for r in usable
                ],
            }
        )

    locres_pairs = {
        "kind": "grouped_barh",
        "title": f"Per-map locres coupling (n = {len(usable)})",
        "x_label": "Spearman ρ vs BlocRes local resolution",
        "series": series,
        "structures": [{"emdb_id": r["emdb_id"], "label": r["label"]} for r in usable],
    }

    return CrossMetricFigmaExport(
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        figure_title="Cross-metric coupling — cohort",
        n_structures=len(usable),
        median_heatmap=median_heatmap,
        locres_pairs=locres_pairs,
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


def write_cross_metric_figma_data(
    out_json: Path = FIGMA_JSON,
    *,
    patch_ui: bool = True,
    manifest: Path = COHORT_MANIFEST,
) -> Path:
    """Write ``cross_metric_data.json`` and optionally embed it in ``ui.html``."""
    export = export_cross_metric_figma_data(manifest=manifest)
    payload = export.to_dict()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2) + "\n")

    if patch_ui and UI_HTML.is_file():
        embedded = (
            f"{DATA_MARKER_START}\n"
            f'<script type="application/json" id="cross-metric-data">\n'
            f"{json.dumps(payload)}\n"
            f"</script>\n"
            f"{DATA_MARKER_END}"
        )
        text = _embed_data_in_ui(UI_HTML.read_text(), embedded)
        UI_HTML.write_text(text)

    return out_json
