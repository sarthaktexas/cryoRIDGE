"""Export cohort ρ(Q, V) by protein-class figure data for the Figma plugin."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from style.palette import PALETTES

from thesis.thesis_figures import QscoreClassRow, collect_cohort_q_vs_v_rows

PLUGIN_DIR = Path("figma-plugins/thesis-reliability-by-class")
FIGMA_JSON = PLUGIN_DIR / "q_vs_v_by_class_data.json"
UI_HTML = PLUGIN_DIR / "ui.html"
DATA_MARKER_START = "<!-- Q_VS_V_BY_CLASS_DATA_START -->"
DATA_MARKER_END = "<!-- Q_VS_V_BY_CLASS_DATA_END -->"

VALUE_ATTR = "spearman_q_vs_V"
Y_LABEL = "Spearman ρ(Q-score, V), in-mask Cα"
TITLE = "Cohort: Q-score vs gradient energy by protein class"
Y_LIM = (-0.15, 0.95)


def _class_colors(classes: list[str]) -> dict[str, str]:
    palette = list(PALETTES["categorical"])
    return {cls: palette[i % len(palette)] for i, cls in enumerate(classes)}


def _box_stats(values: np.ndarray) -> dict[str, float]:
    """Matplotlib-style box stats (1.5×IQR whiskers, no fliers)."""
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    iqr = q3 - q1
    lo_fence = q1 - 1.5 * iqr
    hi_fence = q3 + 1.5 * iqr
    in_range = values[(values >= lo_fence) & (values <= hi_fence)]
    whisker_low = float(in_range.min()) if in_range.size else float(values.min())
    whisker_high = float(in_range.max()) if in_range.size else float(values.max())
    return {
        "q1": float(q1),
        "median": float(median),
        "q3": float(q3),
        "whisker_low": whisker_low,
        "whisker_high": whisker_high,
    }


@dataclass(frozen=True)
class QVsVByClassFigmaExport:
    """Serializable bundle for the cohort ρ(Q, V) by-class box-strip figure."""

    generated_at: str
    figure_title: str
    n_structures: int
    cohort_median: float
    y_lim: tuple[float, float]
    panel: dict

    def to_dict(self) -> dict:
        return {
            "generated_at": self.generated_at,
            "figure_title": self.figure_title,
            "n_structures": self.n_structures,
            "cohort_median": round(self.cohort_median, 4),
            "y_lim": [round(self.y_lim[0], 4), round(self.y_lim[1], 4)],
            "panel": self.panel,
        }


def export_q_vs_v_by_class_figma_data() -> QVsVByClassFigmaExport:
    """Build box-strip payload from ``qscore_correlations.csv``."""
    rows = collect_cohort_q_vs_v_rows()
    usable = [r for r in rows if np.isfinite(r.spearman_q_vs_V)]
    if not usable:
        raise FileNotFoundError(
            "No finite ρ(Q, V) rows — run scripts/run_qscore_validation.py --cohort-summary first."
        )

    by_group: dict[str, list[QscoreClassRow]] = {}
    for row in usable:
        key = str(row.protein_class or "unknown")
        by_group.setdefault(key, []).append(row)

    groups = sorted(
        by_group,
        key=lambda g: float(np.median([r.spearman_q_vs_V for r in by_group[g]])),
        reverse=True,
    )
    colors = _class_colors(groups)
    cohort_median = float(np.median([r.spearman_q_vs_V for r in usable]))
    rng = np.random.default_rng(0)

    group_payloads: list[dict] = []
    for idx, grp in enumerate(groups):
        group_rows = by_group[grp]
        vals = np.array([r.spearman_q_vs_V for r in group_rows], dtype=np.float64)
        jitters = rng.uniform(-0.12, 0.12, size=len(group_rows))
        stats = _box_stats(vals)
        group_payloads.append(
            {
                "label": grp,
                "color": colors[grp],
                "position": idx + 1,
                "n": len(group_rows),
                "box": {k: round(v, 4) for k, v in stats.items()},
                "points": [
                    {
                        "emdb_id": row.emdb_id,
                        "label": row.display_name or f"EMD-{row.emdb_id}",
                        "value": round(float(row.spearman_q_vs_V), 4),
                        "jitter": round(float(jitters[i]), 4),
                    }
                    for i, row in enumerate(group_rows)
                ],
            }
        )

    panel = {
        "kind": "box_strip",
        "title": TITLE,
        "y_label": Y_LABEL,
        "cohort_median": round(cohort_median, 4),
        "cohort_median_label": f"cohort median ({cohort_median:+.2f})",
        "box_width": 0.55,
        "groups": group_payloads,
    }

    return QVsVByClassFigmaExport(
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        figure_title=TITLE,
        n_structures=len(usable),
        cohort_median=cohort_median,
        y_lim=Y_LIM,
        panel=panel,
    )


def _embed_data_in_ui(text: str, embedded: str) -> str:
    """Insert JSON block immediately after ``<body>`` (or replace existing block)."""
    for marker_start, marker_end in (
        (DATA_MARKER_START, DATA_MARKER_END),
        ("<!-- RELIABILITY_BY_CLASS_DATA_START -->", "<!-- RELIABILITY_BY_CLASS_DATA_END -->"),
    ):
        if marker_start in text and marker_end in text:
            start = text.index(marker_start)
            end = text.index(marker_end) + len(marker_end)
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


def write_q_vs_v_by_class_figma_data(
    out_json: Path = FIGMA_JSON,
    *,
    patch_ui: bool = True,
) -> Path:
    """Write ``q_vs_v_by_class_data.json`` and optionally embed it in ``ui.html``."""
    export = export_q_vs_v_by_class_figma_data()
    payload = export.to_dict()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2) + "\n")

    if patch_ui and UI_HTML.is_file():
        embedded = (
            f"{DATA_MARKER_START}\n"
            f'<script type="application/json" id="q-vs-v-by-class-data">\n'
            f"{json.dumps(payload)}\n"
            f"</script>\n"
            f"{DATA_MARKER_END}"
        )
        text = _embed_data_in_ui(UI_HTML.read_text(), embedded)
        UI_HTML.write_text(text)

    return out_json


# Backward-compatible aliases for the runner script name.
export_reliability_by_class_figma_data = export_q_vs_v_by_class_figma_data
write_reliability_by_class_figma_data = write_q_vs_v_by_class_figma_data
