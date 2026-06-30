"""Export conformation-pair panel B (Cα RMSD vs Δreliability) for Figma."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import stats

from cryoem_mrc.cohort_labels import cohort_figure_label
from thesis.conformation_pair import (
    UNASSIGNED_DOMAIN_COLOR,
    compute_conformation_pair_stats,
    compute_per_residue_ca_rmsd,
    get_domain_assignments,
    get_domain_regions_for_pair,
    region_matches_residue,
)
from cryoem_mrc.repo_paths import COHORT_MANIFEST
from cryoem_mrc.structure_validation import (
    default_reliability_out_dir,
    match_residue_rows_by_key,
    read_residue_validation_csv,
)

PLUGIN_DIR = Path("figma-plugins/thesis-conformation-pair")
FIGMA_JSON = PLUGIN_DIR / "conformation_pair_data.json"
UI_HTML = PLUGIN_DIR / "ui.html"
DATA_MARKER_START = "<!-- CONFORMATION_PAIR_DATA_START -->"
DATA_MARKER_END = "<!-- CONFORMATION_PAIR_DATA_END -->"

DEFAULT_EMD_A = "41596"
DEFAULT_EMD_B = "41598"


def _domain_color_for_residue(row, regions) -> tuple[str, str | None]:
    for reg in regions:
        if region_matches_residue(reg, row):
            return reg.color, reg.name
    return UNASSIGNED_DOMAIN_COLOR, None


def _per_domain_spearman(
    rmsd: np.ndarray,
    drel: np.ndarray,
    assignments: dict[str, list[int]],
    domain_order: list[str],
    domain_colors: dict[str, str],
) -> list[dict]:
    legend: list[dict] = []
    for name in domain_order:
        idx = assignments.get(name, [])
        color = domain_colors.get(name, UNASSIGNED_DOMAIN_COLOR)
        if len(idx) < 3:
            legend.append({"name": name, "color": color, "rho": None, "n": len(idx)})
            continue
        sub_r = np.asarray(rmsd, dtype=np.float64)[idx]
        sub_d = np.asarray(drel, dtype=np.float64)[idx]
        ok = np.isfinite(sub_r) & np.isfinite(sub_d)
        n_ok = int(ok.sum())
        if n_ok < 3:
            legend.append({"name": name, "color": color, "rho": None, "n": n_ok})
            continue
        rho, _ = stats.spearmanr(sub_r[ok], sub_d[ok])
        legend.append(
            {
                "name": name,
                "color": color,
                "rho": round(float(rho), 4) if np.isfinite(rho) else None,
                "n": n_ok,
            }
        )
    return legend


@dataclass(frozen=True)
class ConformationPairFigmaExport:
    """Serializable bundle for panel B: Cα RMSD vs Δreliability scatter."""

    generated_at: str
    pair_label: str
    emdb_a: str
    emdb_b: str
    name_a: str
    name_b: str
    n_residues: int
    spearman_rho: float
    panel: dict

    def to_dict(self) -> dict:
        return {
            "generated_at": self.generated_at,
            "pair_label": self.pair_label,
            "emdb_a": self.emdb_a,
            "emdb_b": self.emdb_b,
            "name_a": self.name_a,
            "name_b": self.name_b,
            "n_residues": self.n_residues,
            "spearman_rho": round(self.spearman_rho, 4)
            if np.isfinite(self.spearman_rho)
            else None,
            "panel": self.panel,
        }


def export_conformation_pair_scatter_figma_data(
    emdb_a: str = DEFAULT_EMD_A,
    emdb_b: str = DEFAULT_EMD_B,
    *,
    manifest: Path = COHORT_MANIFEST,
    in_mask_both: bool = True,
) -> ConformationPairFigmaExport:
    """Build scatter payload for one conformation pair (default: MsbA outward vs inward)."""
    emdb_a = str(emdb_a).strip()
    emdb_b = str(emdb_b).strip()
    name_a = cohort_figure_label(emdb_a, manifest=manifest, short=False)
    name_b = cohort_figure_label(emdb_b, manifest=manifest, short=False)

    csv_a = default_reliability_out_dir(emdb_a) / "residue_validation.csv"
    csv_b = default_reliability_out_dir(emdb_b) / "residue_validation.csv"
    if not csv_a.is_file() or not csv_b.is_file():
        missing = [p for p in (csv_a, csv_b) if not p.is_file()]
        raise FileNotFoundError(
            "Missing residue validation CSV — run b-factor validation first:\n"
            + "\n".join(f"  {p}" for p in missing)
        )

    rows_a = read_residue_validation_csv(csv_a)
    rows_b = read_residue_validation_csv(csv_b)
    pairs = match_residue_rows_by_key(rows_a, rows_b)
    pair_stats = compute_conformation_pair_stats(
        pairs, emdb_a=emdb_a, emdb_b=emdb_b, in_mask_both=in_mask_both
    )
    use, rmsd = compute_per_residue_ca_rmsd(pairs, in_mask_both=in_mask_both)
    if len(use) < 10:
        raise ValueError(
            f"Too few in-mask residues for EMD-{emdb_a} vs EMD-{emdb_b} (n={len(use)})"
        )

    drel = np.array(
        [b.reliability_score - a.reliability_score for a, b in use],
        dtype=np.float64,
    )
    regions = get_domain_regions_for_pair(emdb_a, emdb_b)
    domain_order = [reg.name for reg in regions]
    domain_colors = {reg.name: reg.color for reg in regions}

    points: list[dict] = []
    for i, (row_a, _row_b) in enumerate(use):
        color, domain = _domain_color_for_residue(row_a, regions)
        points.append(
            {
                "x": round(float(rmsd[i]), 4),
                "y": round(float(drel[i]), 4),
                "color": color,
                "domain": domain,
            }
        )

    legend: list[dict] = []
    if regions and domain_order:
        assignments = get_domain_assignments(use, regions)
        legend = _per_domain_spearman(rmsd, drel, assignments, domain_order, domain_colors)

    rho = pair_stats.spearman_rmsd_vs_delta_reliability
    rho_txt = f"{rho:.2f}" if np.isfinite(rho) else "n/a"
    panel = {
        "kind": "scatter",
        "letter": "b",
        "title": "Cα RMSD vs Δreliability",
        "x_label": "Cα RMSD (Å, B aligned onto A)",
        "y_label": f"Δreliability ({emdb_b} − {emdb_a})",
        "stats_text": f"Spearman ρ(RMSD, Δrel) = {rho_txt}\nn = {len(use)}",
        "point_radius": 2,
        "point_alpha": 0.6,
        "zero_line": True,
        "points": points,
        "legend": legend,
    }

    return ConformationPairFigmaExport(
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        pair_label=f"{name_a} vs {name_b}",
        emdb_a=emdb_a,
        emdb_b=emdb_b,
        name_a=name_a,
        name_b=name_b,
        n_residues=len(use),
        spearman_rho=float(rho),
        panel=panel,
    )


def _embed_data_in_ui(text: str, embedded: str) -> str:
    for marker_start, marker_end in (
        (DATA_MARKER_START, DATA_MARKER_END),
        (
            "<!-- CONFORMATION_PAIR_SCATTER_DATA_START -->",
            "<!-- CONFORMATION_PAIR_SCATTER_DATA_END -->",
        ),
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


def write_conformation_pair_figma_data(
    out_json: Path = FIGMA_JSON,
    *,
    emdb_a: str = DEFAULT_EMD_A,
    emdb_b: str = DEFAULT_EMD_B,
    manifest: Path = COHORT_MANIFEST,
    patch_ui: bool = True,
) -> Path:
    """Write ``conformation_pair_data.json`` and optionally embed it in ``ui.html``."""
    export = export_conformation_pair_scatter_figma_data(
        emdb_a, emdb_b, manifest=manifest
    )
    payload = export.to_dict()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2) + "\n")

    if patch_ui and UI_HTML.is_file():
        embedded = (
            f"{DATA_MARKER_START}\n"
            f'<script type="application/json" id="conformation-pair-data">\n'
            f"{json.dumps(payload)}\n"
            f"</script>\n"
            f"{DATA_MARKER_END}"
        )
        text = _embed_data_in_ui(UI_HTML.read_text(), embedded)
        UI_HTML.write_text(text)

    return out_json
