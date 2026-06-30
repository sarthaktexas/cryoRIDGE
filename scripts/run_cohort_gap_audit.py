"""Audit cohort manifest rows against on-disk pipeline outputs and write a run list.

Checks ``outputs/emd_<ID>/`` artifacts (``reliability.npz``, BlocRes, Q-score,
B-factor validation) and local ``data/`` / ``pdb/`` inputs. Writes:

- ``cohort/COHORT_GAP_AUDIT.md`` — human-readable table + summary (repo-tracked)
- ``cohort/COHORT_RUN_LIST.sh`` — copy-paste shell commands for missing steps

Example::

    source .venv/bin/activate
    python scripts/run_cohort_gap_audit.py
    python scripts/run_cohort_gap_audit.py --write-run-list
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from cryoem_mrc.repo_paths import (
    COHORT_MANIFEST,
    DOCS_FIGURES_ROOT,
    OUTPUTS_ROOT,
    emd_output_dir,
    find_features_npz,
    halfmap_metrics_npz,
    resolve_halfmap_reliability_dir,
    locres_blocres_mrc,
)

REPO = Path(__file__).resolve().parents[1]
SKIP_SOURCES = frozenset({"excluded", "optional"})
QSCORE_PANEL_EXCLUDE = frozenset({"33736"})
BLOCRES_STATUS = "blocres_status.json"
AUDIT_MD = REPO / "cohort" / "COHORT_GAP_AUDIT.md"
RUN_LIST_SH = REPO / "cohort" / "COHORT_RUN_LIST.sh"


@dataclass(frozen=True)
class RowAudit:
    emdb_id: str
    display_name: str
    flexibility_source: str
    global_resolution_a: str
    has_reference: bool
    has_halves: bool
    has_pdb: bool
    has_reliability: bool
    has_halfmap_metrics: bool
    has_features: bool
    has_blocres: bool
    has_qscore: bool
    has_residue_validation: bool
    has_bfactor_md: bool
    contour_tbd: bool
    skipped: bool
    skip_reason: str

    @property
    def pipeline_ready(self) -> bool:
        return self.has_reliability and self.has_halfmap_metrics

    @property
    def needs_pipeline(self) -> bool:
        if self.skipped or self.contour_tbd:
            return False
        return self.has_reference and self.has_halves and not self.has_reliability

    @property
    def needs_qscore(self) -> bool:
        if self.skipped or self.emdb_id in QSCORE_PANEL_EXCLUDE:
            return False
        return self.has_pdb and self.has_reliability and not self.has_qscore

    @property
    def needs_blocres(self) -> bool:
        if self.skipped or self.contour_tbd:
            return False
        return self.has_reference and self.has_halves and not self.has_blocres


def _has_pdb(path_raw: str) -> bool:
    if not path_raw.strip():
        return False
    path = Path(path_raw.strip())
    return path.suffix.lower() in {".cif", ".pdb"} and path.is_file()


def _blocres_done(emdb_id: str) -> bool:
    if locres_blocres_mrc(emdb_id).is_file():
        return True
    status_path = emd_output_dir(emdb_id) / BLOCRES_STATUS
    if not status_path.is_file():
        return False
    try:
        import json

        payload = json.loads(status_path.read_text())
        return payload.get("status") == "completed"
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def audit_row(row: dict[str, str]) -> RowAudit:
    eid = str(row["emdb_id"]).strip()
    src = row.get("flexibility_source", "").strip()
    contour_raw = row.get("contour", "").strip()
    contour_tbd = not contour_raw or contour_raw.upper() == "TBD"
    skipped = src in SKIP_SOURCES

    ref = Path(row["reference_mrc"])
    h1 = Path(row["half1_path"])
    h2 = Path(row["half2_path"])
    pdb_raw = row.get("flexibility_path_or_pdb", "")

    lh = resolve_halfmap_reliability_dir(eid)
    rel_mrc = lh / f"emd_{eid}_reliability.mrc"
    qscore = lh / "qscore_validation.csv"
    rv = lh / "residue_validation.csv"
    bfac_md = lh / "B_FACTOR_VALIDATION.md"

    features = None
    if ref.is_file() and not contour_tbd:
        try:
            features = find_features_npz(ref.parent, eid, float(contour_raw))
        except ValueError:
            features = None

    skip_reason = ""
    if skipped:
        skip_reason = src

    return RowAudit(
        emdb_id=eid,
        display_name=row.get("display_name", "").strip(),
        flexibility_source=src,
        global_resolution_a=row.get("global_resolution_a", "").strip(),
        has_reference=ref.is_file(),
        has_halves=h1.is_file() and h2.is_file(),
        has_pdb=_has_pdb(pdb_raw),
        has_reliability=rel_mrc.is_file(),
        has_halfmap_metrics=halfmap_metrics_npz(eid).is_file(),
        has_features=features is not None,
        has_blocres=_blocres_done(eid),
        has_qscore=qscore.is_file(),
        has_residue_validation=rv.is_file(),
        has_bfactor_md=bfac_md.is_file(),
        contour_tbd=contour_tbd,
        skipped=skipped,
        skip_reason=skip_reason,
    )


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def narrative_figure_status() -> tuple[list[str], list[str]]:
    narrative = (REPO / "docs" / "THESIS_NARRATIVE.md").read_text(encoding="utf-8")
    refs = sorted(set(re.findall(r"figures/(fig_[a-z0-9_]+\.png)", narrative)))
    have = {p.name for p in DOCS_FIGURES_ROOT.glob("*.png")} if DOCS_FIGURES_ROOT.is_dir() else set()
    missing = [r for r in refs if r not in have]
    return refs, missing


def _status_cell(ok: bool) -> str:
    return "yes" if ok else "**no**"


def render_markdown(
    rows: list[RowAudit],
    *,
    manifest_path: Path,
    outputs_root: Path,
    narrative_refs: list[str],
    missing_figs: list[str],
) -> str:
    active = [r for r in rows if not r.skipped]
    pipelined = [r for r in active if r.has_reliability]
    pending_pipe = [r for r in active if r.needs_pipeline]
    pending_q = [r for r in active if r.needs_qscore]
    pending_blocres = [r for r in active if r.needs_blocres]
    missing_data = [r for r in active if not r.has_reference or not r.has_halves]

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    outputs_ok = outputs_root.is_dir() and outputs_root.exists()
    try:
        manifest_rel = manifest_path.relative_to(REPO)
    except ValueError:
        manifest_rel = manifest_path

    lines = [
        "# Cohort gap audit",
        "",
        f"**Generated:** {ts} UTC",
        f"**Manifest:** `{manifest_rel}`",
        f"**Outputs root:** `{outputs_root}` ({'reachable' if outputs_ok else '**not reachable — mount thesis-data volume**'})",
        "",
        "## Summary",
        "",
        "| Metric | Count |",
        "|--------|------:|",
        f"| Active manifest rows | {len(active)} |",
        f"| Excluded / optional | {sum(r.skipped for r in rows)} |",
        f"| With local reference + halves | {sum(r.has_reference and r.has_halves for r in active)} |",
        f"| **`reliability.npz` present** | **{len(pipelined)}** / {len(active)} |",
        f"| Pending pipeline (`--pending`) | {len(pending_pipe)} |",
        f"| Pending Q-score (`--all`) | {len(pending_q)} |",
        f"| Pending BlocRes (`--all`) | {len(pending_blocres)} |",
        f"| Missing local map data | {len(missing_data)} |",
        f"| Narrative figures in `docs/figures/` | {len(narrative_refs) - len(missing_figs)} / {len(narrative_refs)} |",
        "",
    ]

    if missing_data:
        lines.extend(
            [
                "> **Blocker:** local `data/` symlinks are broken or maps are not downloaded.",
                "> Mount `/Volumes/Undergrad Thesis/thesis-data` (or repoint `data/` / `outputs/`) before running the pipeline.",
                "",
                "## Last verified baseline (2026-06-08)",
                "",
                "When the thesis-data volume was last mounted, [docs/COHORT.md](../docs/COHORT.md) reported:",
                "",
                "- **38/38** maps pipelined (`reliability.npz`), including EMD-33736 (now manifest-`excluded`, RNA-only)",
                "- **33** Q-score validation runs; **38** BlocRes `locres_blocres.mrc` files",
                "- **26/26** narrative figures synced under `docs/figures/`",
                "- **Still open:** spike conformation pair 11497↔11494; FSC-Q Xmipp cluster refresh; Results prose polish",
                "",
                "Re-run `python scripts/run_cohort_gap_audit.py --write-run-list` after mounting to replace this section with live counts.",
                "",
            ]
        )

    if pending_pipe:
        ids = ", ".join(f"EMD-{r.emdb_id}" for r in pending_pipe)
        lines.extend(["## Pending pipeline", "", ids, ""])

    if pending_q:
        ids = ", ".join(f"EMD-{r.emdb_id}" for r in pending_q)
        lines.extend(["## Pending Q-score validation", "", ids, ""])

    if pending_blocres:
        ids = ", ".join(f"EMD-{r.emdb_id}" for r in pending_blocres)
        lines.extend(["## Pending BlocRes", "", ids, ""])

    if missing_figs:
        lines.extend(["## Missing thesis narrative figures", ""])
        lines.extend(f"- `{name}`" for name in missing_figs)
        lines.append("")

    lines.extend(
        [
            "## Per-map status",
            "",
            "| EMDB | Name | res (Å) | src | data | pipeline | BlocRes | Q | B-val |",
            "|------|------|--------:|-----|:----:|:--------:|:-------:|:-:|:-:|",
        ]
    )
    for r in rows:
        if r.skipped:
            lines.append(
                f"| {r.emdb_id} | {r.display_name} | {r.global_resolution_a or '—'} | "
                f"_{r.skip_reason}_ | — | — | — | — | — |"
            )
            continue
        data_ok = r.has_reference and r.has_halves
        pipe_ok = r.has_reliability
        bval = "—"
        if r.flexibility_source == "b_factor":
            bval = _status_cell(r.has_bfactor_md or r.has_residue_validation)
        elif r.has_pdb:
            bval = _status_cell(r.has_residue_validation)
        lines.append(
            f"| {r.emdb_id} | {r.display_name} | {r.global_resolution_a or '—'} | "
            f"{r.flexibility_source} | {_status_cell(data_ok)} | {_status_cell(pipe_ok)} | "
            f"{_status_cell(r.has_blocres)} | {_status_cell(r.has_qscore)} | {bval} |"
        )

    lines.extend(
        [
            "",
            "## Recommended commands (after data volume is mounted)",
            "",
            "```bash",
            "source .venv/bin/activate",
            "export PYTHONUNBUFFERED=1",
            "",
            "# 1. Finish LH pipeline for any missing maps",
            "python scripts/run_cohort_pipeline.py --pending",
            "",
            "# 2. BlocRes local-resolution maps (cohort-wide)",
            "python scripts/run_blocres_local_resolution.py --all",
            "",
            "# 3. Q-score validation (maps with deposited PDB)",
            "python scripts/run_qscore_validation.py --all --cohort-summary --cohort-figure",
            "",
            "# 4. Regenerate cohort figures + sync into docs/figures/",
            "python scripts/rerun_all_figures.py",
            "python scripts/sync_thesis_narrative_figures.py",
            "```",
            "",
            "Machine-readable run list: [`COHORT_RUN_LIST.sh`](./COHORT_RUN_LIST.sh) (regenerate with `--write-run-list`).",
            "",
        ]
    )
    return "\n".join(lines)


def render_run_list(rows: list[RowAudit]) -> str:
    active = [r for r in rows if not r.skipped]
    pending_pipe = [r.emdb_id for r in active if r.needs_pipeline]
    pending_q = [r.emdb_id for r in active if r.needs_qscore]
    pending_blocres = [r.emdb_id for r in active if r.needs_blocres]

    lines = [
        "#!/usr/bin/env bash",
        "# Generated by scripts/run_cohort_gap_audit.py --write-run-list",
        "# Run from repo root with thesis-data volume mounted.",
        "set -euo pipefail",
        "source .venv/bin/activate",
        "export PYTHONUNBUFFERED=1",
        "",
    ]

    if pending_pipe:
        lines.append("# --- LH pipeline (missing reliability.npz) ---")
        for eid in pending_pipe:
            lines.append(f"python scripts/run_cohort_pipeline.py --emd-id {eid}")
        lines.append("")

    if pending_blocres:
        lines.append("# --- BlocRes local resolution ---")
        for eid in pending_blocres:
            lines.append(f"python scripts/run_blocres_local_resolution.py --emd-id {eid}")
        lines.append("")

    if pending_q:
        lines.append("# --- Q-score validation ---")
        for eid in pending_q:
            lines.append(
                f"python scripts/run_qscore_validation.py --emd-id {eid}"
            )
        lines.append(
            "python scripts/run_qscore_validation.py --cohort-summary --cohort-figure"
        )
        lines.append("")

    lines.extend(
        [
            "# --- Figures + thesis sync (always safe to rerun) ---",
            "python scripts/rerun_all_figures.py",
            "python scripts/sync_thesis_narrative_figures.py",
            "",
        ]
    )

    if not pending_pipe and not pending_q and not pending_blocres:
        batch = [
            "",
            "# No per-map gaps detected (or data volume offline). Batch refresh:",
            "python scripts/run_cohort_pipeline.py --pending",
            "python scripts/run_blocres_local_resolution.py --all",
            "python scripts/run_qscore_validation.py --all --cohort-summary --cohort-figure",
            "",
        ]
        lines[6:6] = batch

    return "\n".join(lines)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, default=COHORT_MANIFEST)
    p.add_argument("--outputs-root", type=Path, default=OUTPUTS_ROOT)
    p.add_argument(
        "--write-run-list",
        action="store_true",
        help=f"Write executable {RUN_LIST_SH.name} alongside the audit markdown",
    )
    p.add_argument(
        "--stdout",
        action="store_true",
        help="Print markdown to stdout instead of writing cohort/COHORT_GAP_AUDIT.md",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    manifest_path = args.manifest if args.manifest.is_absolute() else REPO / args.manifest
    outputs_root = args.outputs_root if args.outputs_root.is_absolute() else REPO / args.outputs_root

    rows = [audit_row(r) for r in load_manifest(manifest_path)]
    narrative_refs, missing_figs = narrative_figure_status()
    md = render_markdown(
        rows,
        manifest_path=manifest_path,
        outputs_root=outputs_root,
        narrative_refs=narrative_refs,
        missing_figs=missing_figs,
    )

    if args.stdout:
        print(md)
    else:
        AUDIT_MD.parent.mkdir(parents=True, exist_ok=True)
        AUDIT_MD.write_text(md, encoding="utf-8")
        print(f"[gap_audit] wrote {AUDIT_MD.relative_to(REPO)}", flush=True)

    if args.write_run_list:
        RUN_LIST_SH.write_text(render_run_list(rows), encoding="utf-8")
        RUN_LIST_SH.chmod(0o755)
        print(f"[gap_audit] wrote {RUN_LIST_SH.relative_to(REPO)}", flush=True)

    active = [r for r in rows if not r.skipped]
    pipelined = sum(r.has_reliability for r in active)
    pending = sum(r.needs_pipeline for r in active)
    print(
        f"[gap_audit] pipeline {pipelined}/{len(active)} done; "
        f"{pending} pending; figures {len(narrative_refs) - len(missing_figs)}/{len(narrative_refs)} synced",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
