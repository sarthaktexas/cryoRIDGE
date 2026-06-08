#!/usr/bin/env python3
"""Regenerate all thesis/publication figures (PDF+PNG for 2D; PNG only if 3D)."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

from cryoem_mrc.repo_paths import COHORT_MANIFEST, find_features_npz, halfmap_metrics_npz, lh_map_reliability_dir

REPO = Path(__file__).resolve().parents[1]
PY = REPO / ".venv" / "bin" / "python"
SKIP_SOURCES = frozenset({"excluded", "optional"})

CONFORMATION_PAIRS = [
    ("23129", "23130"),
    ("49450", "48534"),
    ("49450", "48923"),
]


def run(cmd: list[str], label: str) -> int:
    print(f"\n{'=' * 60}\n[figures] {label}\n{'=' * 60}", flush=True)
    rc = subprocess.run(cmd, cwd=REPO).returncode
    if rc != 0:
        print(f"[figures] FAILED ({rc}): {label}", file=sys.stderr, flush=True)
    return rc


def manifest_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with COHORT_MANIFEST.open(newline="") as f:
        for row in csv.DictReader(f):
            eid = str(row.get("emdb_id", "")).strip()
            if not eid:
                continue
            if row.get("flexibility_source", "").strip() in SKIP_SOURCES:
                continue
            if row.get("contour", "").strip().upper() == "TBD":
                continue
            rows.append(row)
    return rows


def main() -> int:
    if not PY.is_file():
        print("Missing .venv/bin/python — run: uv pip install -e .", file=sys.stderr)
        return 2

    rc = 0

    rc = max(rc, run([str(PY), "scripts/run_cohort_summary_figures.py"], "cohort summary heatmap"))

    rc = max(
        rc,
        run([str(PY), "scripts/run_thesis_overview_figures.py"], "thesis overview (EMD-49450)"),
    )

    for row in manifest_rows():
        eid = str(row["emdb_id"]).strip()
        lh_dir = lh_map_reliability_dir(eid)
        if not (lh_dir / "reliability.npz").is_file():
            print(f"[figures] skip lh_export EMD-{eid}: no reliability.npz", flush=True)
            continue
        ref = Path(row["reference_mrc"])
        contour = float(row["contour"])
        features = find_features_npz(ref.parent, eid, contour)
        if features is None:
            print(f"[figures] skip lh_export EMD-{eid}: no features NPZ", flush=True)
            continue
        metrics = halfmap_metrics_npz(eid)
        if not metrics.is_file():
            print(f"[figures] skip lh_export EMD-{eid}: no halfmap_metrics.npz", flush=True)
            continue
        rc = max(
            rc,
            run(
                [
                    str(PY),
                    "scripts/run_lh_map_reliability_export.py",
                    "--data-dir",
                    str(ref.parent),
                    "--emd-id",
                    eid,
                    "--contour",
                    str(row["contour"]),
                    "--features",
                    str(features),
                    "--halfmap-npz",
                    str(metrics),
                    "--out-dir",
                    str(lh_dir),
                ],
                f"lh_map_reliability EMD-{eid}",
            ),
        )

    rc = max(
        rc,
        run([str(PY), "scripts/run_residue_bfactor_validation.py", "--all"], "B-factor validation (--all)"),
    )

    for emd_a, emd_b in CONFORMATION_PAIRS:
        rc = max(
            rc,
            run(
                [
                    str(PY),
                    "scripts/run_residue_bfactor_conformation_pair.py",
                    "--emd-a",
                    emd_a,
                    "--emd-b",
                    emd_b,
                ],
                f"conformation pair EMD-{emd_a} vs {emd_b}",
            ),
        )

    print(f"\n[figures] batch finished (max exit code {rc})", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
