"""Run FSC-Q validation on all PDB-backed cohort maps missing fscq_stats.json.

Resumable: skips maps that already have outputs/emd_<ID>/fscq/fscq_stats.json.
Rebuilds outputs/cohort_summary/fscq_correlations.csv and the cohort figure at the end.

Example::

    uv run python scripts/run_fscq_cohort_batch.py
    uv run python scripts/run_fscq_cohort_batch.py --emd-id 62841
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from cryoem_mrc.repo_paths import COHORT_MANIFEST, OUTPUTS_ROOT, emd_output_dir

SCRIPT = Path(__file__).resolve().parent / "run_fscq_validation.py"


def _eligible_ids(manifest: Path) -> list[str]:
    ids: list[str] = []
    with manifest.open(newline="") as f:
        for row in csv.DictReader(f):
            eid = str(row.get("emdb_id", "")).strip()
            if not eid:
                continue
            pdb = (row.get("flexibility_path_or_pdb") or "").strip()
            src = (row.get("flexibility_source") or "").strip()
            notes = row.get("notes", "")
            if src == "excluded" or "SKIP" in notes or not pdb:
                continue
            ref = Path(row["reference_mrc"])
            h1 = Path(row["half1_path"])
            h2 = Path(row["half2_path"])
            if not all(p.is_file() for p in (ref, h1, h2, Path(pdb))):
                continue
            ids.append(eid)
    return ids


def _pending_ids(manifest: Path, *, only: list[str] | None = None) -> list[str]:
    pool = only if only is not None else _eligible_ids(manifest)
    pending: list[str] = []
    for eid in pool:
        stats = emd_output_dir(eid) / "fscq" / "fscq_stats.json"
        if not stats.is_file():
            pending.append(eid)
    return pending


def _rebuild_cohort_summary() -> int:
    rows: list[dict] = []
    for path in sorted(OUTPUTS_ROOT.glob("emd_*/fscq/fscq_stats.json")):
        rows.append(json.loads(path.read_text()))
    if not rows:
        print("[fscq_batch] no fscq_stats.json files found", file=sys.stderr)
        return 1
    out = OUTPUTS_ROOT / "cohort_summary" / "fscq_correlations.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[fscq_batch] cohort summary → {out} ({len(rows)} maps)", flush=True)
    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, default=COHORT_MANIFEST)
    p.add_argument("--emd-id", action="append", default=[], help="Run one map (repeatable)")
    p.add_argument("--skip-figure", action="store_true")
    p.add_argument("--force", action="store_true", help="Pass --force to run_fscq_validation.py")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    only = args.emd_id or None
    pending = _pending_ids(args.manifest, only=only)
    if not pending:
        print("[fscq_batch] nothing pending", flush=True)
        return _rebuild_cohort_summary()

    ok = fail = 0
    for eid in pending:
        cmd = [sys.executable, str(SCRIPT), "--emd-id", eid]
        if args.force:
            cmd.append("--force")
        print(f"[fscq_batch] EMD-{eid} …", flush=True)
        proc = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1])
        if proc.returncode == 0:
            ok += 1
        else:
            fail += 1
            print(f"[fscq_batch] EMD-{eid} failed (exit {proc.returncode})", file=sys.stderr, flush=True)

    print(f"[fscq_batch] done: ok={ok} fail={fail} pending_was={len(pending)}", flush=True)
    rc = _rebuild_cohort_summary()
    if not args.skip_figure:
        fig = subprocess.run(
            [sys.executable, str(SCRIPT), "--cohort-figure"],
            cwd=Path(__file__).resolve().parents[1],
        )
        rc = max(rc, fig.returncode)
    return rc if fail == 0 else max(rc, 1)


if __name__ == "__main__":
    raise SystemExit(main())
