"""Run Phenix EMRinger for cohort maps missing flat per-residue CSV exports.

Expects ``phenix.emringer`` on PATH or ``PHENIX_BIN`` pointing at the binary.
After each run, normalizes outputs under ``outputs/emringer_flat/`` as::

    {pdb_code}_emringer.csv
    {pdb_code}_emringer.pkl

The chi-scan CSV is the residue-level export used by ``cryoem_mrc.emringer``;
export it from the Phenix EMRinger GUI if the CLI run only writes the ``.pkl``.

Example::

    source .venv/bin/activate
    python scripts/run_emringer_cohort_batch.py --audit
    python scripts/run_emringer_cohort_batch.py
    python scripts/run_emringer_cohort_batch.py --emd-id 6287
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from cryoem_mrc.emringer import (
    EmringerDepositRow,
    iter_emringer_deposits,
    missing_emringer_csvs,
    pdb_code_from_flexibility_path,
)
from cryoem_mrc.manifest_policy import row_qscore_eligible
from cryoem_mrc.repo_paths import COHORT_MANIFEST, EMRINGER_FLAT_DIR

PHENIX_BIN = Path(os.environ.get("PHENIX_BIN", "phenix.emringer"))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--manifest", type=Path, default=COHORT_MANIFEST)
    p.add_argument("--emringer-dir", type=Path, default=EMRINGER_FLAT_DIR)
    p.add_argument("--emd-id", action="append", default=[], help="Run one map (repeatable)")
    p.add_argument(
        "--audit",
        action="store_true",
        help="Print coverage only; do not run Phenix",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-run even when the flat CSV already exists",
    )
    return p.parse_args(argv)


def _pending(
    manifest: Path,
    flat_dir: Path,
    *,
    only: list[str] | None = None,
    force: bool = False,
) -> list[EmringerDepositRow]:
    eligible: set[str] = set()
    with manifest.open(newline="") as f:
        for row in csv.DictReader(f):
            if row_qscore_eligible(row):
                eligible.add(str(row["emdb_id"]).strip())

    rows = iter_emringer_deposits(manifest, flat_dir)
    rows = [r for r in rows if r.emdb_id in eligible]
    if only:
        want = {e.strip() for e in only}
        rows = [r for r in rows if r.emdb_id in want]
    if force:
        return rows
    return [r for r in rows if not r.has_csv]


def _normalize_outputs(row: EmringerDepositRow, work_dir: Path) -> bool:
    """Move Phenix outputs into flat ``{pdb_code}_emringer.*`` names."""
    flat_dir = row.csv_path.parent
    flat_dir.mkdir(parents=True, exist_ok=True)
    code = row.pdb_code

    pkls = sorted(work_dir.glob("*emringer*.pkl"))
    if pkls:
        dest_pkl = flat_dir / f"{code}_emringer.pkl"
        shutil.copy2(pkls[0], dest_pkl)

    if row.csv_path.is_file():
        return True

    csvs = sorted(work_dir.glob("*emringer*.csv"))
    if csvs:
        shutil.copy2(csvs[0], row.csv_path)
        return True

    # Phenix sometimes names after model basename.
    stem = row.pdb_path.stem.lower()
    for candidate in (
        flat_dir / f"{stem}_emringer.csv",
        work_dir / f"{stem}_emringer.csv",
        work_dir / f"{row.pdb_path.name}_emringer.csv",
    ):
        if candidate.is_file():
            shutil.copy2(candidate, row.csv_path)
            return True
    return False


def _run_one(row: EmringerDepositRow, *, flat_dir: Path) -> int:
    if not row.reference_mrc.is_file():
        print(f"[emringer_batch] skip EMD-{row.emdb_id}: missing {row.reference_mrc}", flush=True)
        return 1

    work_dir = flat_dir / f"{row.pdb_code}_emringer_run"
    work_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(PHENIX_BIN),
        str(row.pdb_path.resolve()),
        str(row.reference_mrc.resolve()),
    ]
    print(
        f"[emringer_batch] EMD-{row.emdb_id} ({row.pdb_code}): {' '.join(cmd)}",
        flush=True,
    )
    proc = subprocess.run(cmd, cwd=work_dir)
    if proc.returncode != 0:
        print(f"[emringer_batch] FAIL EMD-{row.emdb_id}: phenix exit {proc.returncode}", flush=True)
        return proc.returncode

    if _normalize_outputs(row, work_dir):
        print(f"[emringer_batch] OK EMD-{row.emdb_id} -> {row.csv_path}", flush=True)
        return 0

    print(
        f"[emringer_batch] WARN EMD-{row.emdb_id}: phenix finished but no chi CSV found. "
        f"Export residue scan CSV from Phenix GUI to {row.csv_path}",
        flush=True,
    )
    return 2


def _print_audit(manifest: Path, flat_dir: Path) -> int:
    all_rows = iter_emringer_deposits(manifest, flat_dir)
    missing = missing_emringer_csvs(manifest, flat_dir)
    print(
        f"[emringer_batch] coverage: {len(all_rows) - len(missing)}/{len(all_rows)} "
        f"deposited structures have {flat_dir}/{{pdb}}_emringer.csv",
        flush=True,
    )
    if missing:
        print("[emringer_batch] missing:", flush=True)
        for row in missing:
            print(
                f"  EMD-{row.emdb_id} {row.pdb_code} -> {row.csv_path.name}",
                flush=True,
            )
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    args.emringer_dir.mkdir(parents=True, exist_ok=True)

    if args.audit:
        return _print_audit(args.manifest, args.emringer_dir)

    only = args.emd_id or None
    pending = _pending(
        args.manifest, args.emringer_dir, only=only, force=args.force
    )
    if not pending:
        print("[emringer_batch] nothing pending", flush=True)
        return _print_audit(args.manifest, args.emringer_dir)

    if shutil.which(str(PHENIX_BIN)) is None and not Path(PHENIX_BIN).is_file():
        print(
            f"[emringer_batch] {PHENIX_BIN} not found; set PHENIX_BIN or install Phenix",
            file=sys.stderr,
            flush=True,
        )
        _print_audit(args.manifest, args.emringer_dir)
        return 127

    rc = 0
    for row in pending:
        rc = max(rc, _run_one(row, flat_dir=args.emringer_dir))
    return max(rc, _print_audit(args.manifest, args.emringer_dir))


if __name__ == "__main__":
    raise SystemExit(main())
