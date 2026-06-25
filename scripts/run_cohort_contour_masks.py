"""Write depositor-contour binary masks next to each cohort reference map.

For every manifest row with a valid ``contour`` and on-disk ``reference_mrc``,
writes ``contour_mask.mrc`` in the same ``data/emd_*`` folder (0/1 float32 MRC,
1 = density >= contour). Intended for BlocRes, ResMap, and other local-resolution
tools that expect a mask on the deposited primary grid.

Example::

    source .venv/bin/activate
    python scripts/run_cohort_contour_masks.py --all
    python scripts/run_cohort_contour_masks.py --emd-id 49450
    python scripts/run_cohort_contour_masks.py --all --dry-run
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from cryoem_mrc.repo_paths import COHORT_MANIFEST
from scripts.run_blocres_local_resolution import _parse_contour, _write_contour_mask

SKIP_SOURCES = frozenset({"excluded", "optional"})
MASK_NAME = "contour_mask.mrc"


def _mask_path(reference_mrc: Path) -> Path:
    return reference_mrc.parent / MASK_NAME


def _manifest_rows(
    manifest: Path,
    *,
    emd_id: str | None,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with manifest.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("flexibility_source", "").strip() in SKIP_SOURCES:
                continue
            eid = str(row["emdb_id"]).strip()
            if emd_id is not None and eid != emd_id.strip():
                continue
            rows.append(row)
    return rows


def _run_one(
    row: dict[str, str],
    *,
    force: bool,
    dry_run: bool,
) -> int:
    emdb_id = str(row["emdb_id"]).strip()
    reference = Path(row["reference_mrc"])
    out_path = _mask_path(reference)

    if not reference.is_file():
        print(f"[contour_mask] skip EMD-{emdb_id}: missing reference {reference}", flush=True)
        return 1

    try:
        contour = _parse_contour(row, override=None)
    except ValueError as exc:
        print(f"[contour_mask] skip EMD-{emdb_id}: {exc}", flush=True)
        return 0

    if out_path.is_file() and not force:
        print(f"[contour_mask] skip EMD-{emdb_id}: exists {out_path}", flush=True)
        return 0

    if dry_run:
        print(
            f"[contour_mask] dry-run EMD-{emdb_id}: contour={contour:g} -> {out_path}",
            flush=True,
        )
        return 0

    t0 = time.perf_counter()
    n_in = _write_contour_mask(reference, contour, out_path)
    dt = time.perf_counter() - t0
    print(
        f"[contour_mask] EMD-{emdb_id}: contour={contour:g} "
        f"voxels={n_in} ({dt:.1f}s) -> {out_path}",
        flush=True,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--emd-id", type=str, default=None, help="Single EMDB ID (e.g. 49450)")
    p.add_argument("--all", action="store_true", help="Process all non-excluded manifest rows")
    p.add_argument("--manifest", type=Path, default=COHORT_MANIFEST)
    p.add_argument("--force", action="store_true", help="Overwrite existing contour_mask.mrc")
    p.add_argument("--dry-run", action="store_true", help="Print planned writes only")
    args = p.parse_args(argv)

    if not args.all and args.emd_id is None:
        p.error("specify --emd-id or --all")

    rows = _manifest_rows(args.manifest, emd_id=args.emd_id)
    if not rows:
        print("[contour_mask] no matching manifest rows", flush=True)
        return 1

    rc = 0
    t0 = time.perf_counter()
    for row in rows:
        rc = max(rc, _run_one(row, force=args.force, dry_run=args.dry_run))
    if len(rows) > 1 and not args.dry_run:
        print(f"[contour_mask] finished {len(rows)} entries in {time.perf_counter() - t0:.1f}s", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
