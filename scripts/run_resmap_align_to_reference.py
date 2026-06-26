"""Align ResMap local-resolution maps onto each cohort entry's deposited reference grid.

Reads raw ``outputs/emd_<ID>/resmap/resmap.mrc`` and writes reference-aligned
``outputs/emd_<ID>/locres_resmap.mrc`` (same role as ``locres_blocres.mrc``).

Resamples in physical Å coordinates when shape/voxel differ from the deposited map.

Example::

    source .venv/bin/activate
    python scripts/run_resmap_align_to_reference.py --all
    python scripts/run_resmap_align_to_reference.py --emd-id 52525
    python scripts/run_resmap_align_to_reference.py --all --force
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from cryoem_mrc.local_resolution import (
    RESMAP_UNRESOLVED_SENTINEL_A,
    align_locres_to_reference,
    locres_resmap_path,
    locres_resmap_raw_path,
)
from cryoem_mrc.repo_paths import COHORT_MANIFEST


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, default=COHORT_MANIFEST)
    p.add_argument("--emd-id", type=str, default=None)
    p.add_argument("--all", action="store_true")
    p.add_argument("--force", action="store_true", help="Overwrite existing locres_resmap.mrc")
    return p.parse_args(argv)


def _rows(manifest: Path, emd_id: str | None) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    with manifest.open(newline="") as f:
        for row in csv.DictReader(f):
            eid = str(row.get("emdb_id", "")).strip()
            if not eid:
                continue
            if emd_id and eid != emd_id.strip():
                continue
            out.append(row)
    return out


def _align_one(row: dict[str, str], *, force: bool) -> int:
    emdb_id = str(row["emdb_id"]).strip()
    reference = Path(row["reference_mrc"])
    raw = locres_resmap_raw_path(emdb_id)
    out = locres_resmap_path(emdb_id)

    if not reference.is_file():
        print(f"[resmap_align] skip EMD-{emdb_id}: missing reference {reference}", flush=True)
        return 1
    if not raw.is_file():
        print(f"[resmap_align] skip EMD-{emdb_id}: missing raw {raw}", flush=True)
        return 0
    if out.is_file() and not force and out.stat().st_mtime >= raw.stat().st_mtime:
        print(f"[resmap_align] skip EMD-{emdb_id}: up to date {out}", flush=True)
        return 0

    notes = align_locres_to_reference(
        reference,
        raw,
        out,
        extra_label="ResMap local resolution (aligned to reference)",
        resample_cval=RESMAP_UNRESOLVED_SENTINEL_A,
    )
    print(f"[resmap_align] EMD-{emdb_id}: {'; '.join(notes)} -> {out}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.all and not args.emd_id:
        print("Specify --emd-id or --all", file=sys.stderr)
        return 2

    rc = 0
    for row in _rows(args.manifest, args.emd_id):
        rc = max(rc, _align_one(row, force=args.force))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
