"""Align MonoRes Chimera exports onto deposited reference maps.

Reads ``data/emd_<ID>*/monores/monoresResolutionChimera.mrc`` and writes
``outputs/emd_<ID>/locres_monores.mrc``.

Example::

    source .venv/bin/activate
    python scripts/run_monores_align_to_reference.py --all
    python scripts/run_monores_align_to_reference.py --emd-id 23129
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from cryoem_mrc.local_resolution import ensure_locres_monores_aligned, find_monores_chimera_mrc
from cryoem_mrc.repo_paths import COHORT_MANIFEST


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--emd-id", type=str, default=None)
    p.add_argument("--all", action="store_true")
    p.add_argument("--manifest", type=Path, default=COHORT_MANIFEST)
    p.add_argument("--force", action="store_true")
    return p.parse_args(argv)


def _manifest_ids(manifest: Path) -> list[str]:
    ids: list[str] = []
    with manifest.open(newline="") as f:
        for row in csv.DictReader(f):
            eid = str(row.get("emdb_id", "")).strip()
            if eid:
                ids.append(eid)
    return ids


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.all and not args.emd_id:
        print("Specify --emd-id or --all", file=sys.stderr)
        return 2

    ids = _manifest_ids(args.manifest) if args.all else [args.emd_id.strip()]
    rc = 0
    aligned = 0
    for eid in ids:
        if find_monores_chimera_mrc(eid) is None:
            continue
        ref = None
        with args.manifest.open(newline="") as f:
            for row in csv.DictReader(f):
                if str(row.get("emdb_id", "")).strip() == eid:
                    ref = Path(row["reference_mrc"])
                    break
        if ref is None or not ref.is_file():
            print(f"[monores_align] skip EMD-{eid}: missing reference", flush=True)
            rc = 1
            continue
        out = ensure_locres_monores_aligned(eid, reference=ref, force=args.force)
        if out is None:
            print(f"[monores_align] FAIL EMD-{eid}", file=sys.stderr, flush=True)
            rc = 1
            continue
        aligned += 1
        print(f"[monores_align] EMD-{eid} -> {out}", flush=True)
    print(f"[monores_align] aligned {aligned} map(s)", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
