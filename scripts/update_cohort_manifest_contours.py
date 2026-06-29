"""Fetch EMDB depositor contour levels and write them into a cohort manifest CSV.

Optionally sync the same values into a companion JSON download manifest
(``entries`` and ``pairs[].states``).

Example::

    source .venv/bin/activate
    python scripts/update_cohort_manifest_contours.py \\
        --manifest cohort/expansion_manifest.csv \\
        --json cohort/expansion_manifest.json
    python scripts/update_cohort_manifest_contours.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cryoem_mrc.cohort_emdb import fetch_emdb_recommended_contour
from cryoem_mrc.repo_paths import COHORT_MANIFEST


def _format_contour(level: float) -> str:
    return f"{level:.10g}"


def _apply_contours_to_json(json_path: Path, contours: dict[str, str]) -> int:
    data = json.loads(json_path.read_text())
    updated = 0
    for entry in data.get("entries", []):
        eid = str(entry.get("emdb_id", "")).strip()
        if eid in contours and str(entry.get("contour")) != contours[eid]:
            entry["contour"] = float(contours[eid])
            updated += 1
    for pair in data.get("pairs", []):
        for state in pair.get("states", []):
            eid = str(state.get("emdb_id", "")).strip()
            if eid in contours and str(state.get("contour")) != contours[eid]:
                state["contour"] = float(contours[eid])
                updated += 1
    json_path.write_text(json.dumps(data, indent=2) + "\n")
    return updated


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, default=COHORT_MANIFEST)
    p.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Optional companion JSON manifest to update in parallel",
    )
    p.add_argument("--dry-run", action="store_true", help="Print contours without writing files")
    p.add_argument("--delay-s", type=float, default=0.15, help="Pause between EMDB API calls")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    manifest = args.manifest
    if not manifest.is_file():
        print(f"[update_contours] missing {manifest}", file=sys.stderr)
        return 2

    with manifest.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"[update_contours] empty {manifest}", file=sys.stderr)
        return 2

    fieldnames = list(rows[0].keys())
    if "contour" not in fieldnames:
        fieldnames.append("contour")

    contours: dict[str, str] = {}
    updated = 0
    missing: list[str] = []

    for row in rows:
        eid = str(row.get("emdb_id", "")).strip()
        if not eid:
            continue
        try:
            level = fetch_emdb_recommended_contour(eid)
        except RuntimeError as exc:
            print(f"[update_contours] EMD-{eid}: {exc}", file=sys.stderr)
            level = None
        if level is None:
            missing.append(eid)
            print(f"[update_contours] EMD-{eid}: no contour on EMDB", file=sys.stderr)
            continue
        new_val = _format_contour(level)
        old_val = str(row.get("contour", "")).strip()
        contours[eid] = new_val
        if old_val != new_val:
            updated += 1
        row["contour"] = new_val
        print(f"EMD-{eid}: {new_val}")
        time.sleep(args.delay_s)

    if missing:
        print(f"[update_contours] missing contour for {len(missing)} entries", file=sys.stderr)

    if args.dry_run:
        print(f"[update_contours] dry run: would update {updated} rows")
        return 0 if not missing else 1

    with manifest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"[update_contours] wrote {manifest} ({updated} values changed)")

    if args.json is not None:
        if not args.json.is_file():
            print(f"[update_contours] missing JSON {args.json}", file=sys.stderr)
            return 2
        n_json = _apply_contours_to_json(args.json, contours)
        print(f"[update_contours] wrote {args.json} ({n_json} values changed)")

    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
