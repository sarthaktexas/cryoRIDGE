"""Export per-residue cross-metric tables via ``thesis.metric_comparison.load_all_metrics``.

Writes under ``outputs/emd_<ID>/metric_comparison/``:

- ``residue_metrics.csv``
- ``cross_metric_correlations.csv``

Example::

    source .venv/bin/activate
    python scripts/run_metric_comparison_export.py --emd-id 49450
    python scripts/run_metric_comparison_export.py --all
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from cryoem_mrc.repo_paths import COHORT_MANIFEST, emd_output_dir
from cryoem_mrc.manifest_policy import row_ca_metrics_eligible, row_resmap_ca_headline_eligible
from thesis.metric_comparison import (
    LocresSource,
    compute_cross_metric_correlations,
    load_all_metrics,
    metric_comparison_dirname,
)
from thesis.reliability_volumes import reliability_mrc_paths


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--emd-id", type=str, default=None)
    p.add_argument("--all", action="store_true")
    p.add_argument("--manifest", type=Path, default=COHORT_MANIFEST)
    p.add_argument(
        "--locres-source",
        choices=("blocres", "resmap", "monores", "all"),
        default="blocres",
        help="Local-resolution map(s) to aggregate at Cα (default: blocres)",
    )
    return p.parse_args(argv)


def _eligible_ids(manifest: Path, *, locres_source: LocresSource) -> list[str]:
    ids: list[str] = []
    with manifest.open(newline="") as f:
        for row in csv.DictReader(f):
            eid = str(row["emdb_id"]).strip()
            if locres_source == "resmap":
                if not row_resmap_ca_headline_eligible(row):
                    reason = "resmap expected failure" if row.get("resmap_expected_failure") else "map-only / no PDB"
                    print(f"[metric_export] skip EMD-{eid} ResMap Cα: {reason}", flush=True)
                    continue
            elif not row_ca_metrics_eligible(row):
                print(f"[metric_export] skip EMD-{eid}: map-only / no deposited PDB", flush=True)
                continue
            rel_mrc, zone_mrc = reliability_mrc_paths(eid)
            if not rel_mrc.is_file() or not zone_mrc.is_file():
                print(f"[metric_export] skip EMD-{eid}: missing reliability MRCs", flush=True)
                continue
            ids.append(eid)
    return ids


def _export_one(emdb_id: str, *, manifest: Path, locres_source: LocresSource) -> int:
    try:
        df = load_all_metrics(emdb_id, manifest=manifest, locres_source=locres_source)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"[metric_export] FAIL EMD-{emdb_id}: {exc}", file=sys.stderr, flush=True)
        return 1

    out_dir = emd_output_dir(emdb_id) / metric_comparison_dirname(locres_source)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "residue_metrics.csv"
    df.to_csv(metrics_path, index=False)

    corr = compute_cross_metric_correlations(df)
    corr_path = out_dir / "cross_metric_correlations.csv"
    corr.to_csv(corr_path)

    n_loc = int(df["local_resolution"].notna().sum()) if "local_resolution" in df.columns else 0
    print(
        f"[metric_export] EMD-{emdb_id} ({locres_source}): {len(df)} residues, "
        f"local_resolution finite={n_loc} -> {metrics_path}",
        flush=True,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.all and not args.emd_id:
        print("Specify --emd-id or --all", file=sys.stderr)
        return 2

    if args.locres_source == "all":
        sources: tuple[LocresSource, ...] = ("blocres", "resmap", "monores")
    elif args.locres_source == "both":
        sources = ("blocres", "resmap")
    else:
        sources = (args.locres_source,)  # type: ignore[assignment]

    if args.all:
        union: set[str] = set()
        for locres_source in sources:
            union.update(_eligible_ids(args.manifest, locres_source=locres_source))
        target_ids = sorted(union)
    else:
        target_ids = [args.emd_id.strip()]

    rc = 0
    for emdb_id in target_ids:
        for locres_source in sources:
            if emdb_id not in _eligible_ids(args.manifest, locres_source=locres_source):
                continue
            rc = max(rc, _export_one(emdb_id, manifest=args.manifest, locres_source=locres_source))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
