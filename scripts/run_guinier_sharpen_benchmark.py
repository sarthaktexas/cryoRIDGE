"""Guinier B / sharpening benchmark: global vs local-Rmax bands vs deposited primary.

Estimates sharpening B-factors on avg-of-halves (unsharpened), compares:
  1. Whole-map Guinier B (scalar) -> global B sharpening
  2. Local Guinier B with R_max = global resolution
  3. Local Guinier B with R_max = BlocRes local resolution (when available)

Sharpening CCC metrics compare **map sharpening** outcomes to the depositor
primary map — not deposited atomic B-factors (those are reported separately
as Spearman rho at Cα when PDB B-factors exist).

Example::

    source .venv/bin/activate
    python scripts/run_guinier_sharpen_benchmark.py --anchors
    python scripts/run_guinier_sharpen_benchmark.py --emd-id 11638
    python scripts/run_guinier_sharpen_benchmark.py --all-b-factor
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from cryoem_mrc.guinier_benchmark import (
    GuinierBenchmarkResult,
    plot_guinier_benchmark_summary,
    result_to_dict,
    run_guinier_benchmark_one,
)
from cryoem_mrc.repo_paths import ANCHOR_EMDB_ID, BFACTOR_VALIDATION_EMDB_IDS, COHORT_MANIFEST, OUTPUTS_ROOT

REPO = Path(__file__).resolve().parents[1]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, default=COHORT_MANIFEST)
    p.add_argument("--out-dir", type=Path, default=OUTPUTS_ROOT / "cohort_summary")
    p.add_argument("--emd-id", action="append", default=[], metavar="ID")
    p.add_argument("--anchors", action="store_true", help=f"Run {ANCHOR_EMDB_ID} + b_factor validation anchors")
    p.add_argument("--all-b-factor", action="store_true", help="All manifest rows with flexibility_source=b_factor")
    p.add_argument("--patch-size", type=int, default=17)
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--skip-local-sharpen", action="store_true", help="Skip expensive local sharpen (B maps only)")
    p.add_argument("--locbfactor-mrc", type=Path, help="Optional external LocBFactor MRC (single-map runs)")
    p.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate guinier_sharpen_benchmark.png from existing CSV in --out-dir",
    )
    return p.parse_args(argv)


def _select_ids(args: argparse.Namespace) -> list[str]:
    if args.emd_id:
        return [str(x).strip() for x in args.emd_id]
    if args.anchors:
        ids = [ANCHOR_EMDB_ID, *BFACTOR_VALIDATION_EMDB_IDS]
        return list(dict.fromkeys(ids))
    if args.all_b_factor:
        ids: list[str] = []
        with args.manifest.open(newline="") as f:
            for row in csv.DictReader(f):
                if row.get("flexibility_source", "").strip() == "b_factor":
                    ids.append(str(row["emdb_id"]).strip())
        return ids
    return [ANCHOR_EMDB_ID]


def _fcsv(val: str) -> float:
    val = str(val).strip()
    if not val:
        return float("nan")
    return float(val)


def _rows_from_csv(csv_path: Path) -> list[GuinierBenchmarkResult]:
    with csv_path.open(newline="") as f:
        dicts = list(csv.DictReader(f))
    rows: list[GuinierBenchmarkResult] = []
    for d in dicts:
        rows.append(
            GuinierBenchmarkResult(
                emdb_id=str(d["emdb_id"]),
                global_resolution_a=_fcsv(d["global_resolution_a"]),
                b_global_guinier=_fcsv(d["b_global_guinier"]),
                b_global_r_squared=_fcsv(d["b_global_r_squared"]),
                local_b_median_global_rmax=_fcsv(d["local_b_median_global_rmax"]),
                local_b_iqr_global_rmax=_fcsv(d["local_b_iqr_global_rmax"]),
                local_b_median_locres_rmax=_fcsv(d["local_b_median_locres_rmax"]),
                local_b_iqr_locres_rmax=_fcsv(d["local_b_iqr_locres_rmax"]),
                delta_median_global_rmax=_fcsv(d["delta_median_global_rmax"]),
                delta_median_locres_rmax=_fcsv(d["delta_median_locres_rmax"]),
                ccc_sharp_global_vs_deposit=_fcsv(d["ccc_sharp_global_vs_deposit"]),
                ccc_sharp_local_global_rmax_vs_deposit=_fcsv(d["ccc_sharp_local_global_rmax_vs_deposit"]),
                ccc_sharp_local_locres_rmax_vs_deposit=_fcsv(d["ccc_sharp_local_locres_rmax_vs_deposit"]),
                ccc_sharp_local_global_rmax_vs_global=_fcsv(d["ccc_sharp_local_global_rmax_vs_global"]),
                ccc_sharp_local_locres_rmax_vs_global=_fcsv(d["ccc_sharp_local_locres_rmax_vs_global"]),
                rho_biso_vs_local_b_global_rmax=_fcsv(d["rho_biso_vs_local_b_global_rmax"]),
                rho_biso_vs_local_b_locres_rmax=_fcsv(d["rho_biso_vs_local_b_locres_rmax"]),
                n_in_mask_ca=int(float(d["n_in_mask_ca"] or 0)),
                has_locres=str(d.get("has_locres", "")).lower() in {"true", "1", "yes"},
                notes=str(d.get("notes", "")),
            )
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.plot_only:
        csv_path = args.out_dir / "guinier_sharpen_benchmark.csv"
        if not csv_path.is_file():
            print(f"[guinier_bench] missing {csv_path}", file=sys.stderr)
            return 2
        rows = _rows_from_csv(csv_path)
        fig_path = args.out_dir / "guinier_sharpen_benchmark.png"
        plot_guinier_benchmark_summary(rows, fig_path)
        from cryoem_mrc.repo_paths import sync_thesis_doc_figure

        sync_thesis_doc_figure(fig_path, "fig_3_8_guinier_sharpen_benchmark.png")
        print(f"[guinier_bench] figure → {fig_path}", flush=True)
        return 0

    ids = _select_ids(args)
    if not ids:
        print("[guinier_bench] no EMDB ids", file=sys.stderr)
        return 2

    rows: list[GuinierBenchmarkResult] = []
    for eid in ids:
        ext = args.locbfactor_mrc if len(ids) == 1 else None
        print(f"[guinier_bench] EMD-{eid} ...", flush=True)
        result = run_guinier_benchmark_one(
            eid,
            manifest=args.manifest,
            patch_size=args.patch_size,
            stride=args.stride,
            external_locbfactor_mrc=ext,
            skip_local_sharpen=args.skip_local_sharpen,
        )
        if result is None:
            print(f"[guinier_bench] skip EMD-{eid}", file=sys.stderr, flush=True)
            continue
        rows.append(result)
        print(
            f"[guinier_bench] EMD-{eid}: B_global={result.b_global_guinier:.1f} "
            f"med_local(g)={result.local_b_median_global_rmax:.1f} "
            f"CCC_global→dep={result.ccc_sharp_global_vs_deposit:.3f} "
            f"CCC_local(g)→dep={result.ccc_sharp_local_global_rmax_vs_deposit:.3f}",
            flush=True,
        )

    if not rows:
        print("[guinier_bench] no results", file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "guinier_sharpen_benchmark.csv"
    json_path = args.out_dir / "guinier_sharpen_benchmark.json"
    dicts = [result_to_dict(r) for r in rows]
    fieldnames = list(dicts[0].keys())
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for d in dicts:
            w.writerow(
                {
                    k: (
                        f"{v:.6f}"
                        if isinstance(v, float) and np.isfinite(v)
                        else ("" if isinstance(v, float) and not np.isfinite(v) else v)
                    )
                    for k, v in d.items()
                }
            )
    json_path.write_text(json.dumps(dicts, indent=2) + "\n")
    fig_path = args.out_dir / "guinier_sharpen_benchmark.png"
    plot_guinier_benchmark_summary(rows, fig_path)
    from cryoem_mrc.repo_paths import sync_thesis_doc_figure

    sync_thesis_doc_figure(fig_path, "fig_3_8_guinier_sharpen_benchmark.png")
    print(f"[guinier_bench] {len(rows)} maps → {csv_path}", flush=True)
    print(f"[guinier_bench] figure → {fig_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
