"""Cohort audit: deposited Cα in low-CC regions vs tercile omit zones.

Reads ``residue_validation.csv`` per map (or runs validation when missing) and
writes ``outputs/cohort_summary/model_placement_audit.csv`` plus a scatter
comparing absolute half-map CC cutoffs to tercile-based omit fractions.

Example::

    source .venv/bin/activate
    python scripts/run_cohort_model_placement_audit.py --all
    python scripts/run_cohort_model_placement_audit.py --emd-id 11638 --cc-threshold 0.5
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from style.nature import PALETTES, apply, savefig as save_nature

from cryoem_mrc.repo_paths import COHORT_MANIFEST, OUTPUTS_ROOT
from cryoem_mrc.structure_validation import (
    compute_model_placement_audit_stats,
    default_reliability_out_dir,
    load_cohort_manifest_row,
    read_residue_validation_csv,
    run_emdb_bfactor_validation,
    write_model_placement_audit_csv,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--emd-id", type=str, default=None, help="Single EMDB ID (e.g. 11638)")
    p.add_argument("--all", action="store_true", help="All manifest rows with a local deposited model")
    p.add_argument("--manifest", type=Path, default=COHORT_MANIFEST)
    p.add_argument(
        "--cc-threshold",
        type=float,
        default=0.5,
        help="Absolute local half-map CC cutoff for questionable placement (default 0.5)",
    )
    p.add_argument(
        "--run-validation",
        action="store_true",
        help="Generate missing residue_validation.csv via run_emdb_bfactor_validation",
    )
    p.add_argument("--out-dir", type=Path, default=OUTPUTS_ROOT / "cohort_summary")
    p.add_argument("--dpi", type=int, default=200)
    return p.parse_args(argv)


def _manifest_rows_with_pdb(manifest: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with manifest.open(newline="") as f:
        for row in csv.DictReader(f):
            pdb = Path(row.get("flexibility_path_or_pdb", "").strip())
            if not pdb.suffix.lower() in {".cif", ".pdb"}:
                continue
            if not pdb.exists():
                print(
                    f"[model_placement] skip EMD-{row['emdb_id']}: no model {pdb}",
                    flush=True,
                )
                continue
            rows.append(row)
    return rows


def _global_resolution(row: dict[str, str]) -> float:
    raw = row.get("global_resolution_a", "").strip()
    if not raw:
        return float("nan")
    try:
        return float(raw)
    except ValueError:
        return float("nan")


def _load_or_validate(
    emd_id: str,
    *,
    manifest: Path,
    run_validation: bool,
) -> list | None:
    out_dir = default_reliability_out_dir(emd_id)
    csv_path = out_dir / "residue_validation.csv"
    if csv_path.is_file():
        return read_residue_validation_csv(csv_path)

    if not run_validation:
        print(
            f"[model_placement] skip EMD-{emd_id}: missing {csv_path} "
            "(pass --run-validation to generate)",
            flush=True,
        )
        return None

    try:
        _, rows, stats, _ = run_emdb_bfactor_validation(
            emd_id,
            manifest=manifest,
            require_b_factor_source=False,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"[model_placement] ERROR EMD-{emd_id}: {exc}", file=sys.stderr)
        return None

    if stats is None or not rows:
        print(f"[model_placement] skip EMD-{emd_id}: validation produced no rows", flush=True)
        return None
    return rows


def _plot_tercile_vs_absolute(
    stats_rows: list,
    *,
    out_path: Path,
    cc_threshold: float,
    dpi: int,
) -> None:
    usable = [
        s
        for s in stats_rows
        if np.isfinite(s.frac_cc_below_threshold) and np.isfinite(s.frac_in_omit_zone)
    ]
    if not usable:
        print("[model_placement] no rows for scatter figure", flush=True)
        return

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    apply(ax)

    x = np.array([s.frac_in_omit_zone for s in usable], dtype=np.float64)
    y = np.array([s.frac_cc_below_threshold for s in usable], dtype=np.float64)
    colors = PALETTES["categorical"]
    ax.scatter(
        x,
        y,
        s=56,
        c=[colors[i % len(colors)] for i in range(len(usable))],
        edgecolors="white",
        linewidths=0.5,
        zorder=3,
    )

    lim_hi = max(0.35, float(np.max(np.concatenate([x, y]))) + 0.05)
    ax.plot([0, lim_hi], [0, lim_hi], color="0.55", linewidth=0.8, linestyle="--", label="y = x")
    ax.axhline(0.33, color="0.75", linewidth=0.6, linestyle=":", label="33% (tercile floor)")
    ax.axvline(0.33, color="0.75", linewidth=0.6, linestyle=":")

    for s in usable:
        label = s.display_name.split("(")[0].strip() or f"EMD-{s.emdb_id}"
        ax.annotate(
            label,
            (s.frac_in_omit_zone, s.frac_cc_below_threshold),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=6,
            alpha=0.9,
        )

    ax.set_xlim(-0.02, lim_hi)
    ax.set_ylim(-0.02, lim_hi)
    ax.set_xlabel("Deposited Cα in omit tercile (fraction)")
    ax.set_ylabel(f"Deposited Cα with local CC < {cc_threshold:.2f} (fraction)")
    ax.set_title("Tercile omit vs absolute low-CC placement")
    ax.legend(loc="upper left", frameon=False, fontsize=6)
    fig.tight_layout()
    save_nature(fig, out_path, dpi=dpi)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.all and not args.emd_id:
        print("Specify --emd-id or --all", file=sys.stderr)
        return 2

    if args.emd_id:
        manifest_row = load_cohort_manifest_row(args.manifest, args.emd_id.strip())
        manifest_rows = [manifest_row]
    else:
        manifest_rows = _manifest_rows_with_pdb(args.manifest)

    stats_rows = []
    for row in manifest_rows:
        emd_id = str(row["emdb_id"]).strip()
        rows = _load_or_validate(emd_id, manifest=args.manifest, run_validation=args.run_validation)
        if rows is None:
            continue
        stats = compute_model_placement_audit_stats(
            rows,
            emdb_id=emd_id,
            display_name=str(row.get("display_name", "")).strip(),
            global_resolution_a=_global_resolution(row),
            cc_threshold=args.cc_threshold,
        )
        stats_rows.append(stats)
        print(
            f"[model_placement] EMD-{emd_id}: omit={stats.frac_in_omit_zone:.1%}, "
            f"CC<{args.cc_threshold}={stats.frac_cc_below_threshold:.1%}, "
            f"median CC={stats.median_local_cc:.3f} (n={stats.n_in_mask})",
            flush=True,
        )

    if not stats_rows:
        print("[model_placement] no maps processed", file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "model_placement_audit.csv"
    write_model_placement_audit_csv(csv_path, stats_rows)
    _plot_tercile_vs_absolute(
        stats_rows,
        out_path=args.out_dir / "model_placement_tercile_vs_absolute.png",
        cc_threshold=args.cc_threshold,
        dpi=args.dpi,
    )
    print(f"[model_placement] wrote {csv_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
