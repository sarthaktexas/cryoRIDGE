"""Parallel per-map placement utility: BlocRes vs ResMap locres vs omit zone.

Fixed operational flag rules (in-map median or global resolution for locres;
omit build zone for reliability). Writes CSVs and AUC violin under
``outputs/cohort_summary/``.

Example::

    source .venv/bin/activate
    python scripts/run_placement_locres_lomo_comparison.py
    python scripts/run_placement_locres_lomo_comparison.py --no-resmap-qc-filter
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from style.nature import apply, savefig as save_nature
from style.thesis_palette import PALETTES

from cryoem_mrc.placement_utility import (
    LOCRES_METHOD_LOMO_LABELS,
    LOCRES_METHOD_LOMO_PREDICTORS,
    QSCORE_PANEL_EXCLUDE,
    RESMAP_QC_EXCLUDE,
    load_per_map_frames_for_locres_lomo,
    run_locres_method_lomo_validation,
    write_locres_method_lomo_csvs,
    write_locres_method_lomo_markdown,
)
from cryoem_mrc.repo_paths import COHORT_MANIFEST, OUTPUTS_ROOT

OUT_DIR = OUTPUTS_ROOT / "cohort_summary"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, default=COHORT_MANIFEST)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--q-threshold", type=float, default=0.5)
    p.add_argument("--sphere-radius-a", type=float, default=2.0)
    p.add_argument(
        "--no-resmap-qc-filter",
        action="store_true",
        help="Only run the full Q-score cohort (skip ResMap QC filtered run).",
    )
    p.add_argument(
        "--filtered-only",
        action="store_true",
        help="Only run the ResMap QC filtered cohort (26 maps).",
    )
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--no-figures", action="store_true")
    return p.parse_args(argv)


def _run_one_cohort(
    frames,
    *,
    q_threshold: float,
    exclude_emdb_ids: frozenset[str],
    out_dir: Path,
    tag: str,
    dpi: int,
    write_figure: bool,
) -> None:
    summary = run_locres_method_lomo_validation(
        frames,
        q_threshold=q_threshold,
        exclude_emdb_ids=exclude_emdb_ids,
    )
    prefix = f"placement_locres_lomo_{tag}" if tag else "placement_locres_lomo"
    cohort_dir = out_dir
    paths = write_locres_method_lomo_csvs(summary, cohort_dir, file_stem=prefix)

    md_path = cohort_dir / (
        f"PLACEMENT_LOCRES_LOMO_{tag.upper()}.md" if tag else "PLACEMENT_LOCRES_LOMO.md"
    )
    write_locres_method_lomo_markdown(summary, md_path)

    meta = {
        "q_threshold": q_threshold,
        "n_maps": len({r.held_out_emdb_id for r in summary.fold_rows}),
        "exclude_emdb_ids": sorted(summary.exclude_emdb_ids),
        "predictor_medians": summary.predictor_medians,
        "csv_paths": {k: str(v) for k, v in paths.items()},
    }
    json_path = cohort_dir / (f"{prefix}.json" if tag else "placement_locres_lomo.json")
    json_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    label = tag or "default"
    print(f"[locres_lomo:{label}] {meta['n_maps']} maps, {len(summary.fold_rows)} fold rows", flush=True)
    for k, p in paths.items():
        print(f"  {k}: {p}", flush=True)
    print(f"  markdown: {md_path}", flush=True)
    for pid in LOCRES_METHOD_LOMO_PREDICTORS:
        meds = summary.predictor_medians.get(pid, {})
        print(
            f"  {LOCRES_METHOD_LOMO_LABELS[pid]}: "
            f"median AUC={meds.get('median_auc', float('nan')):.3f}, "
            f"median BA={meds.get('median_balanced_accuracy', float('nan')):.3f}",
            flush=True,
        )

    if write_figure:
        fig_base = cohort_dir / prefix
        fig = _plot_violin(summary, cohort_dir, dpi, fig_stem=fig_base.name)
        print(f"  figure: {fig}", flush=True)


def _plot_violin(summary, out_dir: Path, dpi: int, *, fig_stem: str = "placement_locres_lomo_held_out") -> Path:
    predictors = list(LOCRES_METHOD_LOMO_PREDICTORS)
    labels = [LOCRES_METHOD_LOMO_LABELS[p] for p in predictors]
    auc_data: list[list[float]] = []
    for pid in predictors:
        auc_data.append(
            [
                r.auc
                for r in summary.fold_rows
                if r.predictor == pid and np.isfinite(r.auc)
            ]
        )

    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    apply(ax)
    colors = PALETTES["categorical"][: len(predictors)]
    parts = ax.violinplot(
        auc_data,
        positions=np.arange(len(predictors)),
        showmeans=True,
        showextrema=False,
    )
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(colors[i])
        body.set_alpha(0.75)
    ax.set_xticks(np.arange(len(predictors)))
    ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=7)
    ax.set_ylabel("Per-map AUC (low-Q classification)")
    n_maps = len({r.held_out_emdb_id for r in summary.fold_rows})
    excluded = len(summary.exclude_emdb_ids)
    ax.set_title(
        f"Fixed-rule placement utility (Q < {summary.q_threshold:.1f}, "
        f"n = {n_maps} maps, excluded = {excluded})"
    )
    ax.set_ylim(0, 1.02)
    ax.axhline(0.5, color="0.75", linewidth=0.8, linestyle="--")
    fig.tight_layout()
    out = out_dir / fig_stem
    save_nature(fig, out, dpi=dpi)
    plt.close(fig)
    return out.with_suffix(".png")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    load_exclude = QSCORE_PANEL_EXCLUDE

    frames = load_per_map_frames_for_locres_lomo(
        manifest=args.manifest,
        sphere_radius_a=args.sphere_radius_a,
        exclude=load_exclude,
    )
    if len(frames) < 3:
        print("[locres_lomo] need >= 3 maps with Q-scores", file=sys.stderr)
        return 2

    run_filtered = not args.no_resmap_qc_filter
    run_full = not args.filtered_only
    if not run_filtered and not run_full:
        print("[locres_lomo] nothing to run", file=sys.stderr)
        return 2

    if run_filtered:
        _run_one_cohort(
            frames,
            q_threshold=args.q_threshold,
            exclude_emdb_ids=RESMAP_QC_EXCLUDE,
            out_dir=args.out_dir,
            tag="filtered26",
            dpi=args.dpi,
            write_figure=not args.no_figures,
        )
    if run_full:
        _run_one_cohort(
            frames,
            q_threshold=args.q_threshold,
            exclude_emdb_ids=frozenset(),
            out_dir=args.out_dir,
            tag="full32",
            dpi=args.dpi,
            write_figure=not args.no_figures and not run_filtered,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
