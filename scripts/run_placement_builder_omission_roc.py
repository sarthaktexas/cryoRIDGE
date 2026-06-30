"""Builder-omission ROC: V vs ResMap on built vs gap-proxy refusal sites.

Operational validation for partially built depositions. Positive class = in-mask
sequence-gap proxy sites (builder did not place Cα); negative = in-mask built Cα.

Example::

    source .venv/bin/activate
    python scripts/run_placement_builder_omission_roc.py
    python scripts/run_placement_builder_omission_roc.py \\
        --manifest cohort/expansion_manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from style.figures import apply, savefig as save_nature
from style.palette import PALETTES

from thesis.builder_omission import (
    BUILDER_OMISSION_MAX_GAP_DEFAULT,
    BuilderOmissionMapStats,
    iter_builder_omission_maps,
    load_per_map_frames_for_builder_omission_roc,
    summarize_builder_omission_roc_per_map,
)
from thesis.placement_utility import (
    PLACEMENT_Q_ROC_PREDICTORS,
    PREDICTOR_LABELS,
    cohort_representative_roc,
)
from cryoem_mrc.repo_paths import COHORT_MANIFEST, OUTPUTS_ROOT

OUT_DIR = OUTPUTS_ROOT / "cohort_summary"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, default=COHORT_MANIFEST)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--sphere-radius-a", type=float, default=2.0)
    p.add_argument("--max-gap-length", type=int, default=BUILDER_OMISSION_MAX_GAP_DEFAULT)
    p.add_argument("--emdb-id", action="append", default=[], help="Restrict to one or more EMDB IDs")
    p.add_argument(
        "--allow-lh-recompute",
        action="store_true",
        help="Recompute V from half-maps when reliability.npz is missing (slow; high memory)",
    )
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--no-figures", action="store_true")
    return p.parse_args(argv)


def _write_map_stats_csv(
    stats: list[BuilderOmissionMapStats],
    out_dir: Path,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "placement_builder_omission_map_stats.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "emdb_id",
                "n_built",
                "n_omission",
                "n_gaps",
                "frac_omission_resmap_finite",
                "frac_built_resmap_finite",
                "median_v_built",
                "median_v_omission",
            ],
        )
        w.writeheader()
        for s in stats:
            w.writerow(
                {
                    "emdb_id": s.emdb_id,
                    "n_built": s.n_built,
                    "n_omission": s.n_omission,
                    "n_gaps": s.n_gaps,
                    "frac_omission_resmap_finite": f"{s.frac_omission_resmap_finite:.4f}",
                    "frac_built_resmap_finite": f"{s.frac_built_resmap_finite:.4f}",
                    "median_v_built": f"{s.median_v_built:.4f}",
                    "median_v_omission": f"{s.median_v_omission:.4f}",
                }
            )
    return path


def _write_roc_csv(rows: list[dict[str, object]], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "placement_builder_omission_roc_per_map.csv"
    if not rows:
        path.write_text("emdb_id\n", encoding="utf-8")
        return path
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    return path


def _median_by_predictor(rows: list[dict[str, object]], key: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for pid in PLACEMENT_Q_ROC_PREDICTORS:
        vals = [
            float(r[key])
            for r in rows
            if r["predictor"] == pid and np.isfinite(float(r[key]))
        ]
        out[pid] = float(np.median(vals)) if vals else float("nan")
    return out


def _plot_roc(
    per_map: list[tuple[str, pd.DataFrame]],
    *,
    out_dir: Path,
    dpi: int,
) -> Path:
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    apply(ax)
    colors = PALETTES["categorical"]
    for i, pid in enumerate(PLACEMENT_Q_ROC_PREDICTORS):
        summary = cohort_representative_roc(
            per_map,
            pid,
            ground_truth="builder_omission",
        )
        if not summary.fpr:
            continue
        ax.plot(
            summary.fpr,
            summary.tpr,
            color=colors[i % len(colors)],
            lw=2.0,
            label=f"{PREDICTOR_LABELS[pid]} (median AUC={summary.median_auc:.2f})",
        )
    ax.plot([0, 1], [0, 1], "--", color="0.65", lw=1.0)
    ax.set_xlabel("False positive rate (built sites flagged)")
    ax.set_ylabel("True positive rate (omission sites flagged)")
    ax.set_title("Builder-omission ROC (representative map)")
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / "placement_builder_omission_roc.png"
    save_nature(fig, png, dpi=dpi)
    plt.close(fig)
    return png


def _write_summary_md(
    *,
    rows: list[dict[str, object]],
    stats: list[BuilderOmissionMapStats],
    manifest: Path,
    out_dir: Path,
    n_eligible: int,
) -> Path:
    med_auc = _median_by_predictor(rows, "auc")
    med_tpr = _median_by_predictor(rows, "tpr_at_10pct_fpr")
    v_auc = med_auc.get("constraint_v", float("nan"))
    res_auc = med_auc.get("resmap_locres_worse_than_median", float("nan"))
    delta = v_auc - res_auc if np.isfinite(v_auc) and np.isfinite(res_auc) else float("nan")

    omission_cov = [s.frac_omission_resmap_finite for s in stats if np.isfinite(s.frac_omission_resmap_finite)]
    med_om_cov = float(np.median(omission_cov)) if omission_cov else float("nan")

    lines = [
        "# Builder-omission ROC (operational validation)",
        "",
        f"Manifest: `{manifest}`",
        "",
        "**Ground truth (operational, not independent):** Class 1 = in-mask interpolated",
        "sites at internal sequence gaps (no deposited Cα); Class 0 = in-mask built Cα.",
        "Interpret as builder-abandonment proxy; see limitations in thesis text.",
        "",
        f"- Maps with ROC: **{len({r['emdb_id'] for r in rows})}** / {n_eligible} eligible",
        f"- Median ResMap coverage on omission sites: **{med_om_cov:.1%}**",
        "",
        "## Headline medians (per-map ROC, pooled median of map AUCs)",
        "",
        "| Predictor | Median AUC | TPR @ 10% FPR |",
        "|-----------|----------:|--------------:|",
    ]
    for pid in PLACEMENT_Q_ROC_PREDICTORS:
        auc = med_auc.get(pid, float("nan"))
        tpr = med_tpr.get(pid, float("nan"))
        lines.append(
            f"| {PREDICTOR_LABELS[pid]} | {auc:.3f} | {tpr:.3f} |"
            if np.isfinite(auc)
            else f"| {PREDICTOR_LABELS[pid]} | — | — |"
        )
    if np.isfinite(delta):
        lines.extend(
            [
                "",
                f"**ΔAUC (V − ResMap):** {delta:+.3f}",
            ]
        )
    lines.extend(
        [
            "",
            "CSV: `placement_builder_omission_roc_per_map.csv`, "
            "`placement_builder_omission_map_stats.csv`.",
            "",
        ]
    )
    path = out_dir / "PLACEMENT_BUILDER_OMISSION_ROC.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    frames, map_stats = load_per_map_frames_for_builder_omission_roc(
        manifest=args.manifest,
        sphere_radius_a=args.sphere_radius_a,
        max_gap_length=args.max_gap_length,
        emdb_ids=args.emdb_id or None,
        allow_lh_recompute=args.allow_lh_recompute,
    )
    n_eligible = len(iter_builder_omission_maps(args.manifest))
    if len(frames) < 1:
        print(
            f"[builder_omission_roc] no maps with enough built + gap-proxy sites (manifest={args.manifest})",
            file=sys.stderr,
        )
        return 2

    per_map = [(eid, df) for eid, df, _ in frames]
    rows = summarize_builder_omission_roc_per_map(per_map)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    _write_map_stats_csv(map_stats, args.out_dir)
    roc_csv = _write_roc_csv(rows, args.out_dir)
    summary_md = _write_summary_md(
        rows=rows,
        stats=map_stats,
        manifest=args.manifest,
        out_dir=args.out_dir,
        n_eligible=n_eligible,
    )

    fig_path = None
    if not args.no_figures and rows:
        fig_path = _plot_roc(per_map, out_dir=args.out_dir, dpi=args.dpi)

    meta = {
        "manifest": str(args.manifest),
        "n_maps": len(frames),
        "median_auc": _median_by_predictor(rows, "auc"),
        "median_tpr_at_10pct_fpr": _median_by_predictor(rows, "tpr_at_10pct_fpr"),
    }
    (args.out_dir / "placement_builder_omission_roc_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"[builder_omission_roc] maps={len(frames)} rows={len(rows)}", flush=True)
    print(f"[builder_omission_roc] {roc_csv}", flush=True)
    print(f"[builder_omission_roc] {summary_md}", flush=True)
    if fig_path:
        print(f"[builder_omission_roc] {fig_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
