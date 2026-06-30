"""Placement ROC AUC: constraint V vs ResMap with Q-threshold robustness.

Primary analysis (manuscript): **Q < 0.5**, per-map ROC, predictors **V** and **ResMap**.
Reliability is reported only in robustness tables (ranked/binned view of the same V signal).

Supplementary thresholds (not replacements):
- Q < 0.4 — sensitivity
- Q <= 0.2 — severe/Poor band (sparse; fewer maps pass ROC stability)

Cross-reference LOMO headline numbers in ``PLACEMENT_ROC_ROBUSTNESS.md``.

Example::

    source .venv/bin/activate
    python scripts/run_qscore_validation.py --all --cohort-summary
    python scripts/run_placement_q_roc.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from style.nature import apply, savefig as save_nature
from style.thesis_palette import PALETTES

from cryoem_mrc.placement_utility import (
    PLACEMENT_Q_ROC_PREDICTORS,
    PLACEMENT_Q_ROC_ROBUSTNESS_PREDICTORS,
    PLACEMENT_ROC_GROUND_TRUTH_LABELS,
    PLACEMENT_ROC_Q_THRESHOLD_DEFAULT,
    PREDICTOR_LABELS,
    PlacementRocGroundTruth,
    cohort_representative_roc,
    filter_emringer_roc_frames,
    iter_qscore_maps,
    load_per_map_frames_for_q_roc,
    placement_roc_positive_mask,
    summarize_q_roc_per_map,
    write_q_roc_summary_csv,
)
from cryoem_mrc.repo_paths import COHORT_MANIFEST, OUTPUTS_ROOT

OUT_DIR = OUTPUTS_ROOT / "cohort_summary"


@dataclass(frozen=True)
class QThresholdSpec:
    threshold: float
    inclusive: bool
    role: str
    label: str


Q_THRESHOLD_SPECS: tuple[QThresholdSpec, ...] = (
    QThresholdSpec(0.5, False, "primary", "Q < 0.5 (below Good band)"),
    QThresholdSpec(0.4, False, "sensitivity", "Q < 0.4 (below side-chain resolved)"),
    QThresholdSpec(0.2, True, "severe_secondary", "Q <= 0.2 (Poor band)"),
)

PANELS: tuple[tuple[PlacementRocGroundTruth, str, str], ...] = (
    ("q_low", "placement_q_roc", "placement_q_roc_per_map.csv"),
    ("emringer_low", "placement_emringer_roc", "placement_emringer_roc_per_map.csv"),
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, default=COHORT_MANIFEST)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument(
        "--q-threshold",
        type=float,
        default=PLACEMENT_ROC_Q_THRESHOLD_DEFAULT,
        help="Positive class for primary Q ROC figure (default 0.5).",
    )
    p.add_argument("--sphere-radius-a", type=float, default=2.0)
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--no-figures", action="store_true")
    return p.parse_args(argv)


def _median_aucs(
    rows: list[dict[str, object]],
    *,
    predictors: tuple[str, ...] = PLACEMENT_Q_ROC_ROBUSTNESS_PREDICTORS,
) -> dict[str, float]:
    return {
        pid: float(
            np.median(
                [
                    float(r["auc"])
                    for r in rows
                    if r["predictor"] == pid and np.isfinite(float(r["auc"]))
                ]
            )
        )
        for pid in predictors
        if any(r["predictor"] == pid for r in rows)
    }


def _maps_with_stable_roc(
    per_map: list[tuple[str, object]],
    *,
    spec: QThresholdSpec,
    predictors: tuple[str, ...] = PLACEMENT_Q_ROC_ROBUSTNESS_PREDICTORS,
) -> tuple[frozenset[str], float]:
    """Maps with >=30 residues and both Q classes for every robustness predictor."""
    eligible: set[str] = set()
    fracs: list[float] = []
    for eid, df in per_map:
        pos_m, positive = placement_roc_positive_mask(
            df,
            ground_truth="q_low",
            q_threshold=spec.threshold,
            q_inclusive=spec.inclusive,
        )
        fracs.append(float(positive[pos_m].mean()) if int(pos_m.sum()) else float("nan"))
        ok = True
        for pid in predictors:
            rows = summarize_q_roc_per_map(
                [(eid, df)],
                ground_truth="q_low",
                q_threshold=spec.threshold,
                q_inclusive=spec.inclusive,
                predictors=(pid,),  # type: ignore[arg-type]
            )
            if not rows or not np.isfinite(float(rows[0]["auc"])):
                ok = False
                break
        if ok:
            eligible.add(str(eid))
    return frozenset(eligible), float(np.median(fracs)) if fracs else float("nan")


def _plot_roc_panel(
    per_map_frames: list[tuple[str, object]],
    *,
    ground_truth: PlacementRocGroundTruth,
    q_threshold: float,
    q_inclusive: bool,
    out_stem: str,
    out_dir: Path,
    dpi: int,
) -> tuple[Path, list]:
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    apply(ax)
    colors = PALETTES["categorical"]
    summaries = []

    for i, pid in enumerate(PLACEMENT_Q_ROC_PREDICTORS):
        summary = cohort_representative_roc(
            per_map_frames,
            pid,
            ground_truth=ground_truth,
            q_threshold=q_threshold,
            q_inclusive=q_inclusive,
            eligible_emdb_ids=None,
        )
        summaries.append(summary)
        if not summary.fpr:
            continue
        label = (
            f"{PREDICTOR_LABELS[pid]} "
            f"(median AUC={summary.median_auc:.2f}, EMD-{summary.representative_emdb_id})"
        )
        ax.plot(summary.fpr, summary.tpr, color=colors[i % len(colors)], linewidth=1.8, label=label)

    ax.plot([0, 1], [0, 1], color="0.75", linestyle="--", linewidth=0.9)
    ax.set_xlabel("False positive rate")
    if ground_truth == "q_low":
        op = "<=" if q_inclusive else "<"
        ax.set_ylabel(f"True positive rate (Q {op} {q_threshold:.1f})")
        title_gt = f"Q {op} {q_threshold:.1f}"
    else:
        ax.set_ylabel("True positive rate (EMRinger < in-map median)")
        title_gt = "EMRinger < in-map median"
    n_maps = summaries[0].n_maps if summaries else 0
    ax.set_title(f"Placement ROC — V vs ResMap (n={n_maps} maps, {title_gt})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", frameon=False, fontsize=7)
    out = out_dir / out_stem
    save_nature(fig, out, dpi=dpi)
    plt.close(fig)
    return out.with_suffix(".png"), summaries


def _run_panel(
    *,
    ground_truth: PlacementRocGroundTruth,
    per_map: list[tuple[str, object]],
    q_threshold: float,
    q_inclusive: bool,
    out_dir: Path,
    out_stem: str,
    csv_name: str,
    dpi: int,
    no_figures: bool,
    predictors: tuple[str, ...] = PLACEMENT_Q_ROC_PREDICTORS,
) -> dict[str, object]:
    rows = summarize_q_roc_per_map(
        per_map,
        ground_truth=ground_truth,
        q_threshold=q_threshold,
        q_inclusive=q_inclusive,
        predictors=predictors,  # type: ignore[arg-type]
    )
    csv_path = write_q_roc_summary_csv(
        rows,
        out_dir,
        filename=csv_name,
        ground_truth=ground_truth,
        predictors=predictors,  # type: ignore[arg-type]
    )
    medians = _median_aucs(rows, predictors=predictors)  # type: ignore[arg-type]

    print(f"[placement_q_roc] {PLACEMENT_ROC_GROUND_TRUTH_LABELS[ground_truth]}", flush=True)
    print(f"  maps: {len({r['emdb_id'] for r in rows})}", flush=True)
    for pid in predictors:
        if pid in medians and np.isfinite(medians[pid]):
            print(f"  {PREDICTOR_LABELS[pid]}: median AUC={medians[pid]:.3f}", flush=True)
    print(f"  per-map table: {csv_path}", flush=True)

    png_path = None
    if not no_figures and ground_truth == "q_low":
        png_path, _ = _plot_roc_panel(
            per_map,
            ground_truth=ground_truth,
            q_threshold=q_threshold,
            q_inclusive=q_inclusive,
            out_stem=out_stem,
            out_dir=out_dir,
            dpi=dpi,
        )
        print(f"  figure: {png_path}", flush=True)

    return {
        "ground_truth": ground_truth,
        "n_maps": len({r["emdb_id"] for r in rows}),
        "median_auc": medians,
        "csv": str(csv_path),
        "figure": str(png_path) if png_path else None,
    }


def _write_q_threshold_robustness(
    per_map: list[tuple[str, object]],
    out_dir: Path,
) -> Path:
    rows: list[dict[str, object]] = []
    for spec in Q_THRESHOLD_SPECS:
        roc_rows = summarize_q_roc_per_map(
            per_map,
            ground_truth="q_low",
            q_threshold=spec.threshold,
            q_inclusive=spec.inclusive,
            predictors=PLACEMENT_Q_ROC_ROBUSTNESS_PREDICTORS,
        )
        eligible, med_frac = _maps_with_stable_roc(per_map, spec=spec)
        med = _median_aucs(roc_rows)
        for pid in PLACEMENT_Q_ROC_ROBUSTNESS_PREDICTORS:
            n_maps = len({r["emdb_id"] for r in roc_rows if r["predictor"] == pid})
            rows.append(
                {
                    "role": spec.role,
                    "q_threshold": spec.threshold,
                    "q_inclusive": spec.inclusive,
                    "q_band_label": spec.label,
                    "predictor": pid,
                    "median_auc": med.get(pid, float("nan")),
                    "n_maps_roc": n_maps,
                    "n_maps_eligible": len(eligible),
                    "median_frac_positive": med_frac,
                    "analysis": "per_map_roc_inmap_median",
                }
            )

    path = out_dir / "placement_q_roc_q_threshold_robustness.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for row in rows:
            w.writerow(
                {
                    **row,
                    "median_auc": f"{float(row['median_auc']):.4f}"
                    if np.isfinite(float(row["median_auc"]))
                    else "",
                    "median_frac_positive": f"{float(row['median_frac_positive']):.4f}"
                    if np.isfinite(float(row["median_frac_positive"]))
                    else "",
                }
            )
    return path


def _v_reliability_auc_identity(per_map: list[tuple[str, object]], q_threshold: float) -> float:
    """Median |AUC(V) - AUC(reliability)| across maps (should be ~0)."""
    diffs: list[float] = []
    for eid, df in per_map:
        v_rows = summarize_q_roc_per_map(
            [(eid, df)],
            q_threshold=q_threshold,
            predictors=("constraint_v",),  # type: ignore[arg-type]
        )
        rel_rows = summarize_q_roc_per_map(
            [(eid, df)],
            q_threshold=q_threshold,
            predictors=("reliability_below_0_33",),  # type: ignore[arg-type]
        )
        if not v_rows or not rel_rows:
            continue
        v_auc = float(v_rows[0]["auc"])
        rel_auc = float(rel_rows[0]["auc"])
        if np.isfinite(v_auc) and np.isfinite(rel_auc):
            diffs.append(abs(v_auc - rel_auc))
    return float(np.median(diffs)) if diffs else float("nan")


def _read_lomo_reference(out_dir: Path) -> dict[str, float]:
    refs: dict[str, float] = {}
    lomo_v = out_dir / "placement_locres_lomo_medians.csv"
    lomo_rel = out_dir / "placement_lomo_medians.csv"
    if lomo_v.is_file():
        df = pd.read_csv(lomo_v)
        for _, row in df.iterrows():
            refs[f"lomo_{row['predictor']}_auc"] = float(row["median_auc"])
    if lomo_rel.is_file():
        df = pd.read_csv(lomo_rel)
        sub = df[df["predictor"] == "reliability_below_0_33"]
        if len(sub):
            refs["lomo_reliability_auc"] = float(sub.iloc[0]["median_auc"])
    auc_v = out_dir / "placement_auc_v_vs_locres.csv"
    if auc_v.is_file():
        df = pd.read_csv(auc_v)
        refs["per_map_v_auc_legacy"] = float(pd.to_numeric(df["auc_V_blocres"], errors="coerce").median())
    return refs


def _write_robustness_markdown(
    *,
    out_dir: Path,
    per_map: list[tuple[str, object]],
    primary_medians: dict[str, float],
    robustness_csv: Path,
    n_qscore_manifest: int,
    n_resmap_panel: int,
) -> Path:
    rob = pd.read_csv(robustness_csv)
    refs = _read_lomo_reference(out_dir)
    v_rel_median_diff = _v_reliability_auc_identity(per_map, 0.5)

    lines = [
        "# Placement ROC robustness — V vs ResMap vs reliability",
        "",
        "## Cohort framing (reconcile map counts)",
        "",
        f"- **Q-score manifest (eligible, after panel excludes):** {n_qscore_manifest} maps",
        f"- **ResMap + Q validation panel (this script):** {n_resmap_panel} maps "
        "(requires `qscore_validation.csv`, production `v_metric`, and "
        "`metric_comparison_resmap/residue_metrics.csv` with >=30 finite ResMap values)",
        "- **Per-threshold ROC maps** can be **lower** when a map has too few "
        "low-Q residues for a stable ROC (both classes need >=30 scored residues "
        "with finite predictors).",
        "",
        "The headline **LOMO** tables (`placement_lomo_medians.csv`, "
        "`placement_locres_lomo_medians.csv`) use **31–32** Q-score maps with "
        "half-map metrics but **do not require ResMap**; V LOMO uses continuous "
        "`-v_metric` on the held-out map while flag thresholds are fit on train maps.",
        "",
        "This script's per-map ROC uses **in-map medians** for ResMap flags and "
        "does not leave-one-map-out — so ResMap AUC here (~0.81) is **not directly "
        "comparable** to train-median LOMO ResMap (~0.83).",
        "",
        "## Predictor identity",
        "",
        "- **Constraint V (`v_metric`)** — primary half-map placement signal.",
        "- **Reliability score** — monotonic ranked/binned transform of the same V "
        f"(median |AUC(V)−AUC(reliability)| = **{v_rel_median_diff:.4f}** at Q<0.5 "
        "on the ResMap panel). Not an independent predictor.",
        "",
        "## Primary analysis (manuscript default)",
        "",
        "**Ground truth:** Q < 0.5 (below Good band; Pintilie et al.).",
        f"**Maps (ROC-stable):** see `n_maps_roc` in robustness CSV (panel loaded: {n_resmap_panel}).",
        "",
        "| Predictor | Median per-map AUC |",
        "|-----------|-------------------:|",
    ]
    for pid in PLACEMENT_Q_ROC_PREDICTORS:
        val = primary_medians.get(pid, float("nan"))
        lines.append(
            f"| {PREDICTOR_LABELS[pid]} | {val:.3f} |"
            if np.isfinite(val)
            else f"| {PREDICTOR_LABELS[pid]} | — |"
        )

    lines.extend(
        [
            "",
            "### LOMO cross-reference (different analysis; do not mix without caption)",
            "",
            "| Source | V / constraint | Reliability <0.33 | ResMap |",
            "|--------|---------------:|------------------:|-------:|",
            f"| Per-map ROC (this script, Q<0.5) | "
            f"{primary_medians.get('constraint_v', float('nan')):.3f} | "
            f"{_median_aucs(summarize_q_roc_per_map(per_map, predictors=PLACEMENT_Q_ROC_ROBUSTNESS_PREDICTORS)).get('reliability_below_0_33', float('nan')):.3f} | "
            f"{primary_medians.get('resmap_locres_worse_than_median', float('nan')):.3f} |",
            f"| LOMO / legacy (`placement_locres_lomo_medians.csv`) | "
            f"{refs.get('lomo_constraint_v_auc', float('nan')):.3f} | "
            f"{refs.get('lomo_reliability_auc', float('nan')):.3f} | "
            f"{refs.get('lomo_resmap_locres_auc', float('nan')):.3f} |",
            "",
            "## Q-threshold robustness (supplementary; primary remains Q < 0.5)",
            "",
            "| Role | Q rule | Maps | Median frac low-Q | V AUC | Reliability AUC | ResMap AUC |",
            "|------|--------|-----:|------------------:|------:|----------------:|-----------:|",
        ]
    )
    for spec in Q_THRESHOLD_SPECS:
        sub = rob[rob["role"] == spec.role]
        if sub.empty:
            continue
        n_maps = int(sub["n_maps_roc"].iloc[0])
        frac = float(sub["median_frac_positive"].iloc[0])
        v = float(sub.loc[sub["predictor"] == "constraint_v", "median_auc"].iloc[0])
        rel = float(sub.loc[sub["predictor"] == "reliability_below_0_33", "median_auc"].iloc[0])
        res = float(sub.loc[sub["predictor"] == "resmap_locres_worse_than_median", "median_auc"].iloc[0])
        caveat = ""
        if spec.role == "severe_secondary":
            caveat = " *(sparse; interpret with caution)*"
        lines.append(
            f"| {spec.role} | {spec.label} | {n_maps} | {frac:.1%} | {v:.3f} | {rel:.3f} | {res:.3f} |{caveat}"
        )

    lines.extend(
        [
            "",
            "**Severe/Poor (Q <= 0.2):** only maps with enough positives for ROC; "
            "median positive rate ~1.5% on the full panel.",
            "",
            "**Sensitivity (Q < 0.4):** AUC stable vs primary — supports the 0.5 choice.",
            "",
            "Full table: `placement_q_roc_q_threshold_robustness.csv`.",
            "",
        ]
    )
    path = out_dir / "PLACEMENT_ROC_ROBUSTNESS.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    frames = load_per_map_frames_for_q_roc(
        manifest=args.manifest,
        sphere_radius_a=args.sphere_radius_a,
        require_resmap=True,
    )
    if len(frames) < 3:
        print(
            "[placement_q_roc] need >= 3 maps with qscore_validation + ResMap",
            file=sys.stderr,
        )
        return 2

    per_map_q = [(eid, df) for eid, df, _ in frames]
    n_qscore = len(iter_qscore_maps(manifest=args.manifest))

    robustness_path = _write_q_threshold_robustness(per_map_q, args.out_dir)
    print(f"[placement_q_roc] Q-threshold robustness: {robustness_path}", flush=True)

    emringer_frames = filter_emringer_roc_frames(frames, manifest=args.manifest)
    per_map_em = [(eid, df) for eid, df, _ in emringer_frames]
    if len(per_map_em) < 3:
        print("[placement_q_roc] need >= 3 EMRinger maps", file=sys.stderr)
        return 2

    primary_spec = next(s for s in Q_THRESHOLD_SPECS if s.role == "primary")
    panel_results: list[dict[str, object]] = []
    panel_results.append(
        _run_panel(
            ground_truth="q_low",
            per_map=per_map_q,
            q_threshold=primary_spec.threshold,
            q_inclusive=primary_spec.inclusive,
            out_dir=args.out_dir,
            out_stem="placement_q_roc",
            csv_name="placement_q_roc_per_map.csv",
            dpi=args.dpi,
            no_figures=args.no_figures,
        )
    )
    panel_results.append(
        _run_panel(
            ground_truth="emringer_low",
            per_map=per_map_em,
            q_threshold=primary_spec.threshold,
            q_inclusive=False,
            out_dir=args.out_dir,
            out_stem="placement_emringer_roc",
            csv_name="placement_emringer_roc_per_map.csv",
            dpi=args.dpi,
            no_figures=True,
            predictors=PLACEMENT_Q_ROC_PREDICTORS,
        )
    )

    robustness_md = _write_robustness_markdown(
        out_dir=args.out_dir,
        per_map=per_map_q,
        primary_medians=panel_results[0]["median_auc"],  # type: ignore[arg-type]
        robustness_csv=robustness_path,
        n_qscore_manifest=n_qscore,
        n_resmap_panel=len(per_map_q),
    )

    meta = {
        "primary_q_threshold": primary_spec.threshold,
        "q_threshold_robustness_csv": str(robustness_path),
        "robustness_markdown": str(robustness_md),
        "n_maps_resmap_panel": len(per_map_q),
        "n_maps_qscore_manifest": n_qscore,
        "predictors_primary": list(PLACEMENT_Q_ROC_PREDICTORS),
        "predictors_robustness": list(PLACEMENT_Q_ROC_ROBUSTNESS_PREDICTORS),
        "panels": panel_results,
        "lomo_reference": _read_lomo_reference(args.out_dir),
    }
    json_path = args.out_dir / "placement_q_roc.json"
    json_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    # Short pointer doc (figure-facing)
    md_path = args.out_dir / "PLACEMENT_Q_ROC.md"
    md_path.write_text(
        "\n".join(
            [
                "# Placement ROC — constraint V vs ResMap",
                "",
                "**Primary:** Q < 0.5; predictors: **V** and **ResMap** (per-map ROC).",
                "",
                "See **`PLACEMENT_ROC_ROBUSTNESS.md`** for:",
                "- Q < 0.4 sensitivity and Q <= 0.2 severe/Poor supplementary rows",
                "- V vs reliability identity and LOMO cross-reference",
                "- Map-count reconciliation (ResMap panel vs full Q cohort)",
                "",
                f"Maps in ResMap panel: **{len(per_map_q)}**.",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"  robustness: {robustness_md}", flush=True)
    print(f"  json: {json_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
