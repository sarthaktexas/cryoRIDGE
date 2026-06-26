"""Compare BlocRes vs ResMap local-resolution coupling on the same cohort maps.

Reads pre-exported ``cross_metric_correlations.csv`` from both
``metric_comparison/`` and ``metric_comparison_resmap/`` (no map reload) and writes:

- ``outputs/cohort_summary/v_vs_locres_summary_both.csv`` — paired V vs locres ρ per map
- ``outputs/cohort_summary/cohort_locres_method_v_pairs.png`` — grouped bars (BlocRes + ResMap)
- ``outputs/cohort_summary/cohort_locres_method_v_scatter.png`` — BlocRes ρ vs ResMap ρ
- ``outputs/cohort_summary/cohort_cross_metric_locres_pairs_both.png`` — four metric pairs × both methods

Re-run after updating ResMap MRCs (ResMap: no ``--maskVol``; auto-mask only)::

    python scripts/run_metric_comparison_export.py --all --locres-source resmap
    python scripts/run_locres_method_comparison.py
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
import pandas as pd
from scipy import stats

from style.nature import apply, savefig as save_nature
from style.thesis_palette import PALETTES

from cryoem_mrc.cohort_labels import cohort_figure_label, load_display_name_map
from cryoem_mrc.half_map_repro import WINDOWED_HALFMAP_CORRELATION_KEY
from cryoem_mrc.metric_comparison import LocresSource, metric_comparison_dirname
from cryoem_mrc.repo_paths import COHORT_MANIFEST, OUTPUTS_ROOT, emd_output_dir

LOCres_PAIR_KEYS = (
    ("v_metric", "local_resolution", "V vs locres"),
    ("b_factor", "local_resolution", "B vs locres"),
    (WINDOWED_HALFMAP_CORRELATION_KEY, "local_resolution", "windowed CC vs locres"),
    ("local_variance", "local_resolution", "Var vs locres"),
)

METHOD_COLORS = {
    "blocres": PALETTES["categorical"][0],
    "resmap": PALETTES["categorical"][1],
}
METHOD_LABELS = {"blocres": "BlocRes", "resmap": "ResMap"}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, default=COHORT_MANIFEST)
    p.add_argument("--out-dir", type=Path, default=OUTPUTS_ROOT / "cohort_summary")
    p.add_argument("--dpi", type=int, default=200)
    return p.parse_args(argv)


def _read_v_rho(emdb_id: str, source: LocresSource) -> float:
    path = emd_output_dir(emdb_id) / metric_comparison_dirname(source) / "cross_metric_correlations.csv"
    if not path.is_file():
        return float("nan")
    corr = pd.read_csv(path, index_col=0)
    corr.index = corr.index.astype(str)
    corr.columns = corr.columns.astype(str)
    if "v_metric" not in corr.index or "local_resolution" not in corr.columns:
        return float("nan")
    return float(corr.loc["v_metric", "local_resolution"])


def _read_pair_rho(emdb_id: str, source: LocresSource, metric_a: str, metric_b: str) -> float:
    path = emd_output_dir(emdb_id) / metric_comparison_dirname(source) / "cross_metric_correlations.csv"
    if not path.is_file():
        return float("nan")
    corr = pd.read_csv(path, index_col=0)
    corr.index = corr.index.astype(str)
    corr.columns = corr.columns.astype(str)
    if metric_a not in corr.index or metric_b not in corr.columns:
        return float("nan")
    return float(corr.loc[metric_a, metric_b])


def _eligible_ids(manifest: Path) -> list[str]:
    ids: list[str] = []
    with manifest.open(newline="") as f:
        for row in csv.DictReader(f):
            eid = str(row["emdb_id"]).strip()
            bloc = emd_output_dir(eid) / metric_comparison_dirname("blocres") / "cross_metric_correlations.csv"
            resmap = emd_output_dir(eid) / metric_comparison_dirname("resmap") / "cross_metric_correlations.csv"
            if bloc.is_file() or resmap.is_file():
                ids.append(eid)
    return ids


def _build_summary_table(ids: list[str], *, manifest: Path) -> pd.DataFrame:
    names = load_display_name_map(manifest)
    rows: list[dict[str, object]] = []
    for eid in ids:
        rho_b = _read_v_rho(eid, "blocres")
        rho_r = _read_v_rho(eid, "resmap")
        rows.append(
            {
                "emdb_id": eid,
                "display_name": names.get(eid, ""),
                "rho_V_blocres": rho_b,
                "rho_V_resmap": rho_r,
                "has_blocres": np.isfinite(rho_b),
                "has_resmap": np.isfinite(rho_r),
                "has_both": np.isfinite(rho_b) and np.isfinite(rho_r),
            }
        )
    df = pd.DataFrame(rows)
    sort_key = df["rho_V_blocres"].fillna(df["rho_V_resmap"])
    return df.assign(_sort=sort_key).sort_values("_sort", na_position="last").drop(columns="_sort")


def _build_v_pairs_figure(df: pd.DataFrame, out_dir: Path, dpi: int, *, manifest: Path) -> Path:
    """Grouped horizontal bars: BlocRes and ResMap ρ(V, locres) per map."""
    usable = df[df["has_blocres"] | df["has_resmap"]].copy()
    if len(usable) < 2:
        raise ValueError("Need at least two maps with finite ρ(V, locres) from either method")

    fig, ax = plt.subplots(figsize=(10.5, max(5.0, 0.24 * len(usable) + 1.5)))
    apply(ax)
    ypos = np.arange(len(usable))
    width = 0.35
    bloc_vals = usable["rho_V_blocres"].to_numpy(dtype=float)
    res_vals = usable["rho_V_resmap"].to_numpy(dtype=float)

    ax.barh(ypos - width / 2, bloc_vals, height=width, color=METHOD_COLORS["blocres"],
            label=METHOD_LABELS["blocres"], edgecolor="0.2", linewidth=0.3)
    ax.barh(ypos + width / 2, res_vals, height=width, color=METHOD_COLORS["resmap"],
            label=METHOD_LABELS["resmap"], edgecolor="0.2", linewidth=0.3)

    names = load_display_name_map(manifest)
    ax.set_yticks(ypos)
    ax.set_yticklabels(
        [cohort_figure_label(str(r["emdb_id"]), names=names) for _, r in usable.iterrows()],
        fontsize=6,
    )
    ax.axvline(0.0, color="0.35", linewidth=0.6)
    ax.set_xlabel("Spearman ρ(V, local resolution)")
    n_both = int(usable["has_both"].sum())
    ax.set_title(f"V vs local resolution — BlocRes and ResMap (n = {len(usable)}, both = {n_both})")
    ax.legend(loc="lower right", frameon=False, fontsize=7)
    fig.tight_layout()
    out = out_dir / "cohort_locres_method_v_pairs"
    save_nature(fig, out, dpi=dpi)
    plt.close(fig)
    return out.with_suffix(".png")


def _build_v_scatter_figure(df: pd.DataFrame, out_dir: Path, dpi: int) -> Path | None:
    both = df[df["has_both"]]
    if len(both) < 3:
        return None

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    apply(ax)
    x = both["rho_V_blocres"].to_numpy(dtype=float)
    y = both["rho_V_resmap"].to_numpy(dtype=float)
    ax.scatter(x, y, s=28, c=METHOD_COLORS["blocres"], edgecolors="0.2", linewidths=0.4, zorder=3)
    lim = max(0.05, float(np.nanmax(np.abs(np.concatenate([x, y])))) * 1.1)
    ax.plot([-lim, lim], [-lim, lim], color="0.5", linewidth=0.6, linestyle="--", zorder=1)
    ax.axhline(0, color="0.75", linewidth=0.4)
    ax.axvline(0, color="0.75", linewidth=0.4)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("BlocRes ρ(V, locres)")
    ax.set_ylabel("ResMap ρ(V, locres)")
    rho, pval = stats.spearmanr(x, y)
    ax.set_title(f"Method agreement (n = {len(both)}, ρ = {rho:+.2f})")
    for _, row in both.iterrows():
        ax.annotate(
            str(row["emdb_id"]),
            (row["rho_V_blocres"], row["rho_V_resmap"]),
            fontsize=5,
            xytext=(3, 3),
            textcoords="offset points",
        )
    fig.tight_layout()
    out = out_dir / "cohort_locres_method_v_scatter"
    save_nature(fig, out, dpi=dpi)
    plt.close(fig)
    return out.with_suffix(".png")


def _build_all_pairs_figure(ids: list[str], out_dir: Path, dpi: int, *, manifest: Path) -> Path:
    """Four metric×locres pairs; each map gets BlocRes + ResMap bars."""
    names = load_display_name_map(manifest)
    records: list[dict[str, object]] = []
    for eid in ids:
        rec: dict[str, object] = {"emdb_id": eid}
        for ma, mb, _label in LOCres_PAIR_KEYS:
            key = f"{ma}|{mb}"
            rec[f"blocres|{key}"] = _read_pair_rho(eid, "blocres", ma, mb)
            rec[f"resmap|{key}"] = _read_pair_rho(eid, "resmap", ma, mb)
        records.append(rec)

    df = pd.DataFrame(records)
    v_key = "v_metric|local_resolution"
    sort_vals = df[f"blocres|{v_key}"].fillna(df[f"resmap|{v_key}"])
    df = df.assign(_sort=sort_vals).sort_values("_sort", na_position="last").drop(columns="_sort")
    usable = df[
        np.isfinite(df[f"blocres|{v_key}"].to_numpy(dtype=float))
        | np.isfinite(df[f"resmap|{v_key}"].to_numpy(dtype=float))
    ]
    if len(usable) < 2:
        raise ValueError("Need at least two maps for combined locres-pairs figure")

    pair_specs = [(ma, mb, label) for ma, mb, label in LOCres_PAIR_KEYS]
    fig, axes = plt.subplots(1, len(pair_specs), figsize=(3.2 * len(pair_specs), max(5.0, 0.22 * len(usable) + 1.5)), sharey=True)
    if len(pair_specs) == 1:
        axes = [axes]

    ypos = np.arange(len(usable))
    width = 0.35
    for ax, (ma, mb, label) in zip(axes, pair_specs):
        apply(ax)
        key = f"{ma}|{mb}"
        bloc = usable[f"blocres|{key}"].to_numpy(dtype=float)
        res = usable[f"resmap|{key}"].to_numpy(dtype=float)
        ax.barh(ypos - width / 2, bloc, height=width, color=METHOD_COLORS["blocres"], edgecolor="0.2", linewidth=0.3)
        ax.barh(ypos + width / 2, res, height=width, color=METHOD_COLORS["resmap"], edgecolor="0.2", linewidth=0.3)
        ax.axvline(0.0, color="0.35", linewidth=0.6)
        ax.set_xlabel("Spearman ρ")
        ax.set_title(label, fontsize=8)

    axes[0].set_yticks(ypos)
    axes[0].set_yticklabels(
        [cohort_figure_label(str(r["emdb_id"]), names=names) for _, r in usable.iterrows()],
        fontsize=5,
    )
    handles = [
        plt.Rectangle((0, 0), 1, 1, fc=METHOD_COLORS["blocres"], ec="0.2", lw=0.3),
        plt.Rectangle((0, 0), 1, 1, fc=METHOD_COLORS["resmap"], ec="0.2", lw=0.3),
    ]
    fig.legend(handles, [METHOD_LABELS["blocres"], METHOD_LABELS["resmap"]], loc="lower right", frameon=False, fontsize=7)
    fig.suptitle(f"Per-map locres coupling — both methods (n = {len(usable)})", fontsize=9, y=1.01)
    fig.tight_layout()
    out = out_dir / "cohort_cross_metric_locres_pairs_both"
    save_nature(fig, out, dpi=dpi)
    plt.close(fig)
    return out.with_suffix(".png")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    ids = _eligible_ids(args.manifest)
    if not ids:
        print("[locres_method] no exported cross_metric_correlations.csv found", file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = _build_summary_table(ids, manifest=args.manifest)
    csv_path = args.out_dir / "v_vs_locres_summary_both.csv"
    summary.to_csv(csv_path, index=False)

    n_bloc = int(summary["has_blocres"].sum())
    n_res = int(summary["has_resmap"].sum())
    n_both = int(summary["has_both"].sum())
    print(
        f"[locres_method] maps: blocres={n_bloc} resmap={n_res} both={n_both} -> {csv_path}",
        flush=True,
    )

    try:
        fig1 = _build_v_pairs_figure(summary, args.out_dir, args.dpi, manifest=args.manifest)
        print(f"[locres_method] V pairs → {fig1}", flush=True)
    except ValueError as exc:
        print(f"[locres_method] skip V pairs figure: {exc}", file=sys.stderr)

    fig2 = _build_v_scatter_figure(summary, args.out_dir, args.dpi)
    if fig2 is not None:
        print(f"[locres_method] V scatter → {fig2}", flush=True)

    try:
        fig3 = _build_all_pairs_figure(ids, args.out_dir, args.dpi, manifest=args.manifest)
        print(f"[locres_method] all pairs → {fig3}", flush=True)
    except ValueError as exc:
        print(f"[locres_method] skip all-pairs figure: {exc}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
