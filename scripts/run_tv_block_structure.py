"""T/V block-structure experiment (docs/ALTERNATIVE_APPROACHES.md).

Compute three model-free, density-derived scalar maps — Laplacian curvature
strength ``|∇²ρ|``, von Weizsäcker kinetic-energy density ``|∇ρ|²``, and squared
Hessian curvature ``‖H‖_F²`` — sample them at Cα across the cohort, and correlate
each (Spearman, in-mask) against deposited B-factors and BlocRes local resolution.

The hypothesis under test is a **block structure**:

    T (|∇²ρ|, |∇ρ|²)  ──►  local resolution
    V (‖H‖_F²)        ──►  B-factor

with weak off-diagonal coupling, and with the von Weizsäcker T de-correlating
from V more than the Laplacian T does.

Outputs (under ``outputs/cohort_summary/``):

- ``tv_block_structure.csv``  — per-map Spearman ρ (feature×target + feature×feature)
- ``tv_block_structure.png``  — median-ρ block heatmap + von-Weizsäcker decorrelation bar
- ``tv_block_structure.json`` — cohort medians and the decorrelation verdict

Example::

    source .venv/bin/activate
    python scripts/run_tv_block_structure.py --all
    python scripts/run_tv_block_structure.py --emd-id 49450
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from style.nature import PALETTES, apply, label_panel, savefig as save_nature

from cryoem_mrc.repo_paths import ANCHOR_EMDB_ID, COHORT_MANIFEST, OUTPUTS_ROOT
from cryoem_mrc.tv_curvature import (
    TV_FEATURE_KEYS,
    TV_TARGET_KEYS,
    compute_map_tv_table,
    tv_block_correlations,
)

FEATURE_LABELS = {
    "T_laplacian_abs": r"$T_\mathrm{Lap}=|\nabla^2\rho|$",
    "T_vonweizsacker": r"$T_\mathrm{vW}=|\nabla\rho|^2$",
    "V_curvature": r"$V=\|H\|_F^2$",
}
TARGET_LABELS = {
    "b_factor": "B-factor",
    "local_resolution": "Local resolution",
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--emd-id", type=str, default=None, help="Single EMDB ID (e.g. 49450)")
    p.add_argument("--all", action="store_true", help="All manifest rows with a local deposited PDB")
    p.add_argument("--manifest", type=Path, default=COHORT_MANIFEST)
    p.add_argument("--out-dir", type=Path, default=OUTPUTS_ROOT / "cohort_summary")
    p.add_argument("--sphere-radius-a", type=float, default=2.0)
    p.add_argument("--min-in-mask", type=int, default=30, help="Skip maps with fewer in-mask Cα")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--figure-only", action="store_true", help="Rebuild figure from existing CSV")
    return p.parse_args(argv)


def _emd_ids_for_all(manifest: Path) -> list[str]:
    ids: list[str] = []
    with manifest.open(newline="") as f:
        for row in csv.DictReader(f):
            src = row.get("flexibility_source", "").strip()
            if src in ("excluded", "skip", ""):
                continue
            pdb = Path(row.get("flexibility_path_or_pdb", "").strip())
            if not pdb.is_file():
                print(f"[tv_block] skip EMD-{row['emdb_id']}: no PDB {pdb}", flush=True)
                continue
            ids.append(str(row["emdb_id"]).strip())
    return ids


def _run_one(emd_id: str, *, manifest: Path, sphere_radius_a: float, min_in_mask: int) -> dict | None:
    try:
        df = compute_map_tv_table(emd_id, manifest=manifest, sphere_radius_a=sphere_radius_a)
    except FileNotFoundError as exc:
        print(f"[tv_block] skip EMD-{emd_id}: {exc}", file=sys.stderr, flush=True)
        return None
    except Exception as exc:  # noqa: BLE001 - one bad map must not abort the cohort
        print(f"[tv_block] ERROR EMD-{emd_id}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return None

    res = tv_block_correlations(df, emdb_id=emd_id)
    if res.n_in_mask < min_in_mask:
        print(f"[tv_block] skip EMD-{emd_id}: only {res.n_in_mask} in-mask Cα", flush=True)
        return None

    rec = res.flat_record()
    print(
        f"[tv_block] EMD-{emd_id}: n={res.n_in_mask} | "
        f"ρ(V,B)={rec.get('rho__V_curvature__vs__b_factor', float('nan')):+.2f} "
        f"ρ(T_vW,res)={rec.get('rho__T_vonweizsacker__vs__local_resolution', float('nan')):+.2f} "
        f"ρ(T_vW,V)={rec.get('rho__T_vonweizsacker__vs__V_curvature', float('nan')):+.2f} "
        f"ρ(T_Lap,V)={rec.get('rho__T_laplacian_abs__vs__V_curvature', float('nan')):+.2f}",
        flush=True,
    )
    return rec


def _write_csv(records: list[dict], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "tv_block_structure.csv"
    fieldnames = list(records[0].keys())
    for rec in records:
        for k in rec:
            if k not in fieldnames:
                fieldnames.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for rec in records:
            w.writerow(
                {
                    k: (f"{v:.6f}" if isinstance(v, float) and np.isfinite(v) else v)
                    for k, v in rec.items()
                }
            )
    return path


def _read_csv(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            rec: dict[str, object] = {}
            for k, v in row.items():
                if k in ("emdb_id",):
                    rec[k] = v
                else:
                    try:
                        rec[k] = float(v) if v not in ("", "nan") else float("nan")
                    except ValueError:
                        rec[k] = v
            rows.append(rec)
    return rows


def _median(values: list[float]) -> float:
    arr = np.array([v for v in values if isinstance(v, float) and np.isfinite(v)], dtype=np.float64)
    return float(np.median(arr)) if arr.size else float("nan")


def _build_summary(records: list[dict]) -> dict:
    block: dict[str, dict[str, float]] = {}
    for feat in TV_FEATURE_KEYS:
        block[feat] = {}
        for tgt in TV_TARGET_KEYS:
            key = f"rho__{feat}__vs__{tgt}"
            block[feat][tgt] = _median([r.get(key, float("nan")) for r in records])

    couplings: dict[str, float] = {}
    for i, fi in enumerate(TV_FEATURE_KEYS):
        for fj in TV_FEATURE_KEYS[i + 1 :]:
            key = f"rho__{fi}__vs__{fj}"
            couplings[key] = _median([r.get(key, float("nan")) for r in records])

    tvw_v = abs(couplings.get("rho__T_vonweizsacker__vs__V_curvature", float("nan")))
    tlap_v = abs(couplings.get("rho__T_laplacian_abs__vs__V_curvature", float("nan")))
    decorrelates = bool(np.isfinite(tvw_v) and np.isfinite(tlap_v) and tvw_v < tlap_v)

    return {
        "n_maps": len(records),
        "median_block": block,
        "median_feature_couplings": couplings,
        "von_weizsacker_decorrelates_from_V": decorrelates,
        "abs_median_rho_TvW_V": tvw_v,
        "abs_median_rho_TLap_V": tlap_v,
    }


def _build_figure(records: list[dict], summary: dict, out_dir: Path, dpi: int) -> Path:
    block = summary["median_block"]
    features = list(TV_FEATURE_KEYS)
    targets = list(TV_TARGET_KEYS)
    mat = np.array([[block[f][t] for t in targets] for f in features], dtype=np.float64)

    fig, (ax_hm, ax_bar) = plt.subplots(1, 2, figsize=(11.0, 5.0))

    apply(ax_hm)
    im = ax_hm.imshow(mat, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto")
    ax_hm.set_xticks(range(len(targets)))
    ax_hm.set_xticklabels([TARGET_LABELS[t] for t in targets], fontsize=8)
    ax_hm.set_yticks(range(len(features)))
    ax_hm.set_yticklabels([FEATURE_LABELS[f] for f in features], fontsize=9)
    for i in range(len(features)):
        for j in range(len(targets)):
            val = mat[i, j]
            txt = f"{val:+.2f}" if np.isfinite(val) else "n/a"
            ax_hm.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                fontsize=9,
                color="white" if np.isfinite(val) and abs(val) > 0.5 else "black",
            )
    cbar = fig.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04)
    cbar.set_label("median Spearman ρ", fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    ax_hm.set_title(f"T/V block structure (cohort median, n={summary['n_maps']} maps)", fontsize=9)
    label_panel(ax_hm, "a")

    apply(ax_bar)
    tvw_v = summary["abs_median_rho_TvW_V"]
    tlap_v = summary["abs_median_rho_TLap_V"]
    bars = ax_bar.bar(
        [r"$T_\mathrm{vW}$ vs $V$", r"$T_\mathrm{Lap}$ vs $V$"],
        [tvw_v, tlap_v],
        color=[PALETTES["categorical"][0], PALETTES["categorical"][1]],
        edgecolor="0.2",
        linewidth=0.5,
    )
    for rect, val in zip(bars, [tvw_v, tlap_v]):
        if np.isfinite(val):
            ax_bar.text(
                rect.get_x() + rect.get_width() / 2.0,
                val,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax_bar.set_ylabel(r"median $|\rho|$ (feature coupling)")
    verdict = "yes" if summary["von_weizsacker_decorrelates_from_V"] else "no"
    ax_bar.set_title(f"Does von Weizsäcker T de-correlate from V?  {verdict}", fontsize=9)
    ax_bar.set_ylim(0.0, 1.0)
    label_panel(ax_bar, "b")

    fig.suptitle("Model-free T/V curvature vs B-factor and local resolution", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = out_dir / "tv_block_structure"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_nature(fig, out, dpi=dpi)
    plt.close(fig)
    return out.with_suffix(".png")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    csv_path = args.out_dir / "tv_block_structure.csv"

    if args.figure_only:
        if not csv_path.is_file():
            print(f"[tv_block] no CSV at {csv_path}", file=sys.stderr)
            return 2
        records = _read_csv(csv_path)
        summary = _build_summary(records)
        (args.out_dir / "tv_block_structure.json").write_text(json.dumps(summary, indent=2) + "\n")
        fig_path = _build_figure(records, summary, args.out_dir, args.dpi)
        print(f"[tv_block] figure → {fig_path}", flush=True)
        return 0

    if not args.all and not args.emd_id:
        print("Specify --emd-id, --all, or --figure-only", file=sys.stderr)
        return 2

    ids = _emd_ids_for_all(args.manifest) if args.all else [args.emd_id.strip()]

    records: list[dict] = []
    for emd_id in ids:
        rec = _run_one(
            emd_id,
            manifest=args.manifest,
            sphere_radius_a=args.sphere_radius_a,
            min_in_mask=args.min_in_mask,
        )
        if rec is not None:
            records.append(rec)

    if not records:
        print("[tv_block] no usable maps", file=sys.stderr)
        return 2

    path = _write_csv(records, args.out_dir)
    summary = _build_summary(records)
    (args.out_dir / "tv_block_structure.json").write_text(json.dumps(summary, indent=2) + "\n")
    fig_path = _build_figure(records, summary, args.out_dir, args.dpi)

    print(f"[tv_block] {len(records)} maps → {path}", flush=True)
    print(f"[tv_block] figure → {fig_path}", flush=True)
    if len(ids) == 1 and ids[0] == ANCHOR_EMDB_ID:
        b = summary["median_block"]
        print(
            f"[tv_block] anchor: ρ(V,B)={b['V_curvature']['b_factor']:+.3f} "
            f"ρ(T_vW,res)={b['T_vonweizsacker']['local_resolution']:+.3f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
