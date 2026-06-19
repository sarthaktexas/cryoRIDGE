"""Sharpening sensitivity of cohort ρ(B_iso, V): map-processing quality diagnostic.

Tests whether maps with more aggressive depositor sharpening show weaker
ρ(B-factor, constraint V) at Cα. Depositor-reported sharpening B is parsed from
EMDB when present; otherwise we use a Guinier proxy:

  B_sharpen ≈ B_primary − B_avg   (negative ⇒ sharpened vs avg-of-halves)

Reads ``outputs/cohort_summary/bfactor_horse_race.csv`` for per-map ρ(B, V).
Writes ``bfactor_sharpening_sensitivity.csv`` and a two-panel cohort figure.

Example::

    source .venv/bin/activate
    python scripts/run_cohort_bfactor_sharpening_sensitivity.py
    python scripts/run_cohort_bfactor_sharpening_sensitivity.py --figure-only
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
from scipy import stats as scipy_stats

from style.nature import apply, label_panel, savefig as save_nature
from style.thesis_palette import PALETTES

from cryoem_mrc.analysis import build_contour_mask
from cryoem_mrc.cohort_emdb import fetch_emdb_reported_sharpening_b
from cryoem_mrc.cohort_labels import cohort_figure_label, load_display_name_map
from cryoem_mrc.guinier_sharpening import R_MIN_A_DEFAULT, compare_guinier_b_avg_vs_primary
from cryoem_mrc.map_grid import load_full_and_half_maps, resample_volume_onto_grid
from cryoem_mrc.repo_paths import COHORT_MANIFEST, OUTPUTS_ROOT
from cryoem_mrc.structure_validation import load_cohort_manifest_row

OUT_DIR = OUTPUTS_ROOT / "cohort_summary"
HORSE_RACE_CSV = OUT_DIR / "bfactor_horse_race.csv"
SHARPENING_CSV = OUT_DIR / "bfactor_sharpening_estimates.csv"
OUTPUT_CSV = OUT_DIR / "bfactor_sharpening_sensitivity.csv"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, default=COHORT_MANIFEST)
    p.add_argument("--horse-race-csv", type=Path, default=HORSE_RACE_CSV)
    p.add_argument("--sharpening-csv", type=Path, default=SHARPENING_CSV)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--figure-only", action="store_true", help="Rebuild figure from existing CSV")
    p.add_argument("--skip-emdb-fetch", action="store_true", help="Do not query EMDB for reported sharpening B")
    p.add_argument("--dpi", type=int, default=200)
    return p.parse_args(argv)


def _load_horse_race(csv_path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            eid = str(row["emdb_id"]).strip()
            raw = row.get("rho_b_vs_V", "").strip()
            if raw in ("", "nan"):
                continue
            rho = float(raw)
            if not np.isfinite(rho):
                continue
            out[eid] = {
                "rho_b_vs_V": rho,
                "n_in_mask": int(row.get("n_in_mask", 0) or 0),
            }
    return out


def _estimate_sharpening_one(
    emd_id: str,
    *,
    manifest: Path,
    skip_emdb_fetch: bool,
) -> dict | None:
    row = load_cohort_manifest_row(manifest, emd_id)
    if row.get("flexibility_source", "").strip() != "b_factor":
        return None

    half1 = Path(row["half1_path"])
    half2 = Path(row["half2_path"])
    primary_path = Path(row["reference_mrc"])
    if not half1.is_file() or not half2.is_file() or not primary_path.is_file():
        print(f"[bfactor_sharpen] skip EMD-{emd_id}: missing map files", file=sys.stderr, flush=True)
        return None

    contour = float(row["contour"])
    r_max = float(row.get("global_resolution_a") or 0.0)
    if r_max <= 0:
        return None

    bundle = load_full_and_half_maps(
        primary_path,
        half1,
        half2,
        dtype=np.float32,
        reference="half1",
        resample_if_needed=True,
    )
    avg = np.asarray(bundle.half1.data, dtype=np.float32) * 0.5 + np.asarray(
        bundle.half2.data, dtype=np.float32
    ) * 0.5
    primary = np.asarray(bundle.full.data, dtype=np.float32)
    if primary.shape != avg.shape:
        primary = resample_volume_onto_grid(bundle.full, bundle.half1).astype(np.float32, copy=False)

    mask = build_contour_mask(primary, contour)
    est = compare_guinier_b_avg_vs_primary(
        avg,
        primary,
        bundle.half1.voxel_size_zyx,
        r_min_a=R_MIN_A_DEFAULT,
        r_max_a=r_max,
        mask=mask,
    )

    reported_b = float("nan")
    if not skip_emdb_fetch:
        try:
            val = fetch_emdb_reported_sharpening_b(emd_id)
            if val is not None:
                reported_b = float(val)
        except RuntimeError as exc:
            print(f"[bfactor_sharpen] EMD-{emd_id}: EMDB fetch failed ({exc})", file=sys.stderr, flush=True)

    return {
        "emdb_id": emd_id,
        "global_resolution_a": r_max,
        "b_avg_guinier": est.b_avg_guinier,
        "b_primary_guinier": est.b_primary_guinier,
        "b_sharpening_guinier": est.b_sharpening_delta,
        "b_avg_r_squared": est.b_avg_r_squared,
        "b_primary_r_squared": est.b_primary_r_squared,
        "reported_sharpening_b_emdb": reported_b,
    }


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def _fmt_csv_val(val: object) -> object:
    if isinstance(val, float):
        return f"{val:.6f}" if np.isfinite(val) else ""
    return val


def _merge_records(
    horse: dict[str, dict],
    sharpen_rows: list[dict],
    manifest: Path,
) -> list[dict]:
    names = load_display_name_map(manifest)
    by_id = {str(r["emdb_id"]): r for r in sharpen_rows}
    recs: list[dict] = []
    for eid, hr in horse.items():
        if int(hr.get("n_in_mask", 0) or 0) < 30:
            continue
        sh = by_id.get(eid)
        if sh is None:
            continue
        rec = {
            "emdb_id": eid,
            "display_name": names.get(eid, eid),
            "n_in_mask": hr["n_in_mask"],
            "rho_b_vs_V": hr["rho_b_vs_V"],
            **sh,
        }
        recs.append(rec)
    return recs


def _sharpening_x(rec: dict) -> float:
    reported = float(rec.get("reported_sharpening_b_emdb", float("nan")))
    if np.isfinite(reported):
        return reported
    return float(rec.get("b_sharpening_guinier", float("nan")))


def _build_figure(recs: list[dict], out_dir: Path, dpi: int) -> Path:
    usable = [r for r in recs if np.isfinite(_sharpening_x(r))]
    if len(usable) < 3:
        raise ValueError("Need at least three maps with finite sharpening B for figure")

    x = np.array([_sharpening_x(r) for r in usable], dtype=np.float64)
    y = np.array([float(r["rho_b_vs_V"]) for r in usable], dtype=np.float64)
    names = {r["emdb_id"]: r["display_name"] for r in usable}
    n_reported = sum(1 for r in usable if np.isfinite(float(r.get("reported_sharpening_b_emdb", float("nan")))))

    fig, (ax_sc, ax_bar) = plt.subplots(1, 2, figsize=(11.0, 4.8))

    apply(ax_sc)
    ax_sc.scatter(x, y, s=28, c=PALETTES["categorical"][0], edgecolors="0.2", linewidths=0.4, zorder=3)
    if x.size >= 3:
        rho_xy = scipy_stats.spearmanr(x, y).statistic
        coef = np.polyfit(x, y, 1)
        xline = np.linspace(float(x.min()), float(x.max()), 50)
        ax_sc.plot(xline, np.polyval(coef, xline), color=PALETTES["categorical"][1], linewidth=0.9)
        title = f"ρ(B, V) vs sharpening B (Spearman={rho_xy:+.2f}, n={len(usable)})"
    else:
        title = "ρ(B, V) vs sharpening B"
    ax_sc.axhline(0.0, color="0.35", linewidth=0.6)
    ax_sc.axvline(0.0, color="0.35", linewidth=0.6, linestyle=":")
    ax_sc.set_xlabel("Sharpening B (A^2; more negative = sharper)")
    ax_sc.set_ylabel("Spearman ρ(B_iso, V)")
    ax_sc.set_title(title)
    label_panel(ax_sc, "a")

    apply(ax_bar)
    sorted_recs = sorted(usable, key=lambda r: float(r["rho_b_vs_V"]))
    rhos = np.array([float(r["rho_b_vs_V"]) for r in sorted_recs])
    labels = [cohort_figure_label(r["emdb_id"], names=names) for r in sorted_recs]
    ypos = np.arange(len(sorted_recs))
    xmin = float(np.nanmin(x))
    xmax = float(np.nanmax(x))
    norm = plt.Normalize(vmin=xmin, vmax=xmax)
    cmap = matplotlib.colormaps["coolwarm"]
    colors = [cmap(norm(_sharpening_x(r))) for r in sorted_recs]
    ax_bar.barh(ypos, rhos, color=colors, edgecolor="0.2", linewidth=0.4)
    ax_bar.set_yticks(ypos)
    ax_bar.set_yticklabels(labels, fontsize=5)
    ax_bar.axvline(0.0, color="0.3", linewidth=0.6)
    ax_bar.axvline(float(np.median(rhos)), color=PALETTES["categorical"][1], linewidth=0.8, linestyle="--")
    ax_bar.set_xlabel("Spearman ρ(B_iso, V)")
    ax_bar.set_title("Per-map B–V coupling (color = sharpening B)")
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_bar, fraction=0.046, pad=0.02)
    cbar.set_label("Sharpening B (A^2)", fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    label_panel(ax_bar, "b")

    proxy_note = (
        f"Guinier proxy for {len(usable) - n_reported}/{len(usable)} maps"
        if n_reported < len(usable)
        else f"{n_reported} maps with EMDB-reported sharpening B"
    )
    fig.suptitle(f"B-factor / V coupling vs map sharpening — {proxy_note}", fontsize=11, y=1.02)
    fig.tight_layout()
    out = out_dir / "bfactor_sharpening_sensitivity"
    save_nature(fig, out, dpi=dpi)
    plt.close(fig)
    return out.with_suffix(".png")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if not args.horse_race_csv.is_file():
        print(f"[bfactor_sharpen] missing {args.horse_race_csv}", file=sys.stderr)
        return 2

    horse = _load_horse_race(args.horse_race_csv)
    if not horse:
        print("[bfactor_sharpen] no finite ρ(B,V) rows in horse-race CSV", file=sys.stderr)
        return 2

    if args.figure_only:
        if not args.out_dir.joinpath("bfactor_sharpening_sensitivity.csv").is_file():
            print("[bfactor_sharpen] missing merged CSV; run without --figure-only first", file=sys.stderr)
            return 2
        with args.out_dir.joinpath("bfactor_sharpening_sensitivity.csv").open(newline="") as f:
            recs = list(csv.DictReader(f))
        for rec in recs:
            for key in (
                "rho_b_vs_V",
                "b_avg_guinier",
                "b_primary_guinier",
                "b_sharpening_guinier",
                "reported_sharpening_b_emdb",
                "global_resolution_a",
            ):
                if key in rec and rec[key] not in ("", "nan"):
                    rec[key] = float(rec[key])
        fig_path = _build_figure(recs, args.out_dir, args.dpi)
        print(f"[bfactor_sharpen] figure → {fig_path}", flush=True)
        return 0

    sharpen_rows: list[dict] = []
    if args.sharpening_csv.is_file():
        with args.sharpening_csv.open(newline="") as f:
            cached = {str(r["emdb_id"]).strip(): r for r in csv.DictReader(f)}
    else:
        cached = {}

    for eid in sorted(horse):
        if eid in cached:
            row = {k: cached[eid].get(k, "") for k in cached[eid]}
            for key in (
                "b_avg_guinier",
                "b_primary_guinier",
                "b_sharpening_guinier",
                "b_avg_r_squared",
                "b_primary_r_squared",
                "reported_sharpening_b_emdb",
                "global_resolution_a",
            ):
                raw = str(row.get(key, "")).strip()
                row[key] = float(raw) if raw not in ("", "nan") else float("nan")
            sharpen_rows.append(row)
            print(f"[bfactor_sharpen] EMD-{eid}: cached sharpening B_guinier={row['b_sharpening_guinier']:+.1f}", flush=True)
            continue

        payload = _estimate_sharpening_one(
            eid, manifest=args.manifest, skip_emdb_fetch=args.skip_emdb_fetch
        )
        if payload is None:
            continue
        sharpen_rows.append(payload)
        print(
            f"[bfactor_sharpen] EMD-{eid}: B_guinier={payload['b_sharpening_guinier']:+.1f} "
            f"(avg={payload['b_avg_guinier']:+.1f}, primary={payload['b_primary_guinier']:+.1f})",
            flush=True,
        )

    if not sharpen_rows:
        print("[bfactor_sharpen] no sharpening estimates", file=sys.stderr)
        return 2

    sharpen_fields = list(sharpen_rows[0].keys())
    args.sharpening_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.sharpening_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sharpen_fields)
        w.writeheader()
        for row in sharpen_rows:
            w.writerow({k: _fmt_csv_val(row[k]) for k in sharpen_fields})

    recs = _merge_records(horse, sharpen_rows, args.manifest)
    if len(recs) < 3:
        print("[bfactor_sharpen] fewer than three merged rows", file=sys.stderr)
        return 2

    merged_fields = [
        "emdb_id",
        "display_name",
        "n_in_mask",
        "rho_b_vs_V",
        "global_resolution_a",
        "b_avg_guinier",
        "b_primary_guinier",
        "b_sharpening_guinier",
        "b_avg_r_squared",
        "b_primary_r_squared",
        "reported_sharpening_b_emdb",
    ]
    _write_csv(args.out_dir / "bfactor_sharpening_sensitivity.csv", recs, merged_fields)
    (args.out_dir / "bfactor_sharpening_sensitivity.json").write_text(
        json.dumps(recs, indent=2) + "\n"
    )

    x = np.array([_sharpening_x(r) for r in recs], dtype=np.float64)
    y = np.array([float(r["rho_b_vs_V"]) for r in recs], dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) >= 3:
        rho = scipy_stats.spearmanr(x[m], y[m]).statistic
        print(
            f"[bfactor_sharpen] cohort Spearman(ρ(B,V), sharpening B) = {rho:+.3f} (n={int(m.sum())})",
            flush=True,
        )

    fig_path = _build_figure(recs, args.out_dir, args.dpi)
    print(f"[bfactor_sharpen] wrote {args.out_dir / 'bfactor_sharpening_sensitivity.csv'}", flush=True)
    print(f"[bfactor_sharpen] figure → {fig_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
