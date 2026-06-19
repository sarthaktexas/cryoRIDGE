"""Discriminating cuts for the B-factor / local-resolution decoupling at the residue level.

Context (see docs/ALTERNATIVE_APPROACHES.md and chat):
The T/V block-structure test found (a) T_vW, T_Lap and V rank-collapse onto each
other at Cα sampling (ρ ≈ 0.97–1.00) — a single-length-scale *degeneracy*, not a
failed reframing — and (b) all three couple only weakly to BlocRes local resolution
(ρ ≈ −0.0 to −0.5) while tracking B-factor strongly. The decisive missing cell is
**ρ(B, local_resolution) itself**: if refined B and FSC local resolution are
themselves decoupled per-residue, that is either a *granularity artifact* (the
local-res axis is spatially smoothed / quantized to shell spacing and barely varies
per residue) or a *genuine decoupling* (B-disorder and FSC-resolution measure
different things). This script computes the cuts that separate those:

1. ρ(B, local_res) per map (Spearman, in-mask)  — the missing cell.
2. Local-res granularity per map: number of unique voxel values (in-mask), a
   step/quantization estimate, the within-chain adjacent-residue redundancy
   (ρ between sequence-neighbor local-res), and a rough spatial autocorrelation
   length (Å) vs. median Cα–Cα spacing.
3. For named bad maps (52525, 28498, 44471): is the failure in B-spread,
   local-res-spread, or both? (CV / IQR for each axis.)

Outputs (under ``outputs/cohort_summary/``):
- ``tv_decoupling_diagnostics.csv``  — one row per map
- ``tv_decoupling_diagnostics.png``  — ρ(B,locres) ranking + ρ vs granularity + bad-map spreads
- ``tv_decoupling_diagnostics.json`` — cohort medians + artifact-vs-genuine verdict

The (T,V) "is it a line?" scatter for representative maps is a separate, heavier
step (it recomputes curvature maps); run it with ``--tv-scatter 49450 16091``.

Example::

    source .venv/bin/activate
    python scripts/run_tv_decoupling_diagnostics.py --all
    python scripts/run_tv_decoupling_diagnostics.py --tv-scatter 49450 16091
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from style.nature import apply, label_panel, savefig as save_nature
from style.thesis_palette import PALETTES

from cryoem_mrc.cohort_labels import cohort_figure_label, load_display_name_map
from cryoem_mrc.local_resolution import _load_locres_volume, locres_blocres_path
from cryoem_mrc.map_grid import load_map_grid
from cryoem_mrc.metric_comparison import load_all_metrics
from cryoem_mrc.repo_paths import COHORT_MANIFEST, OUTPUTS_ROOT
from cryoem_mrc.structure_validation import iter_ca_residues, load_cohort_manifest_row

logging.getLogger("cryoem_mrc.local_resolution").setLevel(logging.ERROR)

BAD_MAPS = ("52525", "28498", "44471")
NOMINAL_CA_SPACING_A = 3.8


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--emd-id", type=str, default=None)
    p.add_argument("--all", action="store_true", help="All manifest rows with a BlocRes map + PDB")
    p.add_argument("--manifest", type=Path, default=COHORT_MANIFEST)
    p.add_argument("--out-dir", type=Path, default=OUTPUTS_ROOT / "cohort_summary")
    p.add_argument("--min-pairs", type=int, default=30)
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument(
        "--tv-scatter",
        nargs="+",
        default=None,
        metavar="EMD_ID",
        help="Recompute curvature maps for these maps and draw raw (T_vW, V) scatters",
    )
    return p.parse_args(argv)


def _spearman(x: np.ndarray, y: np.ndarray, *, min_pairs: int) -> tuple[float, int]:
    m = np.isfinite(x) & np.isfinite(y)
    n = int(m.sum())
    if n < min_pairs:
        return float("nan"), n
    xf, yf = x[m], y[m]
    if xf.std() == 0 or yf.std() == 0:
        return float("nan"), n
    return float(stats.spearmanr(xf, yf).statistic), n


def _cv(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size < 3:
        return float("nan")
    mu = float(np.mean(x))
    if mu == 0:
        return float("nan")
    return float(np.std(x) / abs(mu))


def _iqr(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size < 3:
        return float("nan")
    q1, q3 = np.percentile(x, (25.0, 75.0))
    return float(q3 - q1)


def _adjacent_residue_redundancy(df) -> tuple[float, float]:
    """
    Within-chain sequence-neighbour redundancy of local_resolution.

    Returns (spearman of locres[i] vs locres[i+1], fraction of |Δ|<1e-6 adjacent pairs).
    High redundancy => the resolution axis barely varies between adjacent residues.
    """
    a: list[float] = []
    b: list[float] = []
    n_eq = 0
    n_tot = 0
    for _chain, sub in df.groupby("chain"):
        s = sub.sort_values("seq_num")
        lr = s["local_resolution"].to_numpy(dtype=np.float64)
        sq = s["seq_num"].to_numpy()
        for i in range(len(lr) - 1):
            if sq[i + 1] - sq[i] != 1:
                continue
            if not (np.isfinite(lr[i]) and np.isfinite(lr[i + 1])):
                continue
            a.append(lr[i])
            b.append(lr[i + 1])
            n_tot += 1
            if abs(lr[i + 1] - lr[i]) < 1e-6:
                n_eq += 1
    if n_tot < 10:
        return float("nan"), float("nan")
    av, bv = np.asarray(a), np.asarray(b)
    rho = float("nan")
    if av.std() > 0 and bv.std() > 0:
        rho = float(stats.spearmanr(av, bv).statistic)
    return rho, float(n_eq / n_tot)


def _locres_voxel_granularity(emd_id: str, manifest: Path) -> dict[str, float]:
    """Granularity of the BlocRes volume itself: unique levels, step, autocorr length (Å)."""
    out = {
        "locres_n_unique_voxels": float("nan"),
        "locres_step_a": float("nan"),
        "locres_autocorr_len_a": float("nan"),
        "voxel_size_a": float("nan"),
    }
    path = locres_blocres_path(emd_id)
    if not path.is_file():
        return out
    vol, _ = _load_locres_volume(path)
    grid = load_map_grid(path, dtype=np.float64)
    vsize = float(np.mean(grid.voxel_size_zyx))
    out["voxel_size_a"] = vsize

    finite = vol[np.isfinite(vol) & (vol > 0)]
    if finite.size < 100:
        return out

    # Quantization: number of distinct levels and their typical spacing.
    uniq = np.unique(np.round(finite.astype(np.float64), 4))
    out["locres_n_unique_voxels"] = float(uniq.size)
    if uniq.size >= 2:
        out["locres_step_a"] = float(np.median(np.diff(uniq)))

    # Rough isotropic autocorrelation length via per-axis FFT autocovariance on the
    # mask-cropped, mean-filled volume; report the lag (Å) where it first drops to 1/e.
    mask = np.isfinite(vol) & (vol > 0)
    if mask.sum() < 100:
        return out
    zz, yy, xx = np.where(mask)
    z0, z1 = zz.min(), zz.max() + 1
    y0, y1 = yy.min(), yy.max() + 1
    x0, x1 = xx.min(), xx.max() + 1
    sub = vol[z0:z1, y0:y1, x0:x1].astype(np.float64)
    sub_mask = mask[z0:z1, y0:y1, x0:x1]
    mean_in = float(sub[sub_mask].mean())
    field = np.where(sub_mask, sub - mean_in, 0.0)

    lags_e: list[float] = []
    for axis in range(3):
        n = field.shape[axis]
        if n < 5:
            continue
        F = np.fft.rfft(field, axis=axis)
        ac = np.fft.irfft((F * np.conj(F)).real, n=n, axis=axis)
        # average over the other two axes, normalize lag-0 to 1
        other = tuple(i for i in range(3) if i != axis)
        prof = ac.mean(axis=other)
        if prof[0] <= 0:
            continue
        prof = prof / prof[0]
        below = np.where(prof < (1.0 / np.e))[0]
        if below.size:
            lags_e.append(float(below[0]) * vsize)
    if lags_e:
        out["locres_autocorr_len_a"] = float(np.mean(lags_e))
    return out


def _ca_nn_spacing_a(emd_id: str, manifest: Path) -> float:
    row = load_cohort_manifest_row(manifest, emd_id)
    pdb = Path(row.get("flexibility_path_or_pdb", "").strip())
    if not pdb.is_file():
        return float("nan")
    res = iter_ca_residues(pdb)
    if len(res) < 5:
        return float("nan")
    xyz = np.array([[r.x, r.y, r.z] for r in res], dtype=np.float64)
    # median nearest-neighbour distance (chunked to bound memory)
    nn: list[float] = []
    for i in range(0, len(xyz), 2000):
        block = xyz[i : i + 2000]
        d = np.sqrt(((block[:, None, :] - xyz[None, :, :]) ** 2).sum(axis=2))
        np.fill_diagonal(d[: block.shape[0], i : i + block.shape[0]], np.inf)
        nn.extend(d.min(axis=1).tolist())
    nn_arr = np.array([v for v in nn if np.isfinite(v)])
    return float(np.median(nn_arr)) if nn_arr.size else float("nan")


def _diagnose_one(emd_id: str, *, manifest: Path, min_pairs: int) -> dict | None:
    try:
        df = load_all_metrics(emd_id, manifest=manifest)
    except FileNotFoundError as exc:
        print(f"[decouple] skip EMD-{emd_id}: {exc}", file=sys.stderr, flush=True)
        return None
    except Exception as exc:  # noqa: BLE001
        print(f"[decouple] ERROR EMD-{emd_id}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return None

    if "in_contour_mask" in df.columns:
        df = df[df["in_contour_mask"].astype(bool)]
    if df["local_resolution"].notna().sum() < min_pairs:
        print(f"[decouple] skip EMD-{emd_id}: <{min_pairs} finite local_res", flush=True)
        return None

    b = df["b_factor"].to_numpy(dtype=np.float64)
    lr = df["local_resolution"].to_numpy(dtype=np.float64)
    v = df["v_metric"].to_numpy(dtype=np.float64)

    rho_b_lr, n_b_lr = _spearman(b, lr, min_pairs=min_pairs)
    rho_v_b, _ = _spearman(v, b, min_pairs=min_pairs)
    rho_v_lr, _ = _spearman(v, lr, min_pairs=min_pairs)

    adj_rho, adj_frac_eq = _adjacent_residue_redundancy(df)
    gran = _locres_voxel_granularity(emd_id, manifest)
    ca_spacing = _ca_nn_spacing_a(emd_id, manifest)

    rec = {
        "emdb_id": emd_id,
        "n_b_vs_locres": n_b_lr,
        "rho_b_vs_locres": rho_b_lr,
        "rho_v_vs_b": rho_v_b,
        "rho_v_vs_locres": rho_v_lr,
        "b_cv": _cv(b),
        "b_iqr": _iqr(b),
        "locres_cv": _cv(lr),
        "locres_iqr": _iqr(lr),
        "locres_residue_n_unique": float(np.unique(np.round(lr[np.isfinite(lr)], 3)).size),
        "adjacent_locres_spearman": adj_rho,
        "adjacent_locres_frac_equal": adj_frac_eq,
        "ca_nn_spacing_a": ca_spacing,
        **gran,
    }
    print(
        f"[decouple] EMD-{emd_id}: n={n_b_lr} | ρ(B,locres)={rho_b_lr:+.2f} "
        f"locres_uniq_vox={gran['locres_n_unique_voxels']:.0f} "
        f"autocorr={gran['locres_autocorr_len_a']:.1f}Å vs Cα {ca_spacing:.1f}Å "
        f"adj_ρ={adj_rho:+.2f}",
        flush=True,
    )
    return rec


def _emd_ids_with_locres(manifest: Path) -> list[str]:
    ids: list[str] = []
    with manifest.open(newline="") as f:
        for row in csv.DictReader(f):
            eid = str(row.get("emdb_id", "")).strip()
            if not eid:
                continue
            src = row.get("flexibility_source", "").strip()
            if src in ("excluded", "skip", ""):
                continue
            if not locres_blocres_path(eid).is_file():
                continue
            ids.append(eid)
    return ids


def _median_finite(values: list[float]) -> float:
    arr = np.array([v for v in values if isinstance(v, float) and np.isfinite(v)], dtype=np.float64)
    return float(np.median(arr)) if arr.size else float("nan")


def _build_summary(records: list[dict]) -> dict:
    rho = [r["rho_b_vs_locres"] for r in records]
    finite_rho = [v for v in rho if np.isfinite(v)]
    med_rho = _median_finite(rho)
    med_uniq = _median_finite([r["locres_n_unique_voxels"] for r in records])
    med_adj = _median_finite([r["adjacent_locres_spearman"] for r in records])
    med_autocorr = _median_finite([r["locres_autocorr_len_a"] for r in records])
    med_ca = _median_finite([r["ca_nn_spacing_a"] for r in records])

    # Artifact signature: weak ρ(B,locres) AND coarse/quantized resolution axis
    # (few unique levels, high adjacent-residue redundancy, autocorr length >> Cα spacing).
    artifact_signature = bool(
        np.isfinite(med_rho)
        and abs(med_rho) < 0.4
        and np.isfinite(med_adj)
        and med_adj > 0.9
        and np.isfinite(med_autocorr)
        and np.isfinite(med_ca)
        and med_autocorr > 2.0 * med_ca
    )
    return {
        "n_maps": len(records),
        "median_rho_b_vs_locres": med_rho,
        "n_maps_with_finite_rho": len(finite_rho),
        "median_locres_n_unique_voxels": med_uniq,
        "median_adjacent_locres_spearman": med_adj,
        "median_locres_autocorr_len_a": med_autocorr,
        "median_ca_nn_spacing_a": med_ca,
        "granularity_artifact_signature": artifact_signature,
        "bad_maps": {
            r["emdb_id"]: {
                "rho_b_vs_locres": r["rho_b_vs_locres"],
                "rho_v_vs_b": r["rho_v_vs_b"],
                "b_cv": r["b_cv"],
                "locres_cv": r["locres_cv"],
            }
            for r in records
            if r["emdb_id"] in BAD_MAPS
        },
    }


def _build_figure(records: list[dict], summary: dict, out_dir: Path, dpi: int, *, manifest: Path) -> Path:
    recs = sorted(records, key=lambda r: (np.isfinite(r["rho_b_vs_locres"]), r["rho_b_vs_locres"]))
    rho = np.array([r["rho_b_vs_locres"] for r in recs])
    names = load_display_name_map(manifest)
    labels = [cohort_figure_label(r["emdb_id"], names=names) for r in recs]
    uniq = np.array([r["locres_n_unique_voxels"] for r in recs])
    adj = np.array([r["adjacent_locres_spearman"] for r in recs])

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.5))
    ax_bar, ax_sc, ax_bad = axes

    apply(ax_bar)
    ypos = np.arange(len(recs))
    colors = [
        PALETTES["categorical"][3] if r["emdb_id"] in BAD_MAPS else PALETTES["categorical"][0]
        for r in recs
    ]
    ax_bar.barh(ypos, rho, color=colors, edgecolor="0.2", linewidth=0.4)
    ax_bar.set_yticks(ypos)
    ax_bar.set_yticklabels(labels, fontsize=5)
    ax_bar.axvline(0.0, color="0.3", linewidth=0.6)
    med = summary["median_rho_b_vs_locres"]
    if np.isfinite(med):
        ax_bar.axvline(med, color=PALETTES["categorical"][1], linewidth=0.8, linestyle="--",
                       label=f"median {med:+.2f}")
        ax_bar.legend(loc="lower right", fontsize=6)
    ax_bar.set_xlabel("Spearman ρ(B-factor, local resolution), in-mask Cα")
    ax_bar.set_title("The missing cell: B vs local resolution\n(orange = flagged bad maps)", fontsize=8)
    label_panel(ax_bar, "a")

    # Artifact vs genuine: weak ρ should track high adjacent redundancy if it's an artifact.
    apply(ax_sc)
    m = np.isfinite(rho) & np.isfinite(adj)
    sc = ax_sc.scatter(adj[m], np.abs(rho[m]), s=28, c=uniq[m], cmap="viridis",
                       edgecolors="0.2", linewidths=0.4)
    cbar = fig.colorbar(sc, ax=ax_sc, fraction=0.046, pad=0.04)
    cbar.set_label("# unique local-res voxel values", fontsize=6)
    cbar.ax.tick_params(labelsize=6)
    ax_sc.set_xlabel("adjacent-residue local-res redundancy ρ\n(→1 = axis barely varies per residue)")
    ax_sc.set_ylabel("|ρ(B, local resolution)|")
    ax_sc.set_title("Artifact vs genuine decoupling\n(artifact ⇒ upper-right empty)", fontsize=8)
    label_panel(ax_sc, "b")

    # Bad maps: B-spread vs local-res-spread.
    apply(ax_bad)
    bad = [r for r in records if r["emdb_id"] in BAD_MAPS]
    if bad:
        xb = np.arange(len(bad))
        w = 0.38
        b_cv = [r["b_cv"] for r in bad]
        lr_cv = [r["locres_cv"] for r in bad]
        ax_bad.bar(xb - w / 2, b_cv, w, label="B-factor CV", color=PALETTES["categorical"][0],
                   edgecolor="0.2", linewidth=0.4)
        ax_bad.bar(xb + w / 2, lr_cv, w, label="local-res CV", color=PALETTES["categorical"][2],
                   edgecolor="0.2", linewidth=0.4)
        ax_bad.set_xticks(xb)
        ax_bad.set_xticklabels([cohort_figure_label(r["emdb_id"], names=names) for r in bad], fontsize=7)
        ax_bad.set_ylabel("coefficient of variation")
        ax_bad.legend(fontsize=6)
        ax_bad.set_title("Flagged maps: where is the signal missing?\n(low B-CV ⇒ no disorder signal)", fontsize=8)
    label_panel(ax_bad, "c")

    fig.suptitle("Residue-level B-factor / local-resolution decoupling diagnostics", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = out_dir / "tv_decoupling_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_nature(fig, out, dpi=dpi)
    plt.close(fig)
    return out.with_suffix(".png")


def _write_csv(records: list[dict], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "tv_decoupling_diagnostics.csv"
    fieldnames = list(records[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for rec in records:
            w.writerow(
                {k: (f"{v:.6f}" if isinstance(v, float) and np.isfinite(v) else v) for k, v in rec.items()}
            )
    return path


def _tv_scatter(emd_ids: list[str], manifest: Path, out_dir: Path, dpi: int) -> Path:
    """Raw (T_vW, V) hexbin per representative map — confirm the relationship is a line."""
    from cryoem_mrc.tv_curvature import compute_map_tv_table

    n = len(emd_ids)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.4), squeeze=False)
    for j, eid in enumerate(emd_ids):
        ax = axes[0][j]
        apply(ax)
        df = compute_map_tv_table(eid, manifest=manifest)
        if "in_contour_mask" in df.columns:
            df = df[df["in_contour_mask"].astype(bool)]
        t = df["T_vonweizsacker"].to_numpy(dtype=np.float64)
        vv = df["V_curvature"].to_numpy(dtype=np.float64)
        m = np.isfinite(t) & np.isfinite(vv) & (t > 0) & (vv > 0)
        t, vv = t[m], vv[m]
        rho = float(stats.spearmanr(t, vv).statistic) if t.size > 10 else float("nan")
        hb = ax.hexbin(np.log10(t), np.log10(vv), gridsize=50, mincnt=1, cmap="viridis", bins="log")
        fig.colorbar(hb, ax=ax, label="log(count)", fraction=0.046, pad=0.02)
        ax.set_xlabel(r"$\log_{10} T_\mathrm{vW}=|\nabla\rho|^2$")
        ax.set_ylabel(r"$\log_{10} V=\|H\|_F^2$")
        ax.set_title(
            f"{cohort_figure_label(eid, manifest=manifest)}  (n={t.size}, ρ={rho:+.3f})",
            fontsize=8,
        )
        label_panel(ax, chr(ord("a") + j))
    fig.suptitle("Raw (T, V) at Cα — a line, not a fan", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = out_dir / "tv_line_scatter"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_nature(fig, out, dpi=dpi)
    plt.close(fig)
    return out.with_suffix(".png")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.tv_scatter:
        path = _tv_scatter(args.tv_scatter, args.manifest, args.out_dir, args.dpi)
        print(f"[decouple] (T,V) scatter → {path}", flush=True)
        return 0

    if not args.all and not args.emd_id:
        print("Specify --emd-id, --all, or --tv-scatter", file=sys.stderr)
        return 2

    ids = _emd_ids_with_locres(args.manifest) if args.all else [args.emd_id.strip()]

    records: list[dict] = []
    for eid in ids:
        rec = _diagnose_one(eid, manifest=args.manifest, min_pairs=args.min_pairs)
        if rec is not None:
            records.append(rec)

    if not records:
        print("[decouple] no usable maps", file=sys.stderr)
        return 2

    path = _write_csv(records, args.out_dir)
    summary = _build_summary(records)
    (args.out_dir / "tv_decoupling_diagnostics.json").write_text(json.dumps(summary, indent=2) + "\n")
    fig_path = _build_figure(records, summary, args.out_dir, args.dpi, manifest=args.manifest)

    print(f"[decouple] {len(records)} maps → {path}", flush=True)
    print(f"[decouple] figure → {fig_path}", flush=True)
    print(
        f"[decouple] median ρ(B,locres)={summary['median_rho_b_vs_locres']:+.3f} | "
        f"artifact_signature={summary['granularity_artifact_signature']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
