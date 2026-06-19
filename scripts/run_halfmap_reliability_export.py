"""Thesis bundle: half-map reliability zones, build zones, figures, and write-up (EMD-49450 defaults).

Generates under ``outputs/emd_<ID>/halfmap_reliability/``:

- ``reliability.npz`` — reliability_score, constraint V (legacy key reliability_H_repro), build_zone
- ``*.mrc`` — volume overlays on deposited reference grid
- ``figures/model_building_row.png`` — local resolution, reliability score, build zones, locres–reliability disagreement map
- ``../analysis/figures/analysis_validation_panel.png`` — anchor map only (2×2 validation)
- ``HALFMAP_RELIABILITY_RESULTS.md`` — per-map results (methods in docs/HALFMAP_RELIABILITY.md)

Example::

    source .venv/bin/activate
    PYTHONUNBUFFERED=1 python scripts/run_halfmap_reliability_export.py
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from style.nature import apply, label_panel, savefig as save_nature
from style.thesis_palette import (
    RELIABILITY_CMAP_DISAGREEMENT,
    RELIABILITY_CMAP_LOCRES,
    RELIABILITY_CMAP_SCORE,
)
from scipy import stats

from cryoem_mrc.analysis import (
    build_contour_mask,
    compute_feature_target_correlations,
    plot_analysis_validation_panel,
)
from cryoem_mrc.half_map_repro import (
    WINDOWED_HALFMAP_CORRELATION_KEY,
    WINDOWED_HALFMAP_CORRELATION_LABEL,
    load_windowed_halfmap_correlation,
)
from cryoem_mrc.density_source import DensitySource, rho_normalized_for_reliability
from cryoem_mrc.figure_cleanup import prune_halfmap_reliability_retired_figures
from cryoem_mrc.io import load_mrc
from cryoem_mrc.pipeline import load_feature_maps
from cryoem_mrc.map_grid import load_full_and_half_maps
from cryoem_mrc.reliability import (
    BUILD_ZONE_LABELS,
    attach_reliability_to_features,
    build_zone_colormap,
    save_build_zone_mrc,
    save_reliability_mrc,
)
from cryoem_mrc.local_resolution_io import load_local_resolution_map
from cryoem_mrc.repo_paths import (
    ANCHOR_EMDB_ID,
    DATA_ROOT,
    analysis_dir,
    avg_features_npz_path,
    find_features_npz,
    halfmap_metrics_npz,
    halfmap_reliability_dir,
    locres_blocres_mrc,
    primary_features_npz_path,
)
from cryoem_mrc.mask_bbox import (
    bbox_from_mask,
    crop_array,
    embed_array,
    format_bbox_log,
    pad_voxels_for_filters,
)
from cryoem_mrc.thesis_figures import (
    _locres_robust_limits,
    extract_slice,
    mask_slice_values,
    pick_slice_index,
    slice_crop_from_mask,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=DATA_ROOT / "emd_49450-mgtA_e2p+e1")
    p.add_argument("--emd-id", type=str, default="49450")
    p.add_argument("--reference", type=Path, default=None)
    p.add_argument("--half1", type=Path, default=None)
    p.add_argument("--half2", type=Path, default=None)
    p.add_argument("--features", type=Path, default=None)
    p.add_argument("--halfmap-npz", type=Path, default=halfmap_metrics_npz("49450"))
    p.add_argument("--contour", type=float, default=0.116)
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--out-dir", type=Path, default=halfmap_reliability_dir("49450"))
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--zoom-padding", type=int, default=24)
    p.add_argument(
        "--no-crop-to-contour",
        action="store_true",
        help="Compute constraint V on the full grid (default: tight bbox around contour mask)",
    )
    p.add_argument(
        "--write-analysis-panel",
        action="store_true",
        help=f"Write 2×2 validation panel under analysis/figures/ (default for EMD-{ANCHOR_EMDB_ID})",
    )
    p.add_argument(
        "--no-write-analysis-panel",
        action="store_true",
        help="Skip anchor validation panel even for the canonical anchor map",
    )
    p.add_argument(
        "--density-source",
        choices=("avg_half", "primary"),
        default="avg_half",
        help="ρ for constraint V: avg_half (default, matched to half-map CC) or primary (sensitivity)",
    )
    p.add_argument(
        "--prune-retired-figures",
        action="store_true",
        help="Delete orphaned spearman/binned/bfactor reliability figure exports in figures/",
    )
    p.add_argument(
        "--figures-only",
        action="store_true",
        help="Regenerate figures from existing reliability.npz (skip map recomputation)",
    )
    return p.parse_args(argv)


def _paths(args: argparse.Namespace) -> dict[str, Path]:
    d = args.data_dir
    emd = f"emd_{args.emd_id}"
    source: DensitySource = args.density_source
    default_features = (
        avg_features_npz_path(d, args.emd_id, args.contour)
        if source == "avg_half"
        else primary_features_npz_path(d, args.emd_id, args.contour)
    )
    return {
        "reference": args.reference or d / f"{emd}.map",
        "half1": args.half1 or d / f"{emd}_half_map_1.map",
        "half2": args.half2 or d / f"{emd}_half_map_2.map",
        "features": (
            args.features
            or find_features_npz(d, args.emd_id, args.contour, density_source=source)
            or default_features
        ),
    }


def _load_local_var(features_path: Path) -> np.ndarray:
    with np.load(features_path, allow_pickle=False) as d:
        return np.asarray(d["local_variance"], dtype=np.float32)


def _optional_density_normalized(features_path: Path) -> np.ndarray | None:
    feats = load_feature_maps(features_path)
    if "density_normalized" not in feats:
        return None
    return np.asarray(feats["density_normalized"], dtype=np.float32)


def _load_local_resolution_array(emd_id: str) -> np.ndarray:
    """BlocRes local-resolution map on the deposited reference grid."""
    locres_path = locres_blocres_mrc(emd_id)
    if not locres_path.is_file():
        raise FileNotFoundError(f"EMD-{emd_id}: missing {locres_path}")
    return np.asarray(load_local_resolution_map(locres_path, source="blocres").data, dtype=np.float32)


def _normalized_locres_quality(
    local_resolution: np.ndarray,
    mask: np.ndarray,
    *,
    lo_pct: float = 5.0,
    hi_pct: float = 95.0,
) -> np.ndarray:
    """In-mask locres mapped to 0–1 quality (1 = sharpest / lowest Å)."""
    vals = local_resolution[mask]
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return np.zeros_like(local_resolution, dtype=np.float32)
    lo, hi = np.percentile(finite, [lo_pct, hi_pct])
    if hi <= lo:
        hi = lo + 1e-6
    blurry = np.clip((local_resolution - lo) / (hi - lo), 0.0, 1.0)
    quality = (1.0 - blurry).astype(np.float32)
    return np.where(mask, quality, 0.0).astype(np.float32)


def _locres_reliability_disagreement(
    local_resolution: np.ndarray,
    reliability_score: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """|normalized locres quality − reliability score| inside the mask."""
    quality = _normalized_locres_quality(local_resolution, mask)
    rel = np.where(mask, reliability_score, 0.0).astype(np.float32)
    return np.abs(quality - rel).astype(np.float32)


def _plot_build_zones(
    ax,
    zones_sl: np.ndarray,
    mask_sl: np.ndarray,
    *,
    title: str,
    cax=None,
) -> plt.cm.ScalarMappable:
    apply(ax)
    show = np.ma.masked_where(~mask_sl, zones_sl.astype(float))
    cmap_obj = build_zone_colormap().copy()
    cmap_obj.set_bad(color=(0.12, 0.12, 0.14, 1.0))
    im = ax.imshow(show, cmap=cmap_obj, vmin=0, vmax=2, origin="lower")
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    if cax is not None:
        cb = plt.colorbar(im, cax=cax, orientation="horizontal", ticks=[0, 1, 2])
        cb.ax.set_xticklabels([BUILD_ZONE_LABELS[z] for z in (0, 1, 2)])
        cb.ax.tick_params(labelsize=8)
    return im


def _plot_continuous_panel(
    ax,
    sl: np.ndarray,
    mask_sl: np.ndarray,
    *,
    cmap,
    title: str,
    vmin: float,
    vmax: float,
    cax=None,
) -> plt.cm.ScalarMappable:
    """Masked slice with optional horizontal colorbar on ``cax``."""
    apply(ax)
    masked = mask_slice_values(sl, mask_sl)
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color=(0.12, 0.12, 0.14, 1.0))
    im = ax.imshow(masked, cmap=cmap_obj, vmin=vmin, vmax=vmax, origin="lower")
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    if cax is not None:
        cb = plt.colorbar(im, cax=cax, orientation="horizontal")
        cb.set_ticks([vmin, vmax])
        cb.ax.tick_params(labelsize=8)
    return im


def _write_model_building_row_figure(
    *,
    emd_id: str,
    contour: float,
    mask: np.ndarray,
    local_resolution: np.ndarray,
    reliability_score: np.ndarray,
    zones: np.ndarray,
    fig_dir: Path,
    dpi: int,
    zoom_padding: int,
) -> Path:
    """Write local resolution / reliability / build-zone slices plus disagreement map."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    z = pick_slice_index(mask, axis=0)
    msl = mask[z]
    crop = slice_crop_from_mask(msl, pad_voxels=zoom_padding) if zoom_padding else None

    def _crop_2d(arr: np.ndarray) -> np.ndarray:
        if crop is None:
            return arr
        return arr[crop[0]:crop[1], crop[2]:crop[3]]

    disagreement = _locres_reliability_disagreement(local_resolution, reliability_score, mask)
    loc_sl = _crop_2d(extract_slice(local_resolution, axis=0, index=z))
    rel_sl = _crop_2d(extract_slice(reliability_score, axis=0, index=z))
    zone_sl = _crop_2d(extract_slice(zones.astype(float), axis=0, index=z))
    diff_sl = _crop_2d(extract_slice(disagreement, axis=0, index=z))
    m_c = _crop_2d(msl)
    loc_lo, loc_hi = _locres_robust_limits(loc_sl, m_c)

    fig = plt.figure(figsize=(15.5, 4.8), facecolor="white")
    gs = fig.add_gridspec(
        2,
        4,
        height_ratios=[1, 0.07],
        hspace=0.38,
        wspace=0.16,
        left=0.03,
        right=0.99,
        top=0.86,
        bottom=0.10,
    )

    panels = [
        ("a", loc_sl, RELIABILITY_CMAP_LOCRES, "local resolution (Å)", loc_lo, loc_hi),
        ("b", rel_sl, RELIABILITY_CMAP_SCORE, "reliability score", 0.0, 1.0),
        (
            "d",
            diff_sl,
            RELIABILITY_CMAP_DISAGREEMENT,
            "locres vs reliability\n|Δ|",
            0.0,
            1.0,
        ),
    ]
    panel_cols = (0, 1, 3)
    for col, (letter, sl, cmap, title, vmin, vmax) in zip(panel_cols, panels):
        ax = fig.add_subplot(gs[0, col])
        cax = fig.add_subplot(gs[1, col])
        _plot_continuous_panel(
            ax,
            sl,
            m_c,
            cmap=cmap,
            title=f"{title}\nZ = {z}",
            vmin=vmin,
            vmax=vmax,
            cax=cax,
        )
        label_panel(ax, letter)

    ax_z = fig.add_subplot(gs[0, 2])
    cax_z = fig.add_subplot(gs[1, 2])
    _plot_build_zones(ax_z, zone_sl, m_c, title=f"build zones\nZ = {z}", cax=cax_z)
    label_panel(ax_z, "c")

    fig.suptitle(f"EMD-{emd_id} model-building guidance (mask ρ≥{contour})", fontsize=12)
    out = fig_dir / "model_building_row.png"
    save_nature(fig, out, dpi=dpi)
    plt.close(fig)
    return out


def _write_thesis_md(
    path: Path,
    *,
    emd_id: str,
    contour: float,
    n_mask: int,
    spearman: dict[str, float],
    partial: dict[str, float],
    zone_counts: dict[int, int],
    paths: dict[str, Path],
) -> None:
    rel_s = spearman.get("reliability_score", float("nan"))
    var_s = spearman.get("local_variance", float("nan"))
    v_s = spearman.get("reliability_H_repro", float("nan"))
    text = f"""# Half-map reliability zones — EMD-{emd_id}

Per-map results bundle. **Methods:** [docs/HALFMAP_RELIABILITY.md](docs/HALFMAP_RELIABILITY.md).

Mask: deposited reference at ρ ≥ {contour}. In-mask voxels: **{n_mask:,}**.

---

## Results (ρ_ref ≥ {contour})

| Feature | Spearman ρ vs windowed half-map correlation |
|---------|---------------------------|
| local_variance | {var_s:+.4f} |
| constraint V (legacy key reliability_H_repro) | {v_s:+.4f} |
| **reliability_score** | **{rel_s:+.4f}** |

Partial Spearman vs CC controlling for local_variance:

| Feature | Partial ρ |
|---------|-----------|
| reliability_score | {partial.get('reliability_score', float('nan')):+.4f} |
| constraint V | {partial.get('reliability_H_repro', float('nan')):+.4f} |

**Zone counts (in-mask voxels):**

| Zone | Count |
|------|------:|
| 0 omit | {zone_counts.get(0, 0):,} |
| 1 caution | {zone_counts.get(1, 0):,} |
| 2 build | {zone_counts.get(2, 0):,} |

**Interpretation:** `local_variance` remains the strongest single statistic (ρ ≈ {var_s:.2f}). Constraint **V** (ρ ≈ {v_s:.2f}) ranks placement difficulty; the in-mask percentile **reliability_score** is the default export for model-building guidance.

---

## Draft paragraphs (methods)

> We computed voxel-wise **reliability scores** from half-map–derived statistics: local features on ``0.5*(h₁+h₂)`` and constraint **V** on z-scored ``ρ = ½(h₁+h₂)`` (Decision 001). The exported **reliability_score** is the in-mask percentile rank of constraint V (higher = more reliable locally). Macromolecular voxels were selected with the EMDB-recommended contour ρ_ref ≥ {contour} on the **deposited primary map** (Decision 002). Reliability and build-zone **MRC overlays** are written on that deposited grid for model-building visualization. **Build zones** (omit / caution / build) were assigned by terciles of reliability_score inside this mask.

## Draft paragraphs (results)

> On EMD-{emd_id} ({n_mask:,} in-mask voxels), reliability_score correlated with windowed half-map cross-correlation at Spearman ρ = {rel_s:.2f}, comparable to constraint V (ρ = {v_s:.2f}) and below local variance (ρ = {var_s:.2f}). Zones labeled **build** ({zone_counts.get(2, 0):,} voxels) mark regions where independent half-maps agree and local statistics support confident model placement; **omit** zones ({zone_counts.get(0, 0):,} voxels) flag areas where the map should not be over-interpreted. We treat these labels as **map-quality guidance**, not biophysical flexibility measurements.

## Draft paragraphs (discussion / limitations)

> Reliability scoring identifies trustworthy regions **inside** the density contour. Flexible segments below the contour or absent from the map are invisible to this analysis. Future work should test build-zone transfer across a multi-map cohort and compare against deposited models residue-by-residue when PDB coordinates are available.

---

## Files generated

| File | Description |
|------|-------------|
| `reliability.npz` | reliability_score, constraint V, build_zone |
| `emd_{emd_id}_reliability.mrc` | Reliability overlay (0–1 score) |
| `emd_{emd_id}_build_zones.mrc` | 0/1/2 zone labels |
| `figures/model_building_row.png` | Local resolution, reliability score, build zones, locres–reliability disagreement map |
| `../analysis/figures/analysis_validation_panel.png` | Anchor map: 2×2 variance / reliability validation |
| `run_metadata.json` | Spearman / partial ρ (cohort heatmap reads this) |

**Inputs:** `{paths['reference'].name}`, `{paths['features'].name}`, half-maps, `{paths.get('halfmap_npz', 'halfmap_metrics.npz')}`.

See also: `docs/HALFMAP_RELIABILITY.md`, `docs/STATISTICS_METHODS.md`.
"""
    path.write_text(text)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    paths = _paths(args)
    out_dir = args.out_dir
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    log = "[halfmap_reliability]"
    for k, p in paths.items():
        if not p.exists():
            print(f"{log} ERROR: missing {k}: {p}", file=sys.stderr)
            return 2
    if not args.figures_only and not args.halfmap_npz.exists():
        print(f"{log} ERROR: missing {args.halfmap_npz}", file=sys.stderr)
        return 2

    npz_path = out_dir / "reliability.npz"
    if args.figures_only:
        if not npz_path.is_file():
            print(f"{log} ERROR: --figures-only requires {npz_path}", file=sys.stderr)
            return 2
        print(f"{log} figures-only: loading cached volumes", flush=True)
        reference = load_mrc(paths["reference"], dtype=np.float32)
        with np.load(npz_path, allow_pickle=False) as rel:
            reliability_score = np.asarray(rel["reliability_score"])
            zones = np.asarray(rel["build_zone"])
            contour = float(rel["contour"]) if "contour" in rel else args.contour
        mask = build_contour_mask(reference, contour)
        local_resolution = _load_local_resolution_array(args.emd_id)
        out = _write_model_building_row_figure(
            emd_id=args.emd_id,
            contour=contour,
            mask=mask,
            local_resolution=local_resolution,
            reliability_score=reliability_score,
            zones=zones,
            fig_dir=fig_dir,
            dpi=args.dpi,
            zoom_padding=args.zoom_padding,
        )
        print(f"{log} wrote {out}", flush=True)
        if args.prune_retired_figures:
            removed = prune_halfmap_reliability_retired_figures(fig_dir)
            if removed:
                print(f"{log} pruned {len(removed)} retired figure(s)", flush=True)
        return 0

    print(f"{log} loading reference + mask", flush=True)
    reference = load_mrc(paths["reference"], dtype=np.float32)
    mask = build_contour_mask(reference, args.contour)
    n_mask = int(mask.sum())
    print(f"{log} mask {n_mask:,} voxels at contour {args.contour}", flush=True)

    local_var = _load_local_var(paths["features"])
    bundle = load_full_and_half_maps(
        paths["reference"], paths["half1"], paths["half2"],
        reference="full", dtype=np.float32, resample_if_needed=True,
    )
    rho = rho_normalized_for_reliability(
        source=args.density_source,
        half1=bundle.half1.data,
        half2=bundle.half2.data,
        features_density_normalized=_optional_density_normalized(paths["features"]),
        primary_volume=reference if args.density_source == "primary" else None,
    )
    full_shape = reference.shape
    pad = pad_voxels_for_filters(window=args.window)
    if args.no_crop_to_contour:
        work: dict[str, np.ndarray] = {"density_normalized": rho, "local_variance": local_var}
        attach_reliability_to_features(
            work, bundle.half1.data, bundle.half2.data, window=args.window, mask=mask
        )
        feats = work
    else:
        bbox = bbox_from_mask(mask, pad=pad)
        print(
            f"{log} contour crop: {format_bbox_log(bbox, full_shape, pad=pad)}",
            flush=True,
        )
        work = {
            "density_normalized": crop_array(rho, bbox),
            "local_variance": crop_array(local_var, bbox),
        }
        attach_reliability_to_features(
            work,
            crop_array(bundle.half1.data, bbox),
            crop_array(bundle.half2.data, bbox),
            window=args.window,
            mask=crop_array(mask, bbox),
        )
        rel_keys = (
            "reliability_score",
            "reliability_H_repro",
            "reliability_fluctuation",
            "reliability_smoothness",
            "build_zone",
        )
        feats = {
            k: embed_array(full_shape, bbox, work[k], dtype=work[k].dtype)
            for k in rel_keys
        }
    del bundle
    gc.collect()

    with np.load(args.halfmap_npz, allow_pickle=False) as hm:
        cc = load_windowed_halfmap_correlation(hm)

    compare = {
        "reliability_score": feats["reliability_score"],
        "reliability_H_repro": feats["reliability_H_repro"],
        "local_variance": local_var,
    }
    result = compute_feature_target_correlations(
        compare,
        cc,
        mask,
        target_name=WINDOWED_HALFMAP_CORRELATION_KEY,
        methods=("spearman",),
        max_samples=2_000_000,
    )
    spearman = {c.feature_name: c.correlation for c in result.correlations}

    # Partial vs variance
    idx = np.flatnonzero(mask)
    y = cc.ravel()[idx]
    ctrl = local_var.ravel()[idx]
    partial: dict[str, float] = {}

    def _partial(x, y, z):
        xr, yr, zr = stats.rankdata(x), stats.rankdata(y), stats.rankdata(z)
        r_xy = np.corrcoef(xr, yr)[0, 1]
        r_xz = np.corrcoef(xr, zr)[0, 1]
        r_yz = np.corrcoef(yr, zr)[0, 1]
        d = (1 - r_xz * r_xz) * (1 - r_yz * r_yz)
        return (r_xy - r_xz * r_yz) / np.sqrt(d) if d > 0 else float("nan")

    for name in ("reliability_score", "reliability_H_repro"):
        partial[name] = float(_partial(compare[name].ravel()[idx], y, ctrl))

    zones = feats["build_zone"]
    zone_counts = {int(z): int((zones[mask] == z).sum()) for z in (0, 1, 2)}

    # Save NPZ + MRC
    np.savez_compressed(
        out_dir / "reliability.npz",
        reliability_score=feats["reliability_score"],
        reliability_H_repro=feats["reliability_H_repro"],
        reliability_fluctuation=feats["reliability_fluctuation"],
        reliability_smoothness=feats["reliability_smoothness"],
        build_zone=zones,
        contour=np.float32(args.contour),
        emd_id=np.array(args.emd_id),
    )
    save_reliability_mrc(paths["reference"], feats["reliability_score"], out_dir / f"emd_{args.emd_id}_reliability.mrc")
    save_build_zone_mrc(paths["reference"], zones, out_dir / f"emd_{args.emd_id}_build_zones.mrc")
    (out_dir / "run_metadata.json").write_text(
        json.dumps({"spearman": spearman, "partial": partial, "zone_counts": zone_counts, "n_mask": n_mask}, indent=2) + "\n"
    )

    # Figures
    local_resolution = _load_local_resolution_array(args.emd_id)
    out = _write_model_building_row_figure(
        emd_id=args.emd_id,
        contour=args.contour,
        mask=mask,
        local_resolution=local_resolution,
        reliability_score=feats["reliability_score"],
        zones=zones,
        fig_dir=fig_dir,
        dpi=args.dpi,
        zoom_padding=args.zoom_padding,
    )
    print(f"{log} wrote {out}", flush=True)

    write_panel = (
        args.write_analysis_panel
        or (str(args.emd_id).strip() == ANCHOR_EMDB_ID and not args.no_write_analysis_panel)
    )
    if write_panel:
        feature_maps = load_feature_maps(paths["features"])
        panel_path = analysis_dir(args.emd_id) / "figures" / "analysis_validation_panel.png"
        plot_analysis_validation_panel(
            feature_maps,
            {WINDOWED_HALFMAP_CORRELATION_KEY: cc},
            mask,
            reliability_score=feats["reliability_score"],
            spearman=spearman,
            emd_id=str(args.emd_id),
            contour=args.contour,
            save_path=panel_path,
            dpi=args.dpi,
        )
        print(f"{log} wrote {panel_path}", flush=True)

    if args.prune_retired_figures:
        removed = prune_halfmap_reliability_retired_figures(fig_dir)
        if removed:
            print(f"{log} pruned {len(removed)} retired figure(s)", flush=True)

    paths_meta = {**paths, "halfmap_npz": args.halfmap_npz}
    _write_thesis_md(
        out_dir / "HALFMAP_RELIABILITY_RESULTS.md",
        emd_id=args.emd_id,
        contour=args.contour,
        n_mask=n_mask,
        spearman=spearman,
        partial=partial,
        zone_counts=zone_counts,
        paths=paths_meta,
    )

    print(f"{log} wrote {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
