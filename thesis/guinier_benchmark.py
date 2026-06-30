"""Per-map Guinier B / sharpening benchmark vs deposited primary map."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from scipy import stats

from cryoem_mrc.analysis import build_contour_mask
from cryoem_mrc.guinier_sharpening import (
    R_MIN_A_DEFAULT,
    apply_global_bfactor_sharpen,
    apply_local_bfactor_sharpen,
    estimate_global_guinier_b,
    estimate_local_guinier_b_map,
    load_external_bfactor_map,
    masked_map_ccc,
    summarize_b_map,
)
from cryoem_mrc.local_resolution_io import load_local_resolution_map, resample_local_resolution_onto_reference
from cryoem_mrc.map_grid import load_full_and_half_maps, load_map_grid, resample_volume_onto_grid
from cryoem_mrc.cohort_labels import cohort_figure_label, load_display_name_map
from cryoem_mrc.repo_paths import COHORT_MANIFEST, locres_blocres_mrc
from cryoem_mrc.structure_validation import (
    iter_ca_residues,
    load_cohort_manifest_row,
    physical_xyz_to_voxel_indices,
    sample_volume_at_ca,
)


@dataclass
class GuinierBenchmarkResult:
    emdb_id: str
    global_resolution_a: float
    b_global_guinier: float
    b_global_r_squared: float
    local_b_median_global_rmax: float
    local_b_iqr_global_rmax: float
    local_b_median_locres_rmax: float
    local_b_iqr_locres_rmax: float
    delta_median_global_rmax: float
    delta_median_locres_rmax: float
    ccc_sharp_global_vs_deposit: float
    ccc_sharp_local_global_rmax_vs_deposit: float
    ccc_sharp_local_locres_rmax_vs_deposit: float
    ccc_sharp_local_global_rmax_vs_global: float
    ccc_sharp_local_locres_rmax_vs_global: float
    rho_biso_vs_local_b_global_rmax: float
    rho_biso_vs_local_b_locres_rmax: float
    n_in_mask_ca: int
    has_locres: bool
    notes: str = ""


def _spearman_at_ca(
    b_map: np.ndarray,
    grid,
    residues,
    b_iso: np.ndarray,
    mask: np.ndarray,
    *,
    sphere_radius_a: float,
) -> float:
    sampled = sample_volume_at_ca(b_map, grid, residues, sphere_radius_a=sphere_radius_a)
    in_mask = []
    for res in residues:
        iz, iy, ix = physical_xyz_to_voxel_indices(res.x, res.y, res.z, grid)
        in_mask.append(
            0 <= iz < mask.shape[0]
            and 0 <= iy < mask.shape[1]
            and 0 <= ix < mask.shape[2]
            and bool(mask[iz, iy, ix])
        )
    m = np.asarray(in_mask, dtype=bool) & np.isfinite(b_iso) & np.isfinite(sampled)
    if int(m.sum()) < 10:
        return float("nan")
    if np.std(b_iso[m]) == 0.0 or np.std(sampled[m]) == 0.0:
        return float("nan")
    rho, _ = stats.spearmanr(b_iso[m], sampled[m])
    return float(rho)


def run_guinier_benchmark_one(
    emd_id: str,
    *,
    manifest: Path = COHORT_MANIFEST,
    r_min_a: float = R_MIN_A_DEFAULT,
    patch_size: int = 17,
    stride: int = 8,
    sphere_radius_a: float = 2.0,
    external_locbfactor_mrc: Path | None = None,
    skip_local_sharpen: bool = False,
) -> GuinierBenchmarkResult | None:
    """
    Compare Guinier B estimation and sharpening on avg-of-halves.

    Two local-B bands:
      - **global_rmax**: fit upper limit = manifest global resolution
      - **locres_rmax**: fit upper limit = BlocRes local resolution per patch

    Sharpening comparisons (distinct from atomic B-factors):
      - global B sharpen vs deposited primary
      - local B sharpen (global-rmax band) vs deposited primary
      - local B sharpen (locres-rmax band) vs deposited primary
    """
    row = load_cohort_manifest_row(manifest, emd_id)
    half1 = Path(row["half1_path"])
    half2 = Path(row["half2_path"])
    primary_path = Path(row["reference_mrc"])
    pdb_raw = row.get("flexibility_path_or_pdb", "").strip()
    contour = float(row["contour"])
    r_max_global = float(row.get("global_resolution_a") or 0.0)
    if r_max_global <= 0:
        return None

    for label, p in (("half1", half1), ("half2", half2), ("primary", primary_path)):
        if not p.is_file():
            return None

    bundle = load_full_and_half_maps(
        primary_path,
        half1,
        half2,
        dtype=np.float32,
        reference="half1",
        resample_if_needed=True,
    )
    h1 = bundle.half1
    h2 = bundle.half2
    primary = bundle.full
    avg = np.asarray(h1.data, dtype=np.float32) * 0.5 + np.asarray(h2.data, dtype=np.float32) * 0.5
    primary_on_grid = np.asarray(primary.data, dtype=np.float32)
    if primary_on_grid.shape != avg.shape:
        primary_on_grid = resample_volume_onto_grid(primary, h1).astype(np.float32, copy=False)
    # Depositor contour is defined on the sharpened primary; avg-of-halves is lower scale.
    mask = build_contour_mask(primary_on_grid, contour).astype(bool)
    vox = h1.voxel_size_zyx

    gfit = estimate_global_guinier_b(
        avg,
        vox,
        r_min_a=r_min_a,
        r_max_a=r_max_global,
        mask=mask,
    )
    b_global = gfit.b_factor

    local_b_global_rmax = estimate_local_guinier_b_map(
        avg,
        voxel_size_zyx=vox,
        r_min_a=r_min_a,
        r_max_a=r_max_global,
        patch_size=patch_size,
        stride=stride,
        mask=mask,
        r_max_mode="global",
    )

    locres_path = locres_blocres_mrc(emd_id)
    has_locres = locres_path.is_file()
    local_b_locres_rmax = np.full(avg.shape, np.nan, dtype=np.float32)
    locres_on_grid: np.ndarray | None = None
    if has_locres:
        loc_grid = load_local_resolution_map(locres_path, source="blocres", dtype=np.float32)
        locres_on_grid = np.asarray(
            resample_local_resolution_onto_reference(loc_grid, h1).data, dtype=np.float32
        )
        local_b_locres_rmax = estimate_local_guinier_b_map(
            avg,
            voxel_size_zyx=vox,
            r_min_a=r_min_a,
            r_max_a=locres_on_grid,
            patch_size=patch_size,
            stride=stride,
            mask=mask,
            r_max_mode="locres",
        )

    if external_locbfactor_mrc is not None and external_locbfactor_mrc.is_file():
        local_b_global_rmax = load_external_bfactor_map(external_locbfactor_mrc)

    summ_g = summarize_b_map(local_b_global_rmax, mask)
    summ_l = summarize_b_map(local_b_locres_rmax, mask) if has_locres else {
        "median": float("nan"),
        "iqr": float("nan"),
    }

    sharp_global = apply_global_bfactor_sharpen(
        avg, vox, b_global, r_min_a=r_min_a, r_max_a=r_max_global
    )
    primary_vol = primary_on_grid
    if skip_local_sharpen:
        sharp_local_g = sharp_global
        sharp_local_l = sharp_global
    else:
        sharp_local_g = apply_local_bfactor_sharpen(
            avg,
            local_b_global_rmax,
            voxel_size_zyx=vox,
            r_min_a=r_min_a,
            r_max_a=r_max_global,
            patch_size=patch_size,
            stride=stride,
            mask=mask,
        )
        if has_locres and locres_on_grid is not None:
            sharp_local_l = apply_local_bfactor_sharpen(
                avg,
                local_b_locres_rmax,
                voxel_size_zyx=vox,
                r_min_a=r_min_a,
                r_max_a=locres_on_grid,
                patch_size=patch_size,
                stride=stride,
                mask=mask,
            )
        else:
            sharp_local_l = sharp_local_g

    rho_b_g = float("nan")
    rho_b_l = float("nan")
    n_ca = 0
    if pdb_raw and Path(pdb_raw).is_file() and row.get("flexibility_source", "").strip() == "b_factor":
        residues = iter_ca_residues(Path(pdb_raw))
        b_iso = np.array([r.b_iso for r in residues], dtype=np.float64)
        rho_b_g = _spearman_at_ca(
            local_b_global_rmax, h1, residues, b_iso, mask, sphere_radius_a=sphere_radius_a
        )
        if has_locres:
            rho_b_l = _spearman_at_ca(
                local_b_locres_rmax, h1, residues, b_iso, mask, sphere_radius_a=sphere_radius_a
            )
        in_mask = []
        for res in residues:
            iz, iy, ix = physical_xyz_to_voxel_indices(res.x, res.y, res.z, h1)
            in_mask.append(
                0 <= iz < mask.shape[0]
                and 0 <= iy < mask.shape[1]
                and 0 <= ix < mask.shape[2]
                and bool(mask[iz, iy, ix])
            )
        n_ca = int(np.sum(in_mask))

    notes = ""
    if not np.isfinite(b_global):
        notes = "global Guinier fit failed"
    if not has_locres:
        notes = (notes + "; no BlocRes").strip("; ")

    return GuinierBenchmarkResult(
        emdb_id=str(emd_id),
        global_resolution_a=r_max_global,
        b_global_guinier=b_global,
        b_global_r_squared=gfit.r_squared,
        local_b_median_global_rmax=summ_g["median"],
        local_b_iqr_global_rmax=summ_g["iqr"],
        local_b_median_locres_rmax=summ_l["median"],
        local_b_iqr_locres_rmax=summ_l["iqr"],
        delta_median_global_rmax=summ_g["median"] - b_global if np.isfinite(summ_g["median"]) else float("nan"),
        delta_median_locres_rmax=summ_l["median"] - b_global if np.isfinite(summ_l["median"]) else float("nan"),
        ccc_sharp_global_vs_deposit=masked_map_ccc(sharp_global, primary_vol, mask),
        ccc_sharp_local_global_rmax_vs_deposit=masked_map_ccc(sharp_local_g, primary_vol, mask),
        ccc_sharp_local_locres_rmax_vs_deposit=masked_map_ccc(sharp_local_l, primary_vol, mask),
        ccc_sharp_local_global_rmax_vs_global=masked_map_ccc(sharp_local_g, sharp_global, mask),
        ccc_sharp_local_locres_rmax_vs_global=masked_map_ccc(sharp_local_l, sharp_global, mask),
        rho_biso_vs_local_b_global_rmax=rho_b_g,
        rho_biso_vs_local_b_locres_rmax=rho_b_l,
        n_in_mask_ca=n_ca,
        has_locres=has_locres,
        notes=notes,
    )


def result_to_dict(result: GuinierBenchmarkResult) -> dict:
    return asdict(result)


def plot_guinier_benchmark_summary(
    rows: Sequence[GuinierBenchmarkResult],
    out_path: Path,
    *,
    dpi: int = 300,
) -> Path:
    """Cohort bar chart: global vs local sharpen CCC vs deposited primary."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    from style.figures import apply, label_panel, savefig as save_nature

    dicts = [result_to_dict(r) for r in rows]
    df = pd.DataFrame(dicts).sort_values("global_resolution_a")
    emdb = df["emdb_id"].astype(str).tolist()
    g = df["ccc_sharp_global_vs_deposit"].to_numpy(dtype=float)
    l = df["ccc_sharp_local_global_rmax_vs_deposit"].to_numpy(dtype=float)
    x = np.arange(len(emdb))
    w = 0.38

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [2.2, 1]})
    ax = axes[0]
    apply(ax)
    ax.bar(x - w / 2, g, width=w, color="#4C72B0", label="Global sharpen")
    ax.bar(x + w / 2, l, width=w, color="#DD8452", label="Local sharpen")
    names = load_display_name_map(COHORT_MANIFEST)
    ax.set_xticks(x)
    ax.set_xticklabels([cohort_figure_label(e, names=names) for e in emdb], rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("CCC vs deposited primary")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Guinier sharpening recovery by map")
    label_panel(ax, "a")

    ax2 = axes[1]
    apply(ax2)
    meds = [
        float(np.nanmedian(g)),
        float(np.nanmedian(l)),
    ]
    ax2.bar(["Global", "Local"], meds, color=["#4C72B0", "#DD8452"])
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Cohort median CCC")
    ax2.set_title("Cohort summary")
    for i, v in enumerate(meds):
        ax2.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    label_panel(ax2, "b")

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_nature(fig, out_path, dpi=dpi)
    plt.close(fig)
    return out_path
