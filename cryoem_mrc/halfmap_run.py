"""Run reliability + build-zone export from two half-maps only."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .analysis import build_contour_mask, suggest_contour
from .emringer_cohort import BUILDING_REGIME_MAX_RESOLUTION_A
from .io import save_volume_like_reference
from .local_fsc import estimate_global_halfmap_fsc_resolution
from .map_grid import FullHalfMapBundle, load_full_and_half_maps
from .reliability_driver import main as reliability_main


@dataclass(frozen=True)
class HalfmapPairContext:
    """Loaded half-map pair kept in memory for interactive follow-up."""

    half1: Path
    half2: Path
    bundle: FullHalfMapBundle
    avg: np.ndarray


@dataclass(frozen=True)
class HalfmapPairSummary:
    """Preflight summary shown in interactive mode before the long pipeline."""

    suggested_contour: float
    resolution_a: float
    in_building_regime: bool
    n_mask_voxels: int
    voxel_size_a: float


def load_halfmap_pair_context(half1: Path, half2: Path) -> HalfmapPairContext:
    """Load and align two half-maps; average them on the half-map 1 grid."""
    half1 = Path(half1).expanduser().resolve()
    half2 = Path(half2).expanduser().resolve()
    bundle = load_full_and_half_maps(
        half1,
        half1,
        half2,
        reference="half1",
        dtype=np.float32,
        resample_if_needed=True,
    )
    for name, rep in bundle.reports.items():
        if not rep.ok:
            print(
                f"[cryoridge] resampled {name} onto half-map 1 grid",
                flush=True,
            )
    avg = (0.5 * (bundle.half1.data + bundle.half2.data)).astype(np.float32)
    return HalfmapPairContext(half1=half1, half2=half2, bundle=bundle, avg=avg)


def write_avg_half_map(ctx: HalfmapPairContext, out_dir: Path) -> Path:
    """Write averaged half-maps for ChimeraX contour picking and feature extraction."""
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    avg_path = out_dir / "avg_half.mrc"
    save_volume_like_reference(ctx.half1, ctx.avg, avg_path)
    return avg_path


def feature_start_threshold(avg: np.ndarray, ref: np.ndarray, contour: float) -> float:
    """
    ``--start-threshold`` for feature extraction when the contour mask is defined
    on a deposited primary map but features are computed from averaged half-maps.
    """
    from .analysis import build_contour_mask

    mask = build_contour_mask(ref, contour)
    if not mask.any():
        return float(contour)
    avg_max_in_mask = float(np.max(avg[mask]))
    if avg_max_in_mask < contour:
        return 0.0
    if avg_max_in_mask < contour * 1.05:
        return 0.0
    return float(contour)


def summarize_halfmap_pair(ctx: HalfmapPairContext) -> HalfmapPairSummary:
    """Estimate auto-contour and masked global FSC resolution for interactive prompts."""
    suggested = suggest_contour(ctx.avg)
    mask = build_contour_mask(ctx.avg, suggested)
    vox = float(np.mean(ctx.bundle.half1.voxel_size_zyx))
    res_a = estimate_global_halfmap_fsc_resolution(
        ctx.bundle.half1.data,
        ctx.bundle.half2.data,
        voxel_size_a=vox,
        mask=mask,
    )
    in_building = math.isfinite(res_a) and res_a <= BUILDING_REGIME_MAX_RESOLUTION_A
    return HalfmapPairSummary(
        suggested_contour=suggested,
        resolution_a=res_a,
        in_building_regime=in_building,
        n_mask_voxels=int(mask.sum()),
        voxel_size_a=vox,
    )


def run_cryoridge(
    half1: Path,
    half2: Path,
    *,
    out_dir: Path | None = None,
    contour: float,
    reference_map: Path | None = None,
    context: HalfmapPairContext | None = None,
) -> dict[str, Path | float]:
    """
    Features + reliability from two half-maps.

    ``contour`` is the ChimeraX Volume Viewer level on ``reference_map`` (deposited
    primary when provided, otherwise ``avg_half.mrc``). Uses half-map 1 as the
    alignment grid and exports ``{stem}_reliability.mrc`` and ``{stem}_build_zones.mrc``.
    """
    from cryoem_mrc.__main__ import main as features_main
    from cryoem_mrc.io import load_mrc

    if context is None:
        context = load_halfmap_pair_context(half1, half2)
    else:
        half1 = context.half1
        half2 = context.half2

    if out_dir is None:
        out_dir = half1.parent / "cryoridge_out"
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    avg_path = write_avg_half_map(context, out_dir)
    contour_val = float(contour)

    if reference_map is None:
        mask_ref_path = avg_path
        mask_volume = context.avg
        feature_thr = 0.0
    else:
        mask_ref_path = Path(reference_map).expanduser().resolve()
        if not mask_ref_path.is_file():
            raise FileNotFoundError(f"reference map not found: {mask_ref_path}")
        mask_volume = load_mrc(mask_ref_path, dtype=np.float32)
        feature_thr = feature_start_threshold(context.avg, mask_volume, contour_val)

    n_mask = int(build_contour_mask(mask_volume, contour_val).sum())
    if n_mask == 0:
        raise ValueError(
            f"contour mask is empty at contour={contour_val:g} on {mask_ref_path.name}; "
            "lower the ChimeraX level or open the correct map"
        )
    print(
        f"[cryoridge] contour {contour_val:.6g} on {mask_ref_path.name} "
        f"({n_mask:,} voxels in mask)",
        flush=True,
    )

    features_path = out_dir / "features.npz"
    rc = features_main(
        [
            str(avg_path),
            "--float32",
            "--out",
            str(features_path),
            "--reference",
            str(mask_ref_path),
            "--contour",
            str(contour_val),
            "--start-threshold",
            str(feature_thr),
        ]
    )
    if rc != 0:
        raise RuntimeError(f"feature extraction failed (exit {rc})")

    label = half1.stem
    rc = reliability_main(
        [
            "--reference",
            str(mask_ref_path),
            "--half1",
            str(half1),
            "--half2",
            str(half2),
            "--features",
            str(features_path),
            "--contour",
            str(contour_val),
            "--out-dir",
            str(out_dir),
            "--label",
            label,
        ]
    )
    if rc != 0:
        raise RuntimeError(f"reliability export failed (exit {rc})")

    return {
        "out_dir": out_dir,
        "avg_half": avg_path,
        "reference_map": mask_ref_path,
        "features": features_path,
        "reliability_mrc": out_dir / f"{label}_reliability.mrc",
        "build_zones_mrc": out_dir / f"{label}_build_zones.mrc",
        "contour": contour_val,
    }
