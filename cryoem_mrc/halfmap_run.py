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
    contour: float | None = None,
    context: HalfmapPairContext | None = None,
) -> dict[str, Path | float]:
    """
    Features + reliability from two half-maps.

    Uses half-map 1 as the reference grid, writes an averaged map, picks or
    accepts a contour, and exports ``{stem}_reliability.mrc`` and
    ``{stem}_build_zones.mrc``.
    """
    from cryoem_mrc.__main__ import main as features_main

    if context is None:
        context = load_halfmap_pair_context(half1, half2)
    else:
        half1 = context.half1
        half2 = context.half2

    if out_dir is None:
        out_dir = half1.parent / "cryoridge_out"
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    avg = context.avg
    ref_path = out_dir / "avg_half.mrc"
    save_volume_like_reference(half1, avg, ref_path)

    auto_contour = contour is None
    contour_val = float(suggest_contour(avg) if auto_contour else contour)
    n_mask = int(build_contour_mask(avg, contour_val).sum())
    if n_mask == 0:
        raise ValueError(
            f"contour mask is empty at contour={contour_val:g}; "
            "try a lower contour level"
        )
    label_kind = "auto" if auto_contour else "user"
    print(
        f"[cryoridge] {label_kind} contour {contour_val:.6g} ({n_mask:,} voxels in mask)",
        flush=True,
    )

    features_path = out_dir / "features.npz"
    rc = features_main(
        [
            str(ref_path),
            "--float32",
            "--out",
            str(features_path),
            "--reference",
            str(ref_path),
            "--contour",
            str(contour_val),
            "--start-threshold",
            "0",
        ]
    )
    if rc != 0:
        raise RuntimeError(f"feature extraction failed (exit {rc})")

    label = half1.stem
    rc = reliability_main(
        [
            "--reference",
            str(ref_path),
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
        "avg_half": ref_path,
        "features": features_path,
        "reliability_mrc": out_dir / f"{label}_reliability.mrc",
        "build_zones_mrc": out_dir / f"{label}_build_zones.mrc",
        "contour": contour_val,
    }
