"""Run reliability + build-zone export from two half-maps only."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .analysis import build_contour_mask, suggest_contour
from .io import save_volume_like_reference
from .map_grid import load_full_and_half_maps
from .reliability_driver import main as reliability_main


def run_halfmap_qc(
    half1: Path,
    half2: Path,
    *,
    out_dir: Path | None = None,
) -> dict[str, Path | float]:
    """
    Features + reliability from two half-maps.

    Uses half-map 1 as the reference grid, writes an averaged map, auto-picks
    contour, and exports ``{stem}_reliability.mrc`` and ``{stem}_build_zones.mrc``.
    """
    from cryoem_mrc.__main__ import main as features_main

    half1 = Path(half1).expanduser().resolve()
    half2 = Path(half2).expanduser().resolve()
    if out_dir is None:
        out_dir = half1.parent / "halfmap_qc_out"
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

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
                f"[halfmap-qc] resampled {name} onto half-map 1 grid",
                flush=True,
            )

    avg = (0.5 * (bundle.half1.data + bundle.half2.data)).astype(np.float32)
    ref_path = out_dir / "avg_half.mrc"
    save_volume_like_reference(half1, avg, ref_path)

    contour = suggest_contour(avg)
    n_mask = int(build_contour_mask(avg, contour).sum())
    print(
        f"[halfmap-qc] auto contour {contour:.6g} ({n_mask:,} voxels in mask)",
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
            str(contour),
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
            str(contour),
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
        "contour": contour,
    }
