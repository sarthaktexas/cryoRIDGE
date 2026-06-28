"""Export reliability score and build zones as MRC overlays for one map.

Writes under ``--out-dir``:

- ``{label}_reliability.mrc`` — reliability score on the reference grid
- ``{label}_build_zones.mrc`` — omit / caution / build zone labels

Example::

    halfmap-qc reliability \\
      --reference deposited.map --half1 half1.map --half2 half2.map \\
      --features features.npz --contour 0.116 --out-dir reliability_out
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np

from cryoem_mrc.analysis import build_contour_mask
from cryoem_mrc.density_source import rho_normalized_for_reliability
from cryoem_mrc.io import load_mrc
from cryoem_mrc.map_grid import load_full_and_half_maps
from cryoem_mrc.mask_bbox import (
    bbox_from_mask,
    crop_array,
    embed_array,
    format_bbox_log,
    pad_voxels_for_filters,
)
from cryoem_mrc.pipeline import load_feature_maps
from cryoem_mrc.reliability import (
    attach_reliability_to_features,
    save_build_zone_mrc,
    save_reliability_mrc,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reference", required=True, type=Path, help="Reference map (grid for MRC exports)")
    p.add_argument("--half1", required=True, type=Path)
    p.add_argument("--half2", required=True, type=Path)
    p.add_argument("--features", required=True, type=Path, help="Feature .npz from halfmap-qc features")
    p.add_argument(
        "--contour",
        required=True,
        type=float,
        help="Density contour for the analysis mask (same value as the analyze step)",
    )
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument(
        "--label",
        type=str,
        default=None,
        help="Short label for output filenames (default: reference map stem)",
    )
    p.add_argument("--window", type=int, default=5)
    p.add_argument(
        "--no-crop-to-contour",
        action="store_true",
        help="Compute constraint V on the full grid (default: tight bbox around contour mask)",
    )
    p.add_argument(
        "--density-source",
        choices=("avg_half", "primary"),
        default="avg_half",
        help="ρ for constraint V: avg_half (default, matched to half-map CC) or primary (sensitivity)",
    )
    return p.parse_args(argv)


def _output_label(args: argparse.Namespace) -> str:
    if args.label and str(args.label).strip():
        return str(args.label).strip()
    return args.reference.stem


def _paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "reference": args.reference,
        "half1": args.half1,
        "half2": args.half2,
        "features": args.features,
    }


def _load_local_var(features_path: Path) -> np.ndarray:
    with np.load(features_path, allow_pickle=False) as d:
        return np.asarray(d["local_variance"], dtype=np.float32)


def _optional_density_normalized(features_path: Path) -> np.ndarray | None:
    feats = load_feature_maps(features_path)
    if "density_normalized" not in feats:
        return None
    return np.asarray(feats["density_normalized"], dtype=np.float32)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    paths = _paths(args)
    label = _output_label(args)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    log = "[halfmap_reliability]"
    for k, p in paths.items():
        if not p.exists():
            print(f"{log} ERROR: missing {k}: {p}", file=sys.stderr)
            return 2

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

    zones = feats["build_zone"]
    zone_counts = {int(z): int((zones[mask] == z).sum()) for z in (0, 1, 2)}

    rel_mrc = save_reliability_mrc(
        paths["reference"],
        feats["reliability_score"],
        out_dir / f"{label}_reliability.mrc",
    )
    zone_mrc = save_build_zone_mrc(
        paths["reference"], zones, out_dir / f"{label}_build_zones.mrc"
    )
    print(f"{log} wrote {rel_mrc}", flush=True)
    print(f"{log} wrote {zone_mrc}", flush=True)
    print(
        f"{log} zone voxels (omit/caution/build): "
        f"{zone_counts.get(0, 0):,} / {zone_counts.get(1, 0):,} / {zone_counts.get(2, 0):,}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
