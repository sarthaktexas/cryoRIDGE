"""Reliability / V export from deposited map only (no half-maps required)."""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np

from cryoem_mrc.analysis import build_contour_mask
from cryoem_mrc.density_source import rho_normalized_for_reliability
from cryoem_mrc.io import load_mrc, save_volume_like_reference
from cryoem_mrc.mask_bbox import (
    bbox_from_mask,
    crop_array,
    embed_array,
    format_bbox_log,
    pad_voxels_for_filters,
)
from cryoem_mrc.pipeline import load_feature_maps
from cryoem_mrc.reliability import attach_reliability_to_features, save_build_zone_mrc, save_reliability_mrc


def export_primary_reliability_mrcs(
    reference_path: Path,
    features_path: Path,
    *,
    contour: float,
    out_dir: Path,
    label: str,
    window: int = 5,
) -> tuple[Path, Path, Path]:
    """
    Write reliability score, build zones, and constraint-V (smoothness) on the deposited grid.

    Half-maps are not used; ρ is z-scored from the deposited map (or feature NPZ).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    reference = load_mrc(reference_path, dtype=np.float32)
    mask = build_contour_mask(reference, contour)
    local_var = np.asarray(
        np.load(features_path, allow_pickle=False)["local_variance"], dtype=np.float32
    )
    feats_maps = load_feature_maps(features_path)
    rho_norm = feats_maps.get("density_normalized")
    rho = rho_normalized_for_reliability(
        source="primary",
        half1=reference,
        half2=reference,
        features_density_normalized=(
            np.asarray(rho_norm, dtype=np.float32) if rho_norm is not None else None
        ),
        primary_volume=reference,
    )
    pad = pad_voxels_for_filters(window=window)
    bbox = bbox_from_mask(mask, pad=pad)
    print(
        f"[primary_reliability] contour crop: {format_bbox_log(bbox, reference.shape, pad=pad)}",
        flush=True,
    )
    work = {
        "density_normalized": crop_array(rho, bbox),
        "local_variance": crop_array(local_var, bbox),
    }
    ref_crop = crop_array(reference, bbox)
    attach_reliability_to_features(
        work,
        ref_crop,
        ref_crop,
        window=window,
        mask=crop_array(mask, bbox),
    )
    rel_keys = ("reliability_score", "reliability_smoothness", "build_zone")
    full_feats = {
        k: embed_array(reference.shape, bbox, work[k], dtype=work[k].dtype) for k in rel_keys
    }
    del work, ref_crop, rho, local_var
    gc.collect()

    rel_path = out_dir / f"{label}_reliability.mrc"
    zone_path = out_dir / f"{label}_build_zones.mrc"
    smooth_path = out_dir / f"{label}_reliability_smoothness.mrc"
    save_reliability_mrc(reference_path, full_feats["reliability_score"], rel_path)
    save_build_zone_mrc(reference_path, full_feats["build_zone"], zone_path)
    save_volume_like_reference(
        reference_path,
        np.asarray(full_feats["reliability_smoothness"], dtype=np.float32),
        smooth_path,
    )
    return rel_path, zone_path, smooth_path
