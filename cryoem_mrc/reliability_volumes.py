"""Load halfmap-qc MRC exports and recompute reliability volumes when needed."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .analysis import build_contour_mask
from .density_source import rho_normalized_for_reliability
from .io import load_mrc
from .map_grid import load_full_and_half_maps
from .pipeline import load_feature_maps
from .repo_paths import find_features_npz, resolve_halfmap_reliability_dir
from .reliability import attach_reliability_to_features


def reliability_mrc_paths(emdb_id: str, out_dir: Path | None = None) -> tuple[Path, Path]:
    rel_dir = out_dir or resolve_halfmap_reliability_dir(emdb_id)
    label = f"emd_{emdb_id}"
    return rel_dir / f"{label}_reliability.mrc", rel_dir / f"{label}_build_zones.mrc"


def load_reliability_mrc_pair(
    emdb_id: str,
    *,
    out_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load ``{label}_reliability.mrc`` and ``{label}_build_zones.mrc``."""
    rel_mrc, zone_mrc = reliability_mrc_paths(emdb_id, out_dir)
    if not rel_mrc.is_file() or not zone_mrc.is_file():
        raise FileNotFoundError(
            f"EMD-{emdb_id}: missing reliability exports ({rel_mrc.name}, {zone_mrc.name})"
        )
    score = np.asarray(load_mrc(rel_mrc, dtype=np.float32), dtype=np.float32)
    zones = np.rint(load_mrc(zone_mrc, dtype=np.float32)).astype(np.uint8)
    return score, zones


def recompute_reliability_volumes(
    emdb_id: str,
    *,
    reference_path: Path,
    half1_path: Path,
    half2_path: Path,
    contour: float,
    window: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Recompute ``reliability_score`` and ``reliability_smoothness`` on the reference grid."""
    if not half1_path.is_file() or not half2_path.is_file():
        raise FileNotFoundError(f"EMD-{emdb_id}: missing half-maps for reliability recompute")
    features_npz = find_features_npz(reference_path.parent, emdb_id, contour)
    if features_npz is None or not features_npz.is_file():
        raise FileNotFoundError(f"EMD-{emdb_id}: missing features .npz for reliability recompute")

    reference = load_mrc(reference_path, dtype=np.float32)
    mask = build_contour_mask(reference, contour)
    with np.load(features_npz, allow_pickle=False) as feat:
        local_var = np.asarray(feat["local_variance"], dtype=np.float32)
    feats = load_feature_maps(features_npz)
    rho_norm = feats.get("density_normalized")
    bundle = load_full_and_half_maps(
        reference_path,
        half1_path,
        half2_path,
        reference="full",
        dtype=np.float32,
        resample_if_needed=True,
    )
    rho = rho_normalized_for_reliability(
        source="avg_half",
        half1=bundle.half1.data,
        half2=bundle.half2.data,
        features_density_normalized=(
            np.asarray(rho_norm, dtype=np.float32) if rho_norm is not None else None
        ),
    )
    work: dict[str, np.ndarray] = {
        "density_normalized": rho,
        "local_variance": local_var,
    }
    attach_reliability_to_features(
        work,
        bundle.half1.data,
        bundle.half2.data,
        window=window,
        mask=mask,
    )
    score = np.asarray(work["reliability_score"], dtype=np.float32)
    smooth = np.asarray(work["reliability_smoothness"], dtype=np.float32)
    return score, smooth
