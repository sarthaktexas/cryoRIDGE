"""Unit tests for the ChimeraX reliability bundle compute module."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_COMPUTE_PATH = (
    Path(__file__).resolve().parents[1] / "chimerax_reliability" / "src" / "compute.py"
)
_spec = importlib.util.spec_from_file_location("map_reliability_compute", _COMPUTE_PATH)
assert _spec and _spec.loader
_compute = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_compute)

compute_reliability = _compute.compute_reliability
classify_build_zones = _compute.classify_build_zones
percentile_rank_in_mask = _compute.percentile_rank_in_mask
BUILD_ZONE_PALETTE = _compute.BUILD_ZONE_PALETTE
BUILD_ZONE_RGBA = _compute.BUILD_ZONE_RGBA
RELIABILITY_PALETTE = _compute.RELIABILITY_PALETTE


def test_percentile_rank_in_mask_range() -> None:
    vol = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
    mask = np.ones_like(vol, dtype=bool)
    out = percentile_rank_in_mask(vol, mask)
    assert out[mask].min() > 0
    assert out[mask].max() <= 1.0


def test_build_zones_three_labels() -> None:
    score = np.linspace(0, 1, 30, dtype=np.float32).reshape(2, 3, 5)
    mask = np.ones_like(score, dtype=bool)
    zones = classify_build_zones(score, mask)
    assert set(np.unique(zones[mask])) == {0, 1, 2}


def test_compute_reliability_shapes_and_zones() -> None:
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(8, 10, 12)).astype(np.float32)
    half1 = ref + rng.normal(scale=0.1, size=ref.shape).astype(np.float32)
    half2 = ref + rng.normal(scale=0.1, size=ref.shape).astype(np.float32)
    result = compute_reliability(ref, half1, half2, contour=float(ref.mean()))
    assert result.reliability_score.shape == ref.shape
    assert result.build_zone.shape == ref.shape
    assert result.mask_voxels > 0
    assert sum(result.zone_counts.values()) == result.mask_voxels


def test_build_zone_palette_has_three_colors() -> None:
    assert "red" in BUILD_ZONE_PALETTE
    assert "yellow" in BUILD_ZONE_PALETTE
    assert "green" in BUILD_ZONE_PALETTE
    assert set(BUILD_ZONE_RGBA) == {0, 1, 2}


def test_reliability_palette_white_blue_purple() -> None:
    assert "white" in RELIABILITY_PALETTE
    assert "blue" in RELIABILITY_PALETTE
    assert "purple" in RELIABILITY_PALETTE
    assert "0.5,blue" in RELIABILITY_PALETTE
