"""Unit tests for the ChimeraX reliability bundle compute module."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

_COMPUTE_PATH = (
    Path(__file__).resolve().parents[1] / "chimerax_reliability" / "src" / "compute.py"
)
_spec = importlib.util.spec_from_file_location("map_reliability_compute", _COMPUTE_PATH)
assert _spec and _spec.loader
_compute = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _compute
_spec.loader.exec_module(_compute)

compute_reliability = _compute.compute_reliability
classify_build_zones = _compute.classify_build_zones
percentile_rank_in_mask = _compute.percentile_rank_in_mask
BUILD_ZONE_PALETTE = _compute.BUILD_ZONE_PALETTE
BUILD_ZONE_RGBA = _compute.BUILD_ZONE_RGBA
RELIABILITY_PALETTE = _compute.RELIABILITY_PALETTE


class TestChimeraXReliabilityCompute(unittest.TestCase):
    def test_percentile_rank_in_mask_range(self) -> None:
        vol = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
        mask = np.ones_like(vol, dtype=bool)
        out = percentile_rank_in_mask(vol, mask)
        self.assertGreater(out[mask].min(), 0)
        self.assertLessEqual(out[mask].max(), 1.0)

    def test_build_zones_three_labels(self) -> None:
        score = np.linspace(0, 1, 30, dtype=np.float32).reshape(2, 3, 5)
        mask = np.ones_like(score, dtype=bool)
        zones = classify_build_zones(score, mask)
        self.assertEqual(set(np.unique(zones[mask])), {0, 1, 2})

    def test_compute_reliability_shapes_and_zones(self) -> None:
        rng = np.random.default_rng(0)
        ref = rng.normal(size=(8, 10, 12)).astype(np.float32)
        half1 = ref + rng.normal(scale=0.1, size=ref.shape).astype(np.float32)
        half2 = ref + rng.normal(scale=0.1, size=ref.shape).astype(np.float32)
        result = compute_reliability(ref, half1, half2, contour=float(ref.mean()))
        self.assertEqual(result.reliability_score.shape, ref.shape)
        self.assertEqual(result.build_zone.shape, ref.shape)
        self.assertGreater(result.mask_voxels, 0)
        self.assertEqual(sum(result.zone_counts.values()), result.mask_voxels)

    def test_build_zone_palette_has_three_colors(self) -> None:
        self.assertIn("red", BUILD_ZONE_PALETTE)
        self.assertIn("yellow", BUILD_ZONE_PALETTE)
        self.assertIn("green", BUILD_ZONE_PALETTE)
        self.assertEqual(set(BUILD_ZONE_RGBA), {0, 1, 2})

    def test_reliability_palette_white_blue_purple(self) -> None:
        self.assertIn("white", RELIABILITY_PALETTE)
        self.assertIn("blue", RELIABILITY_PALETTE)
        self.assertIn("purple", RELIABILITY_PALETTE)
        self.assertIn("0.5,blue", RELIABILITY_PALETTE)


if __name__ == "__main__":
    unittest.main()
