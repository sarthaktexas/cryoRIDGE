"""Tests for two-half-map quick run."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from cryoem_mrc.analysis import build_contour_mask, suggest_contour
from cryoem_mrc.emringer_cohort import BUILDING_REGIME_MAX_RESOLUTION_A
from cryoem_mrc.halfmap_run import (
    HalfmapPairContext,
    load_halfmap_pair_context,
    run_halfmap_qc,
    summarize_halfmap_pair,
)
from cryoem_mrc.local_fsc import estimate_global_halfmap_fsc_resolution


class TestSuggestContour(unittest.TestCase):
    def test_suggest_contour_macromolecule_blob(self) -> None:
        d = np.zeros((40, 40, 40), dtype=np.float32)
        blob = np.s_[10:30, 10:30, 10:30]
        d[blob] = 1.0
        d[blob] += np.random.default_rng(0).normal(0, 0.02, (20, 20, 20)).astype(np.float32)
        contour = suggest_contour(d)
        frac = float(build_contour_mask(d, contour).mean())
        self.assertGreaterEqual(frac, 0.002)
        self.assertLessEqual(frac, 0.40)


class TestRunHalfmapQc(unittest.TestCase):
    def _mock_bundle(self, vol: np.ndarray):
        half = type("Half", (), {"data": vol, "voxel_size_zyx": (1.0, 1.0, 1.0)})()
        return type(
            "Bundle",
            (),
            {"half1": half, "half2": half, "reports": {"half2": type("R", (), {"ok": True})()}},
        )()

    def test_run_halfmap_qc_orchestrates_pipeline(self) -> None:
        with (
            patch("cryoem_mrc.halfmap_run.load_full_and_half_maps") as load_bundle,
            patch("cryoem_mrc.halfmap_run.save_volume_like_reference") as save_avg,
            patch("cryoem_mrc.__main__.main", return_value=0) as features,
            patch("cryoem_mrc.halfmap_run.reliability_main", return_value=0) as reliability,
        ):
            vol = np.ones((4, 4, 4), dtype=np.float32)
            load_bundle.return_value = self._mock_bundle(vol)

            half1 = Path("/tmp/h1.map")
            half2 = Path("/tmp/h2.map")
            out = Path("/tmp/out")

            result = run_halfmap_qc(half1, half2, out_dir=out)

            save_avg.assert_called_once()
            features.assert_called_once()
            reliability.assert_called_once()
            self.assertEqual(result["reliability_mrc"], (out / "h1_reliability.mrc").resolve())
            self.assertEqual(result["build_zones_mrc"], (out / "h1_build_zones.mrc").resolve())

    def test_run_halfmap_qc_accepts_explicit_contour(self) -> None:
        with (
            patch("cryoem_mrc.halfmap_run.save_volume_like_reference"),
            patch("cryoem_mrc.__main__.main", return_value=0) as features,
            patch("cryoem_mrc.halfmap_run.reliability_main", return_value=0),
        ):
            vol = np.ones((4, 4, 4), dtype=np.float32)
            half1 = Path("/tmp/h1.map")
            half2 = Path("/tmp/h2.map")
            ctx = HalfmapPairContext(
                half1=half1,
                half2=half2,
                bundle=self._mock_bundle(vol),
                avg=vol,
            )
            result = run_halfmap_qc(half1, half2, out_dir=Path("/tmp/out"), contour=0.5, context=ctx)
            self.assertEqual(result["contour"], 0.5)
            feature_argv = features.call_args[0][0]
            self.assertIn("0.5", feature_argv)


class TestSummarizeHalfmapPair(unittest.TestCase):
    def test_flags_coarse_resolution_outside_building_regime(self) -> None:
        vol = np.zeros((32, 32, 32), dtype=np.float32)
        vol[8:24, 8:24, 8:24] = 1.0
        rng = np.random.default_rng(0)
        h1 = vol + rng.normal(0, 0.3, vol.shape).astype(np.float32)
        h2 = vol + rng.normal(0, 0.3, vol.shape).astype(np.float32)
        half = type("Half", (), {"data": h1, "voxel_size_zyx": (2.0, 2.0, 2.0)})()
        half2 = type("Half", (), {"data": h2, "voxel_size_zyx": (2.0, 2.0, 2.0)})()
        bundle = type("Bundle", (), {"half1": half, "half2": half2, "reports": {}})()
        avg = (0.5 * (h1 + h2)).astype(np.float32)
        ctx = HalfmapPairContext(
            half1=Path("/tmp/h1.map"),
            half2=Path("/tmp/h2.map"),
            bundle=bundle,
            avg=avg,
        )
        with patch(
            "cryoem_mrc.halfmap_run.estimate_global_halfmap_fsc_resolution",
            return_value=BUILDING_REGIME_MAX_RESOLUTION_A + 1.0,
        ):
            summary = summarize_halfmap_pair(ctx)
        self.assertFalse(summary.in_building_regime)
        self.assertGreater(summary.resolution_a, BUILDING_REGIME_MAX_RESOLUTION_A)


class TestGlobalHalfmapFsc(unittest.TestCase):
    def test_identical_halves_yield_finite_resolution(self) -> None:
        vol = np.random.default_rng(1).normal(size=(16, 16, 16)).astype(np.float32)
        res = estimate_global_halfmap_fsc_resolution(vol, vol, voxel_size_a=1.0)
        self.assertTrue(np.isfinite(res))
        self.assertGreater(res, 0.0)


class TestLoadHalfmapPairContext(unittest.TestCase):
    def test_load_delegates_to_map_loader(self) -> None:
        with patch("cryoem_mrc.halfmap_run.load_full_and_half_maps") as load_bundle:
            vol = np.ones((4, 4, 4), dtype=np.float32)
            half = type("Half", (), {"data": vol, "voxel_size_zyx": (1.0, 1.0, 1.0)})()
            load_bundle.return_value = type(
                "Bundle",
                (),
                {"half1": half, "half2": half, "reports": {}},
            )()
            ctx = load_halfmap_pair_context(Path("/tmp/h1.map"), Path("/tmp/h2.map"))
            self.assertEqual(ctx.avg.shape, (4, 4, 4))
            load_bundle.assert_called_once()


if __name__ == "__main__":
    unittest.main()
