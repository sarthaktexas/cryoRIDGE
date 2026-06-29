"""Tests for two-half-map quick run."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from cryoem_mrc.analysis import build_contour_mask, suggest_contour
from cryoem_mrc.halfmap_run import run_halfmap_qc


class TestSuggestContour(unittest.TestCase):
    def test_suggest_contour_macromolecule_blob(self) -> None:
        d = np.zeros((40, 40, 40), dtype=np.float32)
        d[10:30, 10:30, 10:30] = 1.0
        d += np.random.default_rng(0).normal(0, 0.02, d.shape).astype(np.float32)
        contour = suggest_contour(d)
        frac = float(build_contour_mask(d, contour).mean())
        self.assertGreaterEqual(frac, 0.002)
        self.assertLessEqual(frac, 0.40)


class TestRunHalfmapQc(unittest.TestCase):
    def test_run_halfmap_qc_orchestrates_pipeline(self) -> None:
        with (
            patch("cryoem_mrc.halfmap_run.load_full_and_half_maps") as load_bundle,
            patch("cryoem_mrc.halfmap_run.save_volume_like_reference") as save_avg,
            patch("cryoem_mrc.__main__.main", return_value=0) as features,
            patch("cryoem_mrc.halfmap_run.reliability_main", return_value=0) as reliability,
        ):
            vol = np.ones((4, 4, 4), dtype=np.float32)
            half = type("Half", (), {"data": vol})()
            load_bundle.return_value = type(
                "Bundle",
                (),
                {"half1": half, "half2": half, "reports": {"half2": type("R", (), {"ok": True})()}},
            )()

            half1 = Path("/tmp/h1.map")
            half2 = Path("/tmp/h2.map")
            out = Path("/tmp/out")

            result = run_halfmap_qc(half1, half2, out_dir=out)

            save_avg.assert_called_once()
            features.assert_called_once()
            reliability.assert_called_once()
            self.assertEqual(result["reliability_mrc"], out / "h1_reliability.mrc")
            self.assertEqual(result["build_zones_mrc"], out / "h1_build_zones.mrc")


if __name__ == "__main__":
    unittest.main()
