"""Tests for cross-metric loading and correlations."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

import pandas as pd
from scipy import stats

from cryoem_mrc.halfmap_metrics import WINDOWED_HALFMAP_CORRELATION_KEY
from thesis.metric_comparison import METRIC_COLUMNS, compute_cross_metric_correlations, load_all_metrics
from cryoem_mrc.repo_paths import COHORT_MANIFEST


class TestMetricComparison(unittest.TestCase):
    def test_load_all_metrics_local_resolution_nan_without_blocres(self) -> None:
        try:
            df = load_all_metrics("49450", manifest=COHORT_MANIFEST)
        except FileNotFoundError:
            self.skipTest("EMD-49450 pipeline outputs not local")
        self.assertIn("local_resolution", df.columns)
        self.assertIn("v_metric", df.columns)
        locres_path = Path("outputs/emd_49450/locres_blocres.mrc")
        if locres_path.is_file():
            self.assertTrue(df["local_resolution"].notna().any())
        else:
            self.assertTrue(df["local_resolution"].isna().all())

    def test_compute_cross_metric_correlations_shape(self) -> None:
        try:
            df = load_all_metrics("49450", manifest=COHORT_MANIFEST)
        except FileNotFoundError:
            self.skipTest("EMD-49450 pipeline outputs not local")
        corr = compute_cross_metric_correlations(df)
        self.assertEqual(corr.shape[0], corr.shape[1])
        self.assertEqual(len(corr.index), len(METRIC_COLUMNS))
        self.assertIn("v_metric", corr.index)
        self.assertNotIn("reliability_score", corr.index)
        self.assertNotIn("reliability_H_repro", corr.index)
        self.assertIn("local_resolution", corr.columns)
        v_loc = corr.loc["v_metric", "local_resolution"]
        if df["local_resolution"].notna().sum() >= 30:
            self.assertTrue(np.isfinite(v_loc))

    def test_lh_pipeline_redundant_columns_excluded_from_cross_metric(self) -> None:
        rng = np.random.default_rng(0)
        h = rng.uniform(0, 10, size=200)
        order = np.argsort(h)
        ranks = np.empty_like(h)
        ranks[order] = np.arange(1, len(h) + 1, dtype=float)
        score = ranks / (len(h) + 1)
        df = pd.DataFrame(
            {
                "reliability_score": score,
                "reliability_H_repro": h,
                "v_metric": rng.normal(size=200),
                "b_factor": rng.normal(size=200),
                WINDOWED_HALFMAP_CORRELATION_KEY: rng.normal(size=200),
                "local_variance": rng.normal(size=200),
                "local_resolution": rng.normal(size=200),
                "in_contour_mask": True,
            }
        )
        rho, _ = stats.spearmanr(df["reliability_score"], df["reliability_H_repro"])
        self.assertAlmostEqual(float(rho), 1.0, places=10)
        corr = compute_cross_metric_correlations(df)
        self.assertIn("v_metric", corr.index)
        self.assertNotIn("reliability_score", corr.index)
        self.assertNotIn("reliability_H_repro", corr.index)


if __name__ == "__main__":
    unittest.main()
