"""Unit tests for placement utility analyses."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from thesis.placement_utility import (
    balanced_accuracy,
    compute_calibration_bins,
    compute_low_q_enrichment_row,
    compute_misranking_row,
    compute_rank_recovery_row,
    rank_auc,
    summarize_q_roc_per_map,
    _predictor_flags,
    _predictor_scores,
)


def _synthetic_df(*, n: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rel = rng.uniform(0, 1, size=n)
    q = 0.3 + 0.5 * rel + rng.normal(0, 0.1, size=n)
    cc = 0.2 + 0.6 * rel + rng.normal(0, 0.08, size=n)
    loc = rng.uniform(2.5, 5.0, size=n)
    var = rng.uniform(0.5, 5.0, size=n)
    zone = np.digitize(rel, [1 / 3, 2 / 3]) - 1
    return pd.DataFrame(
        {
            "reliability_score": rel,
            "q_score": q,
            "windowed_halfmap_correlation": cc,
            "local_resolution": loc,
            "local_variance": var,
            "build_zone": zone,
            "in_contour_mask": True,
        }
    )


class TestPlacementUtilityMetrics(unittest.TestCase):
    def test_rank_auc_perfect_separation(self) -> None:
        y = np.array([1, 1, 0, 0])
        scores = np.array([1.0, 0.9, 0.1, 0.0])
        self.assertAlmostEqual(rank_auc(y, scores), 1.0)

    def test_balanced_accuracy(self) -> None:
        y = np.array([1, 1, 0, 0])
        pred = np.array([1, 0, 0, 0])
        self.assertAlmostEqual(balanced_accuracy(y, pred), 0.75)


class TestLowQEnrichment(unittest.TestCase):
    def test_enrichment_tracks_correlated_q(self) -> None:
        df = _synthetic_df()
        row = compute_low_q_enrichment_row(df, emdb_id="test", q_threshold=0.5)
        assert row is not None
        self.assertGreater(row.frac_low_q_in_omit_zone, row.omit_zone_baseline)
        self.assertGreater(row.frac_low_q_reliability_below, 0.2)

    def test_predictor_flags_shapes(self) -> None:
        df = _synthetic_df(n=50)
        flags = _predictor_flags(df)
        self.assertEqual(len(flags["omit_zone"]), 50)


class TestMisranking(unittest.TestCase):
    def test_misranking_row_finite(self) -> None:
        df = _synthetic_df()
        row = compute_misranking_row(df, emdb_id="test")
        assert row is not None
        self.assertTrue(np.isfinite(row.frac_omit_zone_low_q_tercile))


class TestLocresDirection(unittest.TestCase):
    def test_locres_flag_agrees_with_negative_q_correlation(self) -> None:
        """Higher BlocRes Å ⇒ lower Q; median flag should still classify better than chance."""
        rng = np.random.default_rng(7)
        n = 200
        loc = rng.uniform(3.0, 5.5, size=n)
        rel = rng.uniform(0.2, 0.9, size=n)
        q = 0.55 - 0.15 * (loc - 3.0) + 0.05 * rel + rng.normal(0, 0.04, size=n)
        cc = 0.3 + 0.4 * rel + rng.normal(0, 0.05, size=n)
        var = rng.uniform(0.5, 3.0, size=n)
        zone = np.digitize(rel, [1 / 3, 2 / 3]) - 1
        df = pd.DataFrame(
            {
                "reliability_score": rel,
                "q_score": q,
                "windowed_halfmap_correlation": cc,
                "local_resolution": loc,
                "local_variance": var,
                "v_metric": rng.uniform(0.1, 0.9, size=n),
                "build_zone": zone,
                "in_contour_mask": True,
            }
        )
        rr = compute_rank_recovery_row(df, emdb_id="test")
        assert rr is not None
        self.assertLess(rr.spearman_q_vs_locres, 0.0)

        low = q < 0.45
        flags = _predictor_flags(df)["locres_worse_than_median"]
        scores = _predictor_scores(df)["locres_worse_than_median"]
        self.assertGreater(balanced_accuracy(low, flags), 0.55)
        self.assertGreater(rank_auc(low, scores), 0.55)


class TestCalibration(unittest.TestCase):
    def test_calibration_bins_monotone_trend(self) -> None:
        df = _synthetic_df()
        bins = compute_calibration_bins([df], n_bins=5)
        self.assertGreater(len(bins), 2)
        means = [b.mean_q for b in bins]
        self.assertGreater(means[-1], means[0])


class TestResmapRoc(unittest.TestCase):
    def test_resmap_risk_score_tracks_low_q(self) -> None:
        rng = np.random.default_rng(3)
        n = 80
        resmap = rng.uniform(2.0, 6.0, size=n)
        q = 0.8 - 0.12 * resmap + rng.normal(0, 0.05, size=n)
        df = _synthetic_df(n=n, seed=1)
        df["q_score"] = q
        df["local_resolution_resmap"] = resmap
        rows = summarize_q_roc_per_map([("test", df)], q_threshold=0.5)
        resmap_row = next(
            r for r in rows if r["predictor"] == "resmap_locres_worse_than_median"
        )
        self.assertGreater(float(resmap_row["auc"]), 0.75)

    def test_v_matches_reliability_auc(self) -> None:
        rng = np.random.default_rng(5)
        n = 80
        v = rng.uniform(0.0, 10.0, size=n)
        q = 0.2 + 0.06 * v + rng.normal(0, 0.05, size=n)
        df = _synthetic_df(n=n, seed=1)
        df["q_score"] = q
        df["v_metric"] = v
        df["reliability_score"] = (v - v.min()) / (v.max() - v.min())
        rows = summarize_q_roc_per_map(
            [("test", df)],
            predictors=("smoothness", "reliability_below_0_33"),
        )
        v_auc = float(next(r for r in rows if r["predictor"] == "smoothness")["auc"])
        rel_auc = float(
            next(r for r in rows if r["predictor"] == "reliability_below_0_33")["auc"]
        )
        self.assertAlmostEqual(v_auc, rel_auc, places=5)

    def test_resmap_tracks_low_emringer(self) -> None:
        rng = np.random.default_rng(4)
        n = 80
        resmap = rng.uniform(2.0, 6.0, size=n)
        em = 0.08 - 0.01 * resmap + rng.normal(0, 0.005, size=n)
        df = _synthetic_df(n=n, seed=1)
        df["emringer_score"] = em
        df["local_resolution_resmap"] = resmap
        rows = summarize_q_roc_per_map(
            [("test", df)], ground_truth="emringer_low"
        )
        resmap_row = next(
            r for r in rows if r["predictor"] == "resmap_locres_worse_than_median"
        )
        self.assertGreater(float(resmap_row["auc"]), 0.75)


if __name__ == "__main__":
    unittest.main()
