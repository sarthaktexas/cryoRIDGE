"""Unit tests for leave-one-map-out incremental prediction (Design A)."""

from __future__ import annotations

import unittest

import numpy as np

from cryoem_mrc.incremental_prediction import (
    MapPredictionFrame,
    build_map_frame_from_metrics,
    normalize_metrics_columns,
    ols_r2,
    percentile_rank,
    run_lomo_incremental_prediction,
)


class TestPercentileRank(unittest.TestCase):
    def test_finite_values_map_to_unit_interval(self) -> None:
        x = np.array([10.0, 20.0, 30.0, np.nan])
        r = percentile_rank(x)
        self.assertTrue(np.allclose(r[:3], [1 / 3, 2 / 3, 1.0]))
        self.assertTrue(np.isnan(r[3]))


class TestMapFrame(unittest.TestCase):
    def test_normalize_legacy_cc_column(self) -> None:
        import pandas as pd

        df = pd.DataFrame({"local_cross_correlation": [0.5, 0.6]})
        out = normalize_metrics_columns(df)
        self.assertIn("windowed_halfmap_correlation", out.columns)

    def test_build_frame_requires_min_residues(self) -> None:
        import pandas as pd

        n = 20
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            {
                "in_contour_mask": np.ones(n, bool),
                "local_variance": rng.normal(size=n),
                "windowed_halfmap_correlation": rng.normal(size=n),
                "local_resolution": rng.uniform(2, 5, size=n),
                "v_metric": rng.normal(size=n),
                "q_score": rng.uniform(0, 1, size=n),
            }
        )
        self.assertIsNone(
            build_map_frame_from_metrics(df, emdb_id="1", target_col="q_score", min_residues=30)
        )


class TestLomoIncremental(unittest.TestCase):
    def _synthetic_frames(self, *, v_independent: bool) -> list[MapPredictionFrame]:
        rng = np.random.default_rng(1)
        frames: list[MapPredictionFrame] = []
        for map_idx in range(4):
            n = 80
            var = rng.normal(size=n)
            cc = rng.normal(size=n)
            loc = rng.uniform(0, 1, size=n)
            if v_independent:
                v = rng.normal(size=n)
                y = 0.4 * var + 0.3 * cc + 0.2 * loc + 0.5 * v + rng.normal(scale=0.05, size=n)
            else:
                v = 0.9 * var + rng.normal(scale=0.02, size=n)
                y = 0.5 * var + 0.3 * cc + 0.2 * loc + rng.normal(scale=0.05, size=n)
            baseline = np.column_stack(
                [percentile_rank(var), percentile_rank(cc), percentile_rank(loc)]
            )
            full = np.column_stack([baseline, percentile_rank(v)])
            frames.append(
                MapPredictionFrame(
                    emdb_id=str(map_idx),
                    X_baseline=baseline,
                    X_full=full,
                    y=percentile_rank(y),
                    n_residues=n,
                )
            )
        return frames

    def test_v_redundant_yields_near_zero_median_delta_r2(self) -> None:
        summary = run_lomo_incremental_prediction(
            self._synthetic_frames(v_independent=False), target="q_score"
        )
        self.assertLess(abs(summary.median_delta_r2), 0.05)

    def test_v_independent_can_improve_held_out_r2(self) -> None:
        summary = run_lomo_incremental_prediction(
            self._synthetic_frames(v_independent=True), target="q_score"
        )
        self.assertGreater(summary.median_delta_r2, 0.02)
        self.assertGreater(summary.n_positive_delta_r2, 0)


class TestOlsR2(unittest.TestCase):
    def test_perfect_prediction(self) -> None:
        y = np.array([1.0, 2.0, 3.0])
        self.assertAlmostEqual(ols_r2(y, y), 1.0)


if __name__ == "__main__":
    unittest.main()
