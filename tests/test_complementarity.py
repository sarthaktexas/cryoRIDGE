"""Tests for gradient-family + geometry complementarity helpers."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from thesis.complementarity import (
    FOCUS_PREDICTORS,
    LOMO_MODEL_FEATURES,
    masked_percentile_rank,
    spearman_vs_q,
)


class TestComplementarity(unittest.TestCase):
    def test_focus_predictors_match_lomo_union(self) -> None:
        union = set()
        for cols in LOMO_MODEL_FEATURES.values():
            union.update(c for c in cols if c not in ("local_variance", "local_resolution", "windowed_halfmap_correlation"))
        self.assertEqual(set(FOCUS_PREDICTORS), union)

    def test_spearman_vs_q(self) -> None:
        n = 50
        df = pd.DataFrame(
            {
                "in_contour_mask": np.ones(n, bool),
                "q_score": np.linspace(0, 1, n),
                "smoothness": np.linspace(0, 1, n),
                "neg_lam_min": np.linspace(0, -1, n),
                "T_vonweizsacker": np.linspace(0, 1, n),
                "V_curvature": np.linspace(0, 1, n),
            }
        )
        rhos = spearman_vs_q(df, FOCUS_PREDICTORS)
        self.assertAlmostEqual(rhos["smoothness"], 1.0, places=5)
        self.assertAlmostEqual(rhos["neg_lam_min"], -1.0, places=5)

    def test_masked_percentile_rank(self) -> None:
        x = np.array([1.0, 2.0, 3.0, np.nan])
        m = np.array([True, True, True, False])
        r = masked_percentile_rank(x, m)
        self.assertTrue(np.allclose(r[:3], [1 / 3, 2 / 3, 1.0]))
        self.assertTrue(np.isnan(r[3]))


if __name__ == "__main__":
    unittest.main()
