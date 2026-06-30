"""Tests for builder-omission gap enumeration and ROC wiring."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from cryoem_mrc.builder_omission import (
    enumerate_sequence_gaps,
    interpolate_gap_residue,
    summarize_builder_omission_roc_per_map,
    tpr_at_fpr,
)
from cryoem_mrc.placement_utility import placement_roc_positive_mask, rank_auc
from cryoem_mrc.structure_validation import CaResidue


def _ca(chain: str, seq: int, x: float) -> CaResidue:
    return CaResidue(
        chain=chain,
        seq_num=seq,
        seq_icode="",
        res_name="ALA",
        x=x,
        y=0.0,
        z=0.0,
        b_iso=50.0,
    )


class TestSequenceGaps(unittest.TestCase):
    def test_enumerates_internal_gap(self) -> None:
        residues = [_ca("A", 10, 0.0), _ca("A", 11, 3.8), _ca("A", 15, 19.0)]
        gaps = enumerate_sequence_gaps(residues, max_gap_length=10)
        self.assertEqual(len(gaps), 1)
        self.assertEqual(gaps[0].missing_seq_nums, (12, 13, 14))

    def test_skips_large_gap(self) -> None:
        residues = [_ca("A", 1, 0.0), _ca("A", 30, 100.0)]
        gaps = enumerate_sequence_gaps(residues, max_gap_length=5)
        self.assertEqual(gaps, [])

    def test_interpolation_midpoint(self) -> None:
        left = _ca("A", 10, 0.0)
        right = _ca("A", 20, 10.0)
        from cryoem_mrc.builder_omission import SequenceGap

        gap = SequenceGap(chain="A", left=left, right=right, missing_seq_nums=(15,))
        mid = interpolate_gap_residue(gap, 15)
        self.assertAlmostEqual(mid.x, 5.0)


class TestBuilderOmissionRoc(unittest.TestCase):
    def _synthetic_frame(self, n_built: int = 60, n_omit: int = 40) -> pd.DataFrame:
        rng = np.random.default_rng(0)
        built_v = rng.uniform(0.0, 2.0, size=n_built)
        omit_v = rng.uniform(3.0, 6.0, size=n_omit)
        built_res = rng.uniform(2.0, 4.0, size=n_built)
        omit_res = rng.uniform(5.0, 8.0, size=n_omit)
        return pd.DataFrame(
            {
                "reliability_score": np.concatenate([1 - built_v / 6, 1 - omit_v / 6]),
                "v_metric": np.concatenate([built_v, omit_v]),
                "build_zone": 2,
                "windowed_halfmap_correlation": 0.7,
                "local_resolution": 3.5,
                "local_resolution_resmap": np.concatenate([built_res, omit_res]),
                "local_variance": 1.0,
                "builder_omission": [False] * n_built + [True] * n_omit,
                "in_contour_mask": True,
            }
        )

    def test_positive_mask(self) -> None:
        df = self._synthetic_frame()
        m, pos = placement_roc_positive_mask(df, ground_truth="builder_omission")
        self.assertEqual(int(m.sum()), len(df))
        self.assertEqual(int(pos.sum()), 40)

    def test_v_auc_beats_resmap(self) -> None:
        df = self._synthetic_frame()
        rows = summarize_builder_omission_roc_per_map([("test", df)])
        v_auc = float(next(r for r in rows if r["predictor"] == "constraint_v")["auc"])
        res_auc = float(
            next(r for r in rows if r["predictor"] == "resmap_locres_worse_than_median")["auc"]
        )
        self.assertGreaterEqual(v_auc, res_auc)
        self.assertGreater(v_auc, 0.85)

    def test_tpr_at_fpr_perfect(self) -> None:
        y = np.array([1, 1, 0, 0])
        scores = np.array([1.0, 0.9, 0.1, 0.0])
        self.assertAlmostEqual(tpr_at_fpr(y, scores, target_fpr=0.5), 1.0)
        self.assertAlmostEqual(rank_auc(y, scores), 1.0)


if __name__ == "__main__":
    unittest.main()
