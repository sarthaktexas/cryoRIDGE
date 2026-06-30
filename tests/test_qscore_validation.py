"""Unit tests for Q-score validation helpers (no qscore runtime required)."""

from __future__ import annotations

import unittest

import numpy as np

from cryoem_mrc.qscore_validation import (
    QscoreResidueRow,
    attach_production_v_metric,
    compute_qscore_validation_stats,
)


def _row(q: float, v: float, *, in_mask: bool = True, b: float = 50.0) -> QscoreResidueRow:
    return QscoreResidueRow(
        chain="A",
        seq_num=1,
        seq_icode="",
        res_name="ALA",
        x=0.0,
        y=0.0,
        z=0.0,
        b_iso=b,
        q_score=q,
        reliability_constraint_V=v,
        reliability_constraint_V_rank=v,
        in_contour_mask=in_mask,
    )


class TestQscoreValidationStats(unittest.TestCase):
    def test_positive_correlation(self):
        rows = [_row(float(i) / 20.0, float(i), b=float(i)) for i in range(20)]
        stats = compute_qscore_validation_stats(rows, emdb_id="49450", pdb_id="9nhz")
        self.assertEqual(stats.n_in_mask, 20)
        self.assertGreater(stats.spearman_q_vs_V, 0.95)

    def test_respects_mask(self):
        rows = [_row(0.5, 1.0, in_mask=False) for _ in range(20)]
        stats = compute_qscore_validation_stats(rows, emdb_id="49450", pdb_id="9nhz")
        self.assertTrue(np.isnan(stats.spearman_q_vs_V))

    def test_attach_production_v_metric(self):
        rows = [
            QscoreResidueRow(
                chain="A",
                seq_num=i,
                seq_icode="",
                res_name="ALA",
                x=0.0,
                y=0.0,
                z=0.0,
                b_iso=50.0,
                q_score=float(i) / 20.0,
                reliability_constraint_V=float("nan"),
                reliability_constraint_V_rank=float("nan"),
                in_contour_mask=True,
            )
            for i in range(1, 21)
        ]
        import cryoem_mrc.qscore_validation as qv

        original_loader = qv.load_production_v_metric_lookup
        qv.load_production_v_metric_lookup = lambda emdb_id: {
            ( "A", i): float(i) for i in range(1, 21)
        }
        try:
            merged = attach_production_v_metric(rows, "6287")
        finally:
            qv.load_production_v_metric_lookup = original_loader

        stats = compute_qscore_validation_stats(merged, emdb_id="6287", pdb_id="6bdf")
        self.assertGreater(stats.spearman_q_vs_V, 0.95)


if __name__ == "__main__":
    unittest.main()
