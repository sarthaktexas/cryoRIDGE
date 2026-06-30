"""Tests for cryoem_mrc.qscore_validation."""

from __future__ import annotations

import numpy as np

from cryoem_mrc.qscore_validation import (
    QscoreResidueRow,
    compute_qscore_validation_stats,
)


def _rows(n: int = 50) -> list[QscoreResidueRow]:
    q = np.linspace(0.2, 0.9, n)
    v = q + np.random.default_rng(0).normal(0, 0.02, n)
    return [
        QscoreResidueRow(
            chain="A",
            seq_num=i + 1,
            seq_icode="",
            res_name="ALA",
            x=0.0,
            y=0.0,
            z=0.0,
            b_iso=50.0,
            q_score=float(q[i]),
            reliability_smoothness=float(v[i]),
            reliability_smoothness_rank=float(i + 1) / n,
            in_contour_mask=True,
        )
        for i in range(n)
    ]


def test_qscore_validation_stats_positive_correlation() -> None:
    stats = compute_qscore_validation_stats(_rows(), emdb_id="49450", pdb_id="9nhz")
    assert stats.spearman_q_vs_smoothness > 0.95


def test_qscore_validation_stats_too_few() -> None:
    rows = _rows(5)
    stats = compute_qscore_validation_stats(rows, emdb_id="49450", pdb_id="9nhz")
    assert np.isnan(stats.spearman_q_vs_smoothness)


def test_qscore_validation_nan_smoothness_excluded() -> None:
    rows = _rows(20)
    rows.append(
        QscoreResidueRow(
            chain="A",
            seq_num=99,
            seq_icode="",
            res_name="GLY",
            x=0.0,
            y=0.0,
            z=0.0,
            b_iso=50.0,
            q_score=0.5,
            reliability_smoothness=float("nan"),
            reliability_smoothness_rank=float("nan"),
            in_contour_mask=True,
        )
    )
    stats = compute_qscore_validation_stats(rows, emdb_id="49450", pdb_id="9nhz")
    assert stats.spearman_q_vs_smoothness > 0.95
