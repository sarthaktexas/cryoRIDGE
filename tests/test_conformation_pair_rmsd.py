"""Conformation-pair Cα RMSD vs Δreliability helpers."""

from __future__ import annotations

import numpy as np

from cryoem_mrc.conformation_pair import (
    ConformationPairStats,
    compute_conformation_pair_stats,
    compute_per_residue_ca_rmsd,
)
from cryoem_mrc.structure_validation import ResidueValidationRow
from cryoem_mrc.thesis_figures import (
    _sorted_conformation_motion,
    compute_coupling_layout_scores,
    compute_diagonal_coupling_contrast_score,
    compute_domain_coupling_block_score,
    select_conformation_pair_figure_layout,
    DEFAULT_COUPLING_LAYOUT_THRESHOLD,
)


def _row(
    chain: str,
    seq: int,
    rel: float,
    *,
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
    in_mask: bool = True,
) -> ResidueValidationRow:
    return ResidueValidationRow(
        chain=chain,
        seq_num=seq,
        seq_icode="",
        res_name="ALA",
        x=x,
        y=y,
        z=z,
        b_iso=50.0,
        reliability_score=rel,
        reliability_H_repro=1.0 - rel,
        build_zone=2,
        in_contour_mask=in_mask,
    )


def test_per_residue_ca_rmsd_identical_models() -> None:
    pairs = [
        (_row("A", i, 0.1, x=1, y=float(i), z=2), _row("A", i, 0.2, x=1, y=float(i), z=2))
        for i in range(10)
    ]
    use, rmsd = compute_per_residue_ca_rmsd(pairs)
    assert len(use) == 10
    assert np.allclose(rmsd, 0.0, atol=1e-9)


def test_sorted_conformation_motion_returns_rmsd_and_drel() -> None:
    pairs = [
        (
            _row("A", i, float(i) * 0.01, x=0, y=float(i), z=0),
            _row("A", i, float(i) * 0.02, x=float(i) * 0.5, y=float(i), z=0),
        )
        for i in range(1, 21)
    ]
    packed = _sorted_conformation_motion(pairs)
    assert packed is not None
    _use, rmsd, drel, _chains = packed
    assert rmsd.shape == drel.shape
    assert np.all(drel > 0)
    assert np.all(rmsd > 0)


def test_conformation_pair_stats_rmsd_vs_reliability(monkeypatch) -> None:
    from cryoem_mrc import conformation_pair as cp

    pairs = [
        (_row("A", i, float(i) * 0.1), _row("A", i, float(i) * 0.2))
        for i in range(1, 21)
    ]
    use = pairs
    rmsd = np.arange(1, 21, dtype=np.float64)

    def _fake_rmsd(_pairs, *, in_mask_both=True):
        return use, rmsd

    monkeypatch.setattr(cp, "compute_per_residue_ca_rmsd", _fake_rmsd)
    stats = compute_conformation_pair_stats(pairs, emdb_a="1", emdb_b="2")
    assert isinstance(stats, ConformationPairStats)
    assert stats.n_matched_in_mask_both == 20
    assert stats.spearman_rmsd_vs_delta_reliability > 0.99


def test_diagonal_coupling_contrast_prefers_stronger_diagonal() -> None:
    n = 40
    corr = np.full((n, n), 0.1, dtype=np.float64)
    np.fill_diagonal(corr, 0.5)
    score = compute_diagonal_coupling_contrast_score(corr)
    assert score > 0.5


def test_domain_coupling_block_score_prefers_within_domain_blocks() -> None:
    corr = np.full((20, 20), 0.1, dtype=np.float64)
    corr[:10, :10] = 0.4
    corr[10:, 10:] = 0.35
    assignments = {"D1": list(range(10)), "D2": list(range(10, 20))}
    score = compute_domain_coupling_block_score(corr, assignments, ["D1", "D2"])
    assert score > 0.2


def test_coupling_layout_score_without_domains_uses_diagonal() -> None:
    corr = np.full((20, 20), 0.1, dtype=np.float64)
    np.fill_diagonal(corr, 0.5)
    scores = compute_coupling_layout_scores(corr)
    assert scores["diagonal_coupling_score"] > 0.5
    assert scores["coupling_layout_score"] == scores["diagonal_coupling_score"]
    assert not np.isfinite(scores["domain_coupling_score"])


def test_select_layout_uses_coupling_threshold() -> None:
    assert select_conformation_pair_figure_layout(0.05) == "domain"
    assert select_conformation_pair_figure_layout(0.12) == "block"
    assert select_conformation_pair_figure_layout(0.12, threshold=DEFAULT_COUPLING_LAYOUT_THRESHOLD) == "block"
