"""Model-placement audit: tercile zones vs absolute local CC."""

from __future__ import annotations

from cryoem_mrc.structure_validation import (
    ModelPlacementAuditStats,
    ResidueValidationRow,
    compute_model_placement_audit_stats,
)


def _row(
    *,
    zone: int,
    cc: float,
    reliability: float = 0.5,
    in_mask: bool = True,
) -> ResidueValidationRow:
    return ResidueValidationRow(
        chain="A",
        seq_num=1,
        seq_icode="",
        res_name="ALA",
        x=0.0,
        y=0.0,
        z=0.0,
        b_iso=50.0,
        reliability_score=reliability,
        reliability_H_repro=reliability,
        build_zone=zone,
        in_contour_mask=in_mask,
        windowed_halfmap_correlation=cc,
    )


def test_high_quality_map_low_omit_but_no_low_cc() -> None:
    """Apoferritin-like: many omit-tercile residues, none truly low CC."""
    rows = [_row(zone=0, cc=0.98) for _ in range(2)]
    rows += [_row(zone=1, cc=0.99) for _ in range(3)]
    rows += [_row(zone=2, cc=0.99) for _ in range(5)]
    stats = compute_model_placement_audit_stats(rows, emdb_id="11638", cc_threshold=0.5)
    assert stats.frac_in_omit_zone == 0.2
    assert stats.frac_cc_below_threshold == 0.0
    assert stats.median_local_cc > 0.97


def test_poor_map_fraction_below_cc_threshold() -> None:
    rows = [_row(zone=0, cc=0.1) for _ in range(4)]
    rows += [_row(zone=2, cc=0.8) for _ in range(6)]
    stats = compute_model_placement_audit_stats(rows, emdb_id="49450", cc_threshold=0.5)
    assert stats.frac_in_omit_zone == 0.4
    assert stats.frac_cc_below_threshold == 0.4


def test_out_of_mask_ignored() -> None:
    rows = [_row(zone=0, cc=0.1, in_mask=False)]
    stats = compute_model_placement_audit_stats(rows, emdb_id="test")
    assert stats.n_in_mask == 0
    assert stats.notes == "no in-mask residues"


def test_multi_cc_thresholds_and_coverage() -> None:
    rows = [_row(zone=0, cc=0.45, reliability=0.1) for _ in range(3)]
    rows += [_row(zone=2, cc=0.75, reliability=0.8) for _ in range(7)]
    rows += [_row(zone=2, cc=0.9, reliability=0.9, in_mask=False) for _ in range(2)]
    stats = compute_model_placement_audit_stats(rows, emdb_id="4941", cc_threshold=0.5)
    assert stats.frac_in_contour_mask == 10 / 12
    assert stats.frac_cc_below_0_50 == 0.3
    assert stats.frac_cc_below_0_60 == 0.3
    assert stats.frac_cc_below_0_70 == 0.3
    assert stats.frac_reliability_below_threshold == 0.3
    assert stats.spearman_reliability_vs_cc > 0.9


def test_decoupled_reliability_vs_cc() -> None:
    """ClpB WT-2A-like: high reliability rank can anti-track local CC."""
    rows = [_row(zone=0, cc=0.8, reliability=0.1) for _ in range(4)]
    rows += [_row(zone=2, cc=0.6, reliability=0.9) for _ in range(6)]
    stats = compute_model_placement_audit_stats(rows, emdb_id="4941")
    assert stats.spearman_reliability_vs_cc < 0
