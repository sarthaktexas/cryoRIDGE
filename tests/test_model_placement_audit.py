"""Model-placement audit: tercile zones vs absolute local CC."""

from __future__ import annotations

from cryoem_mrc.structure_validation import (
    ModelPlacementAuditStats,
    ResidueValidationRow,
    compute_model_placement_audit_stats,
)


def _row(*, zone: int, cc: float, in_mask: bool = True) -> ResidueValidationRow:
    return ResidueValidationRow(
        chain="A",
        seq_num=1,
        seq_icode="",
        res_name="ALA",
        x=0.0,
        y=0.0,
        z=0.0,
        b_iso=50.0,
        reliability_score=0.5,
        reliability_H_repro=0.5,
        build_zone=zone,
        in_contour_mask=in_mask,
        local_cross_correlation=cc,
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
