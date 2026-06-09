"""Placement supplement: zone medians and inversion detection."""

from __future__ import annotations

from cryoem_mrc.placement_supplement import in_mask_arrays, median_by_zone, spearman_pair
from cryoem_mrc.structure_validation import ResidueValidationRow


def _row(*, zone: int, cc: float, b: float, rel: float, in_mask: bool = True) -> ResidueValidationRow:
    return ResidueValidationRow(
        chain="A",
        seq_num=1,
        seq_icode="",
        res_name="ALA",
        x=0.0,
        y=0.0,
        z=0.0,
        b_iso=b,
        reliability_score=rel,
        reliability_H_repro=rel,
        build_zone=zone,
        in_contour_mask=in_mask,
        local_cross_correlation=cc,
    )


def test_wt2a_like_inversion() -> None:
    """Omit zone higher CC than build — WT-2A-like pattern."""
    rows = [_row(zone=0, cc=0.8, b=120.0, rel=0.1) for _ in range(4)]
    rows += [_row(zone=2, cc=0.6, b=60.0, rel=0.9) for _ in range(6)]
    cc, b, rel, zones = in_mask_arrays(rows)
    med_cc = median_by_zone(cc, zones)
    assert med_cc[0] > med_cc[2]
    assert spearman_pair(rel, cc) < 0
    assert spearman_pair(rel, b) < 0
