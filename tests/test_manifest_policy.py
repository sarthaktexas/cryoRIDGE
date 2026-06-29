"""Tests for cohort manifest eligibility rules."""

from cryoem_mrc.manifest_policy import (
    row_ca_metrics_eligible,
    row_qscore_eligible,
    row_uses_maps_only_metrics,
)


def _row(**kwargs: str) -> dict[str, str]:
    base = {
        "emdb_id": "49450",
        "flexibility_source": "b_factor",
        "flexibility_path_or_pdb": "pdb/9nhz.cif",
        "model_metrics": "pdb",
        "qscore_eligible": "yes",
    }
    base.update(kwargs)
    return base


def test_maps_only_by_source() -> None:
    row = _row(flexibility_source="windowed_halfmap_correlation_only", flexibility_path_or_pdb="")
    assert row_uses_maps_only_metrics(row)
    assert not row_qscore_eligible(row)
    assert not row_ca_metrics_eligible(row)


def test_npc_sta_excluded_from_qscore() -> None:
    row = _row(
        emdb_id="52153",
        flexibility_source="windowed_halfmap_correlation_only",
        flexibility_path_or_pdb="",
        model_metrics="maps_only",
        qscore_eligible="no",
    )
    assert not row_qscore_eligible(row)


def test_borderline_qscore() -> None:
    row = _row(emdb_id="50267", qscore_eligible="borderline")
    assert row_qscore_eligible(row, include_borderline=True)
    assert not row_qscore_eligible(row, include_borderline=False)


def test_resmap_expected_failure_excludes_headline() -> None:
    from cryoem_mrc.manifest_policy import row_resmap_ca_headline_eligible, row_resmap_expected_failure

    row = _row(emdb_id="29262", resmap_expected_failure="document")
    assert row_resmap_expected_failure(row)
    assert not row_resmap_ca_headline_eligible(row)
    assert row_ca_metrics_eligible(row)
