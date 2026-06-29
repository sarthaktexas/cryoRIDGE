"""Tests for cohort manifest eligibility rules."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from cryoem_mrc.manifest_policy import (
    cohort_tag_for_manifest,
    row_ca_metrics_eligible,
    row_qscore_eligible,
    row_resmap_ca_headline_eligible,
    row_resmap_expected_failure,
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


class TestManifestPolicy(unittest.TestCase):
    def test_maps_only_by_source(self) -> None:
        row = _row(
            flexibility_source="windowed_halfmap_correlation_only",
            flexibility_path_or_pdb="",
        )
        self.assertTrue(row_uses_maps_only_metrics(row))
        self.assertFalse(row_qscore_eligible(row))
        self.assertFalse(row_ca_metrics_eligible(row))

    def test_npc_sta_excluded_from_qscore(self) -> None:
        row = _row(
            emdb_id="52153",
            flexibility_source="windowed_halfmap_correlation_only",
            flexibility_path_or_pdb="",
            model_metrics="maps_only",
            qscore_eligible="no",
        )
        self.assertFalse(row_qscore_eligible(row))

    def test_borderline_qscore(self) -> None:
        with TemporaryDirectory() as tmp:
            pdb = Path(tmp) / "9nhz.cif"
            pdb.write_text("data\n", encoding="utf-8")
            row = _row(
                emdb_id="50267",
                qscore_eligible="borderline",
                flexibility_path_or_pdb=str(pdb),
            )
            self.assertTrue(row_qscore_eligible(row, include_borderline=True))
            self.assertFalse(row_qscore_eligible(row, include_borderline=False))

    def test_resmap_expected_failure_excludes_headline(self) -> None:
        with TemporaryDirectory() as tmp:
            pdb = Path(tmp) / "9nhz.cif"
            pdb.write_text("data\n", encoding="utf-8")
            row = _row(
                emdb_id="29262",
                resmap_expected_failure="document",
                flexibility_path_or_pdb=str(pdb),
            )
            self.assertTrue(row_resmap_expected_failure(row))
            self.assertFalse(row_resmap_ca_headline_eligible(row))
            self.assertTrue(row_ca_metrics_eligible(row))

    def test_cohort_tag_for_manifest(self) -> None:
        self.assertEqual(cohort_tag_for_manifest(Path("cohort/manifest.csv")), "core")
        self.assertEqual(
            cohort_tag_for_manifest(Path("cohort/expansion_manifest.csv")),
            "expansion",
        )


if __name__ == "__main__":
    unittest.main()
