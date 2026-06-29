"""Tests for Q-score cohort exclusion helpers."""

from __future__ import annotations

import unittest

from cryoem_mrc.qscore_cohort import (
    QSCORE_CORE_EXCLUDE,
    QSCORE_PANEL_EXCLUDE,
    filter_emdb_ids,
    qscore_exclude_ids,
)


class TestQscoreCohort(unittest.TestCase):
    def test_panel_exclude_is_subset_of_core(self) -> None:
        self.assertTrue(QSCORE_PANEL_EXCLUDE <= qscore_exclude_ids(core=True))

    def test_core_adds_groel_tight(self) -> None:
        self.assertIn("13308", QSCORE_CORE_EXCLUDE)
        self.assertNotIn("13308", QSCORE_PANEL_EXCLUDE)

    def test_filter_emdb_ids_core(self) -> None:
        ids = ["49450", "13308", "33736", "4940", "16119"]
        out = filter_emdb_ids(ids, core=True)
        self.assertEqual(out, ["49450", "16119"])


if __name__ == "__main__":
    unittest.main()
