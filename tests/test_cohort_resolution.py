"""Tests for shared cohort resolution bins and figure labels."""

from __future__ import annotations

import unittest

from cryoem_mrc.cohort_labels import cohort_figure_label, short_display_name
from cryoem_mrc.cohort_resolution import (
    COHORT_RESOLUTION_BINS,
    cutoff_median_table,
    median_rho_by_resolution_bin,
    resolution_bin_label,
    summarize_resolution_bins,
    sweep_resolution_bins,
)


class TestResolutionBins(unittest.TestCase):
    def test_standard_bin_labels(self) -> None:
        self.assertEqual(resolution_bin_label(2.73), "2.5–4 Å")
        self.assertEqual(resolution_bin_label(1.22), "≤2.5 Å")
        self.assertEqual(resolution_bin_label(5.4), "4–6 Å")
        self.assertEqual(resolution_bin_label(7.0), ">6 Å")

    def test_median_by_bin(self) -> None:
        pairs = [(2.6, 0.8), (3.2, 0.6), (4.5, 0.1), (5.0, 0.0)]
        out = median_rho_by_resolution_bin(pairs, metric="q_vs_v")
        self.assertAlmostEqual(out["median_spearman_q_vs_v_2.5_4"], 0.7)
        self.assertAlmostEqual(out["median_spearman_q_vs_v_4_6"], 0.05)

    def test_cutoff_table(self) -> None:
        pairs = [(2.6, 0.8), (3.2, 0.6), (4.5, 0.1), (5.0, 0.0)]
        rows = cutoff_median_table(pairs, cutoffs=(4.0,))
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(float(rows[0]["median_rho_le_cutoff"]), 0.7)
        self.assertAlmostEqual(float(rows[0]["median_rho_gt_cutoff"]), 0.05)

    def test_sweep_and_summarize_nonempty(self) -> None:
        pairs = [(2.6, 0.8), (3.2, 0.6), (3.8, 0.4)]
        self.assertGreater(len(summarize_resolution_bins(pairs)), 0)
        self.assertGreater(len(sweep_resolution_bins(pairs)), 0)

    def test_bin_count(self) -> None:
        self.assertEqual(len(COHORT_RESOLUTION_BINS), 4)


class TestCohortLabels(unittest.TestCase):
    def test_short_display_name_strips_state(self) -> None:
        self.assertEqual(short_display_name("MgtA (E2P+E1)"), "MgtA")

    def test_cohort_figure_label_fallback(self) -> None:
        self.assertEqual(cohort_figure_label("99999"), "EMD-99999")


if __name__ == "__main__":
    unittest.main()
