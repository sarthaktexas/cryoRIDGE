"""Tests for graphical-abstract cohort export."""

from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

from cryoem_mrc.graphical_abstract_export import (
    COHORT_DATA_MARKER_END,
    COHORT_DATA_MARKER_START,
    COHORT_JSON,
    _embed_cohort_data_in_ui,
    export_graphical_abstract_cohort_data,
    select_exemplar_map,
)


class GraphicalAbstractExportTest(unittest.TestCase):
    def test_embed_cohort_data_placed_after_body(self) -> None:
        html = "<html><body>\n<h1>Test</h1>\n<script>load()</script>\n</body></html>"
        block = (
            f"{COHORT_DATA_MARKER_START}\n"
            '<script type="application/json" id="cohort-data">{"n_maps":1}</script>\n'
            f"{COHORT_DATA_MARKER_END}"
        )
        out = _embed_cohort_data_in_ui(html, block)
        body_idx = out.index("<body>")
        data_idx = out.index('id="cohort-data"')
        script_idx = out.index("load()")
        self.assertLess(body_idx, data_idx)
        self.assertLess(data_idx, script_idx)

    def test_select_exemplar_prefers_high_rho_in_atomic_regime(self) -> None:
        try:
            pick = select_exemplar_map(min_in_mask=500)
        except FileNotFoundError:
            self.skipTest("placement_rank_recovery.csv not available")
        self.assertGreaterEqual(float(pick["global_resolution_a"]), 2.5)
        self.assertLess(float(pick["global_resolution_a"]), 4.0)
        self.assertGreaterEqual(float(pick["spearman_q_vs_reliability"]), 0.6)

    def test_export_exemplar_produces_scope_metadata(self) -> None:
        try:
            export = export_graphical_abstract_cohort_data(max_total=200, seed=1)
        except FileNotFoundError:
            self.skipTest("cohort Q-score + metrics outputs not available")

        self.assertEqual(export.n_maps, 1)
        self.assertGreater(export.n_residues_pooled, 500)
        self.assertGreater(export.spearman_q_vs_reliability, 0.6)
        self.assertTrue(len(export.locres_q_points) > 0)
        self.assertTrue(len(export.reliability_q_points) > 0)
        self.assertGreaterEqual(len(export.calibration_bins), 5)

        payload = export.to_dict()
        self.assertIn("scope", payload)
        self.assertEqual(payload["scope"]["emdb_id"], export.emdb_id)
        self.assertIn("spearman_q_vs_v", payload["stats"])

    def test_cohort_json_exists_when_export_ran(self) -> None:
        if not COHORT_JSON.is_file():
            self.skipTest("cohort_data.json not generated yet")
        data = json.loads(COHORT_JSON.read_text())
        self.assertIn("scope", data)
        self.assertIn("panels", data)


if __name__ == "__main__":
    unittest.main()
