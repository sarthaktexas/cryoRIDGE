"""Tests for cohort Q vs V Figma export."""

from __future__ import annotations

import json
import unittest

from cryoem_mrc.qscore_figma_export import (
    DATA_MARKER_END,
    DATA_MARKER_START,
    FIGMA_JSON,
    _cohort_records,
    _embed_data_in_ui,
    export_q_vs_v_cohort_figma_data,
)


class QscoreFigmaExportTest(unittest.TestCase):
    def test_embed_data_placed_after_body(self) -> None:
        html = "<html><body>\n<h1>Test</h1>\n</body></html>"
        block = (
            f"{DATA_MARKER_START}\n"
            '<script type="application/json" id="q-vs-v-data">{"panels":{"a":{"bars":[]}}}</script>\n'
            f"{DATA_MARKER_END}"
        )
        out = _embed_data_in_ui(html, block)
        body_idx = out.index("<body>")
        data_idx = out.index('id="q-vs-v-data"')
        self.assertLess(body_idx, data_idx)

    def test_cohort_records_when_csv_available(self) -> None:
        try:
            recs = _cohort_records()
        except FileNotFoundError:
            self.skipTest("qscore_correlations.csv not available")
        self.assertGreater(len(recs), 0)
        self.assertLessEqual(recs[0]["rho"], recs[-1]["rho"])

    def test_export_produces_cohort_panels(self) -> None:
        try:
            export = export_q_vs_v_cohort_figma_data()
        except FileNotFoundError:
            self.skipTest("qscore_correlations.csv not available")

        payload = export.to_dict()
        panel_a = payload["panels"]["a"]
        panel_b = payload["panels"]["b"]
        self.assertEqual(panel_a["letter"], "a")
        self.assertEqual(panel_a["kind"], "barh")
        self.assertGreater(len(panel_a["bars"]), 0)
        self.assertIn("rho", panel_a["bars"][0])
        self.assertEqual(panel_b["kind"], "scatter")
        self.assertGreater(len(panel_b["points"]), 0)
        sweep = payload["panels"]["resolution_sweep"]
        self.assertEqual(sweep["kind"], "line")
        self.assertGreater(len(sweep["points"]), 0)
        self.assertAlmostEqual(sweep["bin_width_a"], 0.5)
        std = payload["panels"]["resolution_standard_bins"]
        self.assertEqual(std["kind"], "bar")
        self.assertGreater(len(std["bars"]), 0)
        cutoff = payload["panels"]["resolution_cutoff"]
        self.assertEqual(cutoff["letter"], "c")
        self.assertGreater(len(cutoff["series_le"]), 0)
        self.assertGreater(len(cutoff["series_gt"]), 0)

    def test_figma_json_exists_when_export_ran(self) -> None:
        if not FIGMA_JSON.is_file():
            self.skipTest("q_vs_v_data.json not generated yet")
        data = json.loads(FIGMA_JSON.read_text())
        self.assertIn("panels", data)
        self.assertIn("bars", data["panels"]["a"])


if __name__ == "__main__":
    unittest.main()
