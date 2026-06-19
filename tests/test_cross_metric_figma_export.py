"""Tests for cohort cross-metric Figma export."""

from __future__ import annotations

import json
import unittest

from cryoem_mrc.cross_metric_figma_export import (
    DATA_MARKER_END,
    DATA_MARKER_START,
    FIGMA_JSON,
    _embed_data_in_ui,
    export_cross_metric_figma_data,
)


class CrossMetricFigmaExportTest(unittest.TestCase):
    def test_embed_data_placed_after_body(self) -> None:
        html = "<html><body>\n<h1>Test</h1>\n</body></html>"
        block = (
            f"{DATA_MARKER_START}\n"
            '<script type="application/json" id="cross-metric-data">{"panels":{}}</script>\n'
            f"{DATA_MARKER_END}"
        )
        out = _embed_data_in_ui(html, block)
        body_idx = out.index("<body>")
        data_idx = out.index('id="cross-metric-data"')
        self.assertLess(body_idx, data_idx)

    def test_export_produces_panels(self) -> None:
        try:
            export = export_cross_metric_figma_data()
        except FileNotFoundError:
            self.skipTest("cross_metric_correlations.csv not available for cohort")

        payload = export.to_dict()
        heat = payload["panels"]["median_heatmap"]
        locres = payload["panels"]["locres_pairs"]
        self.assertEqual(heat["kind"], "heatmap")
        self.assertEqual(len(heat["row_labels"]), len(heat["col_labels"]))
        self.assertGreater(len(heat["cells"]), 0)
        self.assertEqual(locres["kind"], "grouped_barh")
        self.assertGreater(len(locres["series"]), 0)
        self.assertGreater(len(locres["structures"]), 2)
        self.assertEqual(len(locres["series"][0]["values"]), len(locres["structures"]))

    def test_figma_json_exists_when_export_ran(self) -> None:
        if not FIGMA_JSON.is_file():
            self.skipTest("cross_metric_data.json not generated yet")
        data = json.loads(FIGMA_JSON.read_text())
        self.assertIn("panels", data)
        self.assertIn("median_heatmap", data["panels"])


if __name__ == "__main__":
    unittest.main()
