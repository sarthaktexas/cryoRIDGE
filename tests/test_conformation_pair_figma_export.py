"""Tests for conformation-pair panel B Figma export."""

from __future__ import annotations

import json
import unittest

from cryoem_mrc.conformation_pair_figma_export import (
    DATA_MARKER_END,
    DATA_MARKER_START,
    FIGMA_JSON,
    _embed_data_in_ui,
    export_conformation_pair_scatter_figma_data,
)


class ConformationPairFigmaExportTest(unittest.TestCase):
    def test_embed_data_placed_after_body(self) -> None:
        html = "<html><body>\n<h1>Test</h1>\n</body></html>"
        block = (
            f"{DATA_MARKER_START}\n"
            '<script type="application/json" id="conformation-pair-data">{"panel":{"points":[]}}</script>\n'
            f"{DATA_MARKER_END}"
        )
        out = _embed_data_in_ui(html, block)
        body_idx = out.index("<body>")
        data_idx = out.index('id="conformation-pair-data"')
        self.assertLess(body_idx, data_idx)

    def test_export_msba_when_data_available(self) -> None:
        try:
            export = export_conformation_pair_scatter_figma_data("41596", "41598")
        except FileNotFoundError:
            self.skipTest("MsbA residue_validation.csv not available")
        payload = export.to_dict()
        panel = payload["panel"]
        self.assertEqual(panel["kind"], "scatter")
        self.assertEqual(panel["letter"], "b")
        self.assertGreater(len(panel["points"]), 0)
        self.assertEqual(panel["points"][0]["x"], panel["points"][0]["x"])  # finite
        self.assertIn("color", panel["points"][0])
        self.assertGreaterEqual(len(panel["legend"]), 2)
        self.assertEqual(payload["emdb_a"], "41596")
        self.assertEqual(payload["emdb_b"], "41598")
        self.assertIsNotNone(payload["spearman_rho"])

    def test_figma_json_exists_when_export_ran(self) -> None:
        if not FIGMA_JSON.is_file():
            self.skipTest("conformation_pair_data.json not generated yet")
        data = json.loads(FIGMA_JSON.read_text())
        self.assertIn("panel", data)
        self.assertGreater(len(data["panel"]["points"]), 0)


if __name__ == "__main__":
    unittest.main()
