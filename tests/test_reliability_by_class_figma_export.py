"""Tests for cohort ρ(Q, V) by protein-class Figma export."""

from __future__ import annotations

import json
import unittest

from cryoem_mrc.reliability_by_class_figma_export import (
    DATA_MARKER_END,
    DATA_MARKER_START,
    FIGMA_JSON,
    Y_LABEL,
    _embed_data_in_ui,
    export_q_vs_v_by_class_figma_data,
)


class QVsVByClassFigmaExportTest(unittest.TestCase):
    def test_embed_data_placed_after_body(self) -> None:
        html = "<html><body>\n<h1>Test</h1>\n</body></html>"
        block = (
            f"{DATA_MARKER_START}\n"
            '<script type="application/json" id="q-vs-v-by-class-data">{"panel":{"groups":[]}}</script>\n'
            f"{DATA_MARKER_END}"
        )
        out = _embed_data_in_ui(html, block)
        body_idx = out.index("<body>")
        data_idx = out.index('id="q-vs-v-by-class-data"')
        self.assertLess(body_idx, data_idx)

    def test_export_produces_box_strip_panel(self) -> None:
        try:
            export = export_q_vs_v_by_class_figma_data()
        except FileNotFoundError:
            self.skipTest("qscore_correlations.csv not available")

        payload = export.to_dict()
        panel = payload["panel"]
        self.assertEqual(panel["kind"], "box_strip")
        self.assertEqual(panel["y_label"], Y_LABEL)
        self.assertGreater(len(panel["groups"]), 0)
        grp = panel["groups"][0]
        self.assertIn("box", grp)
        self.assertIn("median", grp["box"])
        self.assertGreater(len(grp["points"]), 0)
        self.assertIn("jitter", grp["points"][0])
        self.assertEqual(len(payload["y_lim"]), 2)

    def test_figma_json_exists_when_export_ran(self) -> None:
        if not FIGMA_JSON.is_file():
            self.skipTest("q_vs_v_by_class_data.json not generated yet")
        data = json.loads(FIGMA_JSON.read_text())
        self.assertIn("panel", data)
        self.assertIn("groups", data["panel"])


if __name__ == "__main__":
    unittest.main()
