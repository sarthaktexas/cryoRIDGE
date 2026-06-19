"""Tests for placement utility Figma export."""

from __future__ import annotations

import json
import unittest

from cryoem_mrc.placement_figma_export import (
    DATA_MARKER_END,
    DATA_MARKER_START,
    FIGMA_JSON,
    _embed_data_in_ui,
    export_placement_figma_data,
)
from cryoem_mrc.placement_utility import aligned_rank_recovery_rho


class PlacementFigmaExportTest(unittest.TestCase):
    def test_embed_data_placed_after_body(self) -> None:
        html = "<html><body>\n<h1>Test</h1>\n</body></html>"
        block = (
            f"{DATA_MARKER_START}\n"
            '<script type="application/json" id="placement-data">{"panels":{}}</script>\n'
            f"{DATA_MARKER_END}"
        )
        out = _embed_data_in_ui(html, block)
        body_idx = out.index("<body>")
        data_idx = out.index('id="placement-data"')
        self.assertLess(body_idx, data_idx)

    def test_export_produces_panels(self) -> None:
        try:
            export = export_placement_figma_data()
        except FileNotFoundError:
            self.skipTest("placement utility CSVs not available")

        payload = export.to_dict()
        h2h = payload["panels"]["head_to_head"]
        rr = payload["panels"]["rank_recovery"]
        self.assertEqual(h2h["kind"], "head_to_head_triple")
        self.assertEqual(len(h2h["predictors"]), 5)
        self.assertEqual(len(h2h["panels"]), 3)
        self.assertIn("frac_low_q_flagged", h2h["predictors"][0])
        self.assertEqual(rr["kind"], "bar")
        self.assertEqual(len(rr["bars"]), 5)
        self.assertIsNotNone(rr["bars"][0]["median_rho"])
        loc_bar = next(b for b in rr["bars"] if b["key"] == "spearman_q_vs_locres")
        self.assertGreater(loc_bar["median_rho"], 0.0)
        roc = payload["panels"].get("low_q_roc", {})
        self.assertEqual(roc.get("kind"), "roc")

    def test_aligned_locres_flips_sign(self) -> None:
        self.assertAlmostEqual(
            aligned_rank_recovery_rho(-0.127, "spearman_q_vs_locres"),
            0.127,
        )

    def test_figma_json_exists_when_export_ran(self) -> None:
        if not FIGMA_JSON.is_file():
            self.skipTest("placement_data.json not generated yet")
        data = json.loads(FIGMA_JSON.read_text())
        self.assertIn("panels", data)
        self.assertIn("head_to_head", data["panels"])


if __name__ == "__main__":
    unittest.main()
