"""Compose thesis ChimeraX figures from existing MgtA renders when present."""

from __future__ import annotations

import unittest
from pathlib import Path

from cryoem_mrc.chimerax_figures import (
    MGTA_CONFORMATION_PAIR,
    chimerax_render_png,
    export_pipeline_panel_assets,
    write_map_shell_surface_cxc,
)
from cryoem_mrc.repo_paths import OUTPUTS_ROOT


class ChimeraXMgtaPairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        emd_a, _ = MGTA_CONFORMATION_PAIR
        cls.map_shell = chimerax_render_png(emd_a, "map_shell")
        cls.stat = chimerax_render_png(emd_a, "local_resolution")
        cls.has_renders = cls.map_shell.is_file() and cls.stat.is_file()

    def test_write_map_shell_script(self) -> None:
        from cryoem_mrc.chimerax_figures import resolve_protein_bundle

        bundle = resolve_protein_bundle(MGTA_CONFORMATION_PAIR[0])
        script = Path("/tmp/test_map_shell.cxc")
        write_map_shell_surface_cxc(
            bundle,
            out_png=Path("/tmp/test_map_shell.png"),
            out_script=script,
            width=320,
            height=320,
            step=4,
            supersample=1,
        )
        text = script.read_text()
        self.assertIn("color #bbbbbb #2", text)
        self.assertNotIn("select ", text)

    def test_export_pipeline_panels(self) -> None:
        if not self.has_renders:
            self.skipTest("EMD-49450 map_shell + statistic renders not local")
        out_root = OUTPUTS_ROOT / "chimerax_figures" / "mgta_pair" / "_test_panels"
        outputs = export_pipeline_panel_assets("49450", out_dir=out_root)
        self.assertTrue(outputs["map_shell_3d"].is_file())
        self.assertTrue(outputs["map_shell_slice"].is_file())
        self.assertTrue(outputs["manifest"].is_file())


if __name__ == "__main__":
    unittest.main()
