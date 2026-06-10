"""Unit tests for ChimeraX script generation helpers."""

from __future__ import annotations

import unittest
from pathlib import Path

from cryoem_mrc.chimerax_figures import (
    ProteinFigureBundle,
    STATISTIC_SPECS,
    chimerax_domain_select_expr,
    write_domain_surface_cxc,
    write_statistic_surface_cxc,
)
from cryoem_mrc.conformation_pair import get_domain_regions_for_emdb


class ChimeraXFiguresTests(unittest.TestCase):
    def test_mgta_domain_select_uses_auth_seq_range(self) -> None:
        regions = get_domain_regions_for_emdb("49450")
        self.assertTrue(regions)
        had = next(r for r in regions if r.name == "HAD")
        expr = chimerax_domain_select_expr(had, Path("pdb/9nhz.cif"))
        self.assertEqual(expr, ":396-676")

    def test_write_statistic_surface_cxc_contains_color_sample(self) -> None:
        bundle = ProteinFigureBundle(
            emdb_id="49450",
            display_name="MgtA",
            reference_mrc=Path("data/emd_49450-mgtA_e2p+e1/emd_49450.map"),
            structure_path=Path("pdb/9nhz.cif"),
            contour=0.116,
        )
        spec = STATISTIC_SPECS["local_resolution"]
        script = Path("outputs/emd_49450/chimerax_figures/scripts/test_locres.cxc")
        png = Path("outputs/emd_49450/chimerax_figures/renders/test_locres.png")
        write_statistic_surface_cxc(
            bundle,
            statistic=spec,
            statistic_mrc=Path("outputs/emd_49450/locres_blocres.mrc"),
            out_png=png,
            out_script=script,
            vmin=2.5,
            vmax=5.0,
        )
        text = script.read_text()
        self.assertIn("color sample #1 map #2", text)
        self.assertIn("palette buylrd", text)
        self.assertIn("level 0.116", text)

    def test_write_domain_surface_cxc_colors_each_region(self) -> None:
        regions = get_domain_regions_for_emdb("49450")
        bundle = ProteinFigureBundle(
            emdb_id="49450",
            display_name="MgtA",
            reference_mrc=Path("data/emd_49450-mgtA_e2p+e1/emd_49450.map"),
            structure_path=Path("pdb/9nhz.cif"),
            contour=0.116,
            domain_regions=regions,
        )
        script = Path("outputs/emd_49450/chimerax_figures/scripts/test_domain.cxc")
        png = Path("outputs/emd_49450/chimerax_figures/renders/test_domain.png")
        write_domain_surface_cxc(bundle, out_png=png, out_script=script)
        text = script.read_text()
        self.assertIn("cartoon #2", text)
        self.assertGreaterEqual(text.count("color "), len(regions))


if __name__ == "__main__":
    unittest.main()
