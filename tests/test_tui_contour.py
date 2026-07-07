"""Tests for interactive ChimeraX contour helpers."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from cryoem_mrc.tui import _mask_fraction, _prompt_chimerax_contour


class TestChimeraxContourHelpers(unittest.TestCase):
    def test_mask_fraction_on_blob(self) -> None:
        d = np.zeros((20, 20, 20), dtype=np.float32)
        d[5:15, 5:15, 5:15] = 1.0
        frac = _mask_fraction(d, 0.5)
        self.assertAlmostEqual(frac, 0.125, places=3)

    def test_prompt_chimerax_contour_accepts_level(self) -> None:
        density = np.zeros((10, 10, 10), dtype=np.float32)
        density[2:8, 2:8, 2:8] = 1.0
        with patch("cryoem_mrc.tui.Prompt.ask", return_value="0.5"):
            with patch("cryoem_mrc.tui.Confirm.ask", return_value=True):
                val = _prompt_chimerax_contour(
                    density,
                    contour_map=Path("/tmp/avg_half.mrc"),
                )
        self.assertEqual(val, 0.5)

    def test_prompt_chimerax_contour_rejects_empty_mask(self) -> None:
        density = np.zeros((10, 10, 10), dtype=np.float32)
        density[2:8, 2:8, 2:8] = 1.0
        answers = iter(["9.9", "0.5"])
        with patch("cryoem_mrc.tui.Prompt.ask", side_effect=lambda *a, **k: next(answers)):
            with patch("cryoem_mrc.tui.Confirm.ask", return_value=True):
                val = _prompt_chimerax_contour(
                    density,
                    contour_map=Path("/tmp/map.mrc"),
                )
        self.assertEqual(val, 0.5)


if __name__ == "__main__":
    unittest.main()
