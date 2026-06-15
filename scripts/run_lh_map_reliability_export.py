"""Deprecated wrapper — use ``scripts/run_halfmap_reliability_export.py`` instead."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

_NEW = Path(__file__).resolve().parent / "run_halfmap_reliability_export.py"

if __name__ == "__main__":
    print(
        "NOTE: run_lh_map_reliability_export.py is deprecated; "
        "use run_halfmap_reliability_export.py",
        file=sys.stderr,
    )
    runpy.run_path(str(_NEW), run_name="__main__")
