#!/usr/bin/env python3
"""Export cohort cross-metric figure data for the Figma thesis-cross-metric plugin.

Reads ``outputs/emd_<ID>/metric_comparison/cross_metric_correlations.csv`` (same
source as ``cohort_cross_metric_median.png``).

Usage:
    uv run python scripts/run_cross_metric_figma_export.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from cryoem_mrc.cross_metric_figma_export import FIGMA_JSON, write_cross_metric_figma_data  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=FIGMA_JSON)
    p.add_argument("--no-patch-ui", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        path = write_cross_metric_figma_data(args.out, patch_ui=not args.no_patch_ui)
    except FileNotFoundError as e:
        print(f"[cross_metric_figma_export] ERROR: {e}", file=sys.stderr)
        return 2

    data = json.loads(path.read_text())
    heat = data["panels"]["median_heatmap"]
    locres = data["panels"]["locres_pairs"]
    n_cells = len(heat.get("cells", []))
    n_maps = data.get("n_structures", 0)
    print(
        f"[cross_metric_figma_export] {n_maps} maps · "
        f"{len(heat.get('row_labels', []))}×{len(heat.get('col_labels', []))} heatmap · "
        f"{n_cells} annotated cells",
        flush=True,
    )
    print(
        f"[cross_metric_figma_export] locres pairs · "
        f"{len(locres.get('series', []))} series",
        flush=True,
    )
    print(f"[cross_metric_figma_export] wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
