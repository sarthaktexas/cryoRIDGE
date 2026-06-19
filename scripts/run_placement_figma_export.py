#!/usr/bin/env python3
"""Export placement utility figure data for the Figma thesis-placement plugin.

Reads ``placement_predictor_head_to_head.csv`` and ``placement_rank_recovery.csv``.

Usage:
    uv run python scripts/run_placement_figma_export.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from cryoem_mrc.placement_figma_export import FIGMA_JSON, write_placement_figma_data  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=FIGMA_JSON)
    p.add_argument("--no-patch-ui", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        path = write_placement_figma_data(args.out, patch_ui=not args.no_patch_ui)
    except FileNotFoundError as e:
        print(f"[placement_figma_export] ERROR: {e}", file=sys.stderr)
        return 2

    data = json.loads(path.read_text())
    h2h = data["panels"]["head_to_head"]
    rr = data["panels"]["rank_recovery"]
    roc = data["panels"].get("low_q_roc", {})
    print(
        f"[placement_figma_export] {data['n_maps']} maps · Q<{data['q_threshold']:.1f} · "
        f"ROC cohort n={data.get('n_roc_maps', len(roc.get('curves', [])))}",
        flush=True,
    )
    print(
        f"[placement_figma_export] head-to-head: {len(h2h['predictors'])} predictors · "
        f"{len(h2h['panels'])} subpanels",
        flush=True,
    )
    print(
        f"[placement_figma_export] rank recovery: {len(rr['bars'])} proxies · "
        f"ROC curves: {len(roc.get('curves', []))}",
        flush=True,
    )
    print(f"[placement_figma_export] wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
