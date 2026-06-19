#!/usr/bin/env python3
"""Export cohort ρ(Q, V) by protein-class data for the Figma plugin.

Reads ``outputs/cohort_summary/qscore_correlations.csv`` (same source as
``cohort_q_vs_v_by_class.png``).

Usage:
    uv run python scripts/run_reliability_by_class_figma_export.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from cryoem_mrc.reliability_by_class_figma_export import (  # noqa: E402
    FIGMA_JSON,
    write_q_vs_v_by_class_figma_data,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=FIGMA_JSON)
    p.add_argument("--no-patch-ui", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        path = write_q_vs_v_by_class_figma_data(args.out, patch_ui=not args.no_patch_ui)
    except FileNotFoundError as e:
        print(f"[q_vs_v_by_class_figma_export] ERROR: {e}", file=sys.stderr)
        return 2

    data = json.loads(path.read_text())
    n_groups = len(data["panel"]["groups"])
    n_structures = data.get("n_structures") or sum(g["n"] for g in data["panel"]["groups"])
    print(
        f"[q_vs_v_by_class_figma_export] {n_structures} structures · "
        f"{n_groups} protein classes · "
        f"cohort median ρ={data['cohort_median']:+.3f}",
        flush=True,
    )
    print(f"[q_vs_v_by_class_figma_export] wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
