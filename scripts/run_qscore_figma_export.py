#!/usr/bin/env python3
"""Export cohort Q vs V summary data for the Figma thesis-q-vs-v plugin.

Reads ``outputs/cohort_summary/qscore_correlations.csv`` (same source as
``qscore_vs_V_cohort.png``).

Usage:
    uv run python scripts/run_qscore_figma_export.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from cryoem_mrc.qscore_figma_export import FIGMA_JSON, write_q_vs_v_figma_data  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=FIGMA_JSON)
    p.add_argument("--no-patch-ui", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        path = write_q_vs_v_figma_data(args.out, patch_ui=not args.no_patch_ui)
    except FileNotFoundError as e:
        print(f"[qscore_figma_export] ERROR: {e}", file=sys.stderr)
        return 2

    data = json.loads(path.read_text())
    stats = data["stats"]
    n_bars = len(data["panels"]["a"]["bars"])
    print(
        f"[qscore_figma_export] {data['n_structures']} structures · "
        f"median ρ={stats['median_rho']:+.3f} · "
        f"ρ vs res={stats['spearman_rho_vs_resolution']:+.3f}",
        flush=True,
    )
    print(f"[qscore_figma_export] panel A bars={n_bars} · panel B points={len(data['panels']['b']['points'])}", flush=True)
    n_sweep = len(data["panels"]["resolution_sweep"]["points"])
    n_std = len(data["panels"]["resolution_standard_bins"]["bars"])
    n_cut = len(data["panels"]["resolution_cutoff"]["series_le"])
    print(
        f"[qscore_figma_export] resolution sweep bins={n_sweep} · "
        f"standard bins={n_std} · cutoffs={n_cut}",
        flush=True,
    )
    print(f"[qscore_figma_export] wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
