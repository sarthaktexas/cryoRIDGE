#!/usr/bin/env python3
"""Export exemplar-map scatter data for the Figma graphical-abstract plugin.

Default: highest ρ(Q, reliability) in the 2.5–4 Å atomic-building regime.

Usage:
    uv run python scripts/run_graphical_abstract_export.py
    uv run python scripts/run_graphical_abstract_export.py --anchor
    uv run python scripts/run_graphical_abstract_export.py --emd-id 49450
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from cryoem_mrc.graphical_abstract_export import (  # noqa: E402
    ATOMIC_REGIME_HI_A,
    ATOMIC_REGIME_LO_A,
    COHORT_JSON,
    write_graphical_abstract_cohort_data,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--emd-id", type=str, default=None, help="Explicit exemplar map")
    p.add_argument(
        "--anchor",
        action="store_true",
        help="Use thesis anchor EMD-49450 (if in atomic-building regime)",
    )
    p.add_argument("--resolution-lo", type=float, default=ATOMIC_REGIME_LO_A)
    p.add_argument("--resolution-hi", type=float, default=ATOMIC_REGIME_HI_A)
    p.add_argument("--min-in-mask", type=int, default=500)
    p.add_argument("--max-total", type=int, default=1200, help="Scatter points cap")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=COHORT_JSON)
    p.add_argument("--no-patch-ui", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        path = write_graphical_abstract_cohort_data(
            args.out,
            patch_ui=not args.no_patch_ui,
            emd_id=args.emd_id,
            prefer_anchor=args.anchor,
            resolution_lo=args.resolution_lo,
            resolution_hi=args.resolution_hi,
            min_in_mask=args.min_in_mask,
            max_total=args.max_total,
            seed=args.seed,
        )
    except FileNotFoundError as e:
        print(f"[graphical_abstract_export] ERROR: {e}", file=sys.stderr)
        return 2

    data = json.loads(path.read_text())
    scope = data["scope"]
    stats = data["stats"]
    print(
        f"[graphical_abstract_export] exemplar EMD-{scope['emdb_id']} "
        f"({scope['display_name']}, {scope['global_resolution_a']:.2f} Å, {scope['regime_label']})",
        flush=True,
    )
    print(f"[graphical_abstract_export] {scope['selection_note']}", flush=True)
    print(f"[graphical_abstract_export] n={data['n_residues_pooled']:,} Cα", flush=True)
    print(
        f"[graphical_abstract_export] ρ(Q, reliability)={stats['spearman_q_vs_reliability']:+.3f} "
        f"ρ(Q, V)={stats['spearman_q_vs_v']:+.3f} "
        f"ρ(Q, locRes)={stats['spearman_q_vs_locres']:+.3f}",
        flush=True,
    )
    print(f"[graphical_abstract_export] wrote {path}", flush=True)
    if not args.no_patch_ui:
        print("[graphical_abstract_export] embedded cohort data in ui.html", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
