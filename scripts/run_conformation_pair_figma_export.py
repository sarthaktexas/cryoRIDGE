#!/usr/bin/env python3
"""Export conformation-pair panel B scatter data for the Figma plugin.

Default pair: MsbA outward-facing (EMD-41596) vs inward-facing (EMD-41598).

Usage:
    uv run python scripts/run_conformation_pair_figma_export.py
    uv run python scripts/run_conformation_pair_figma_export.py --emd-a 23129 --emd-b 23130
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from cryoem_mrc.conformation_pair_figma_export import (  # noqa: E402
    DEFAULT_EMD_A,
    DEFAULT_EMD_B,
    FIGMA_JSON,
    write_conformation_pair_figma_data,
)
from cryoem_mrc.repo_paths import COHORT_MANIFEST  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--emd-a", type=str, default=DEFAULT_EMD_A)
    p.add_argument("--emd-b", type=str, default=DEFAULT_EMD_B)
    p.add_argument("--manifest", type=Path, default=COHORT_MANIFEST)
    p.add_argument("--out", type=Path, default=FIGMA_JSON)
    p.add_argument("--no-patch-ui", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        path = write_conformation_pair_figma_data(
            args.out,
            emdb_a=args.emd_a,
            emdb_b=args.emd_b,
            manifest=args.manifest,
            patch_ui=not args.no_patch_ui,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"[conformation_pair_figma_export] ERROR: {e}", file=sys.stderr)
        return 2

    data = json.loads(path.read_text())
    panel = data["panel"]
    print(
        f"[conformation_pair_figma_export] {data['pair_label']} · "
        f"n={data['n_residues']} · ρ={data['spearman_rho']:+.3f}",
        flush=True,
    )
    print(
        f"[conformation_pair_figma_export] panel b points={len(panel['points'])} · "
        f"legend={len(panel.get('legend', []))}",
        flush=True,
    )
    print(f"[conformation_pair_figma_export] wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
