"""Generate resolvability-metrics schematic (CC vs local FSC vs BlocRes).

Example::

    source .venv/bin/activate
    python scripts/run_metric_schematic_figure.py
    python scripts/run_metric_schematic_figure.py --sync-docs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cryoem_mrc.metric_schematic import export_resolvability_metrics_schematic
from cryoem_mrc.repo_paths import OUTPUTS_ROOT, sync_thesis_doc_figure


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--out",
        type=Path,
        default=OUTPUTS_ROOT / "cohort_summary" / "resolvability_metrics_schematic.png",
    )
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument(
        "--sync-docs",
        action="store_true",
        help="Copy PNG/PDF to docs/figures/fig_metric_resolvability_schematic.png",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out = export_resolvability_metrics_schematic(args.out, dpi=args.dpi)
    print(f"[metric_schematic] wrote {out}", flush=True)
    if args.sync_docs:
        dest = sync_thesis_doc_figure(out, "fig_metric_resolvability_schematic.png")
        print(f"[metric_schematic] synced {dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
