"""Thesis-only ChimeraX figures for MgtA EMD-49450 and its conformational partner.

Generates ChimeraX renders (if missing), then exports **individual pipeline panels**
(3D render + slice per step) for manual assembly in Illustrator/InDesign.

Step 1 is the gray map shell + deposited cartoon (not domain colors). Domain
coloring is reserved for the conformation-pair summary figure.

Default pair: EMD-49450 (E2P+E1) vs EMD-48923 (E2·Mg·BeF₃).

Example::

    source .venv/bin/activate
    python scripts/run_chimerax_mgta_pair_thesis_figures.py
    python scripts/run_chimerax_mgta_pair_thesis_figures.py --skip-render
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from cryoem_mrc.chimerax_figures import (
    MGTA_CONFORMATION_PAIR,
    export_pipeline_panel_assets,
    find_chimerax_executable,
    generate_protein_figures,
)
from cryoem_mrc.repo_paths import OUTPUTS_ROOT, emd_output_dir


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--emd-a", type=str, default=MGTA_CONFORMATION_PAIR[0])
    p.add_argument("--emd-b", type=str, default=MGTA_CONFORMATION_PAIR[1])
    p.add_argument("--out-dir", type=Path, default=OUTPUTS_ROOT / "chimerax_figures" / "mgta_pair")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--skip-render", action="store_true", help="Only export from existing renders")
    p.add_argument("--publication", action="store_true", help="Full-quality ChimeraX (slow)")
    return p.parse_args(argv)


def _ensure_renders(
    emdb_id: str,
    *,
    chimerax_exe,
    preview: bool,
    dpi: int,
) -> None:
    out_dir = emd_output_dir(emdb_id) / "chimerax_figures"
    generate_protein_figures(
        emdb_id,
        out_dir=out_dir,
        modes=("map_shell", "statistics"),
        chimerax_exe=chimerax_exe,
        dry_run=False,
        dpi=dpi,
        preview=preview,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    emd_a = str(args.emd_a).strip()
    emd_b = str(args.emd_b).strip()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    exe = find_chimerax_executable()
    preview = not args.publication
    if exe is None and not args.skip_render:
        print("[mgta_pair] ChimeraX not found; need existing renders or install ChimeraX", file=sys.stderr)
        return 2
    if exe is not None:
        print(f"[mgta_pair] ChimeraX: {exe}", flush=True)

    if not args.skip_render:
        for emdb_id in (emd_a, emd_b):
            print(f"[mgta_pair] rendering EMD-{emdb_id}...", flush=True)
            try:
                _ensure_renders(emdb_id, chimerax_exe=exe, preview=preview, dpi=args.dpi)
            except (FileNotFoundError, ValueError) as exc:
                print(f"[mgta_pair] ERROR EMD-{emdb_id}: {exc}", file=sys.stderr)
                return 2

    for emdb_id in (emd_a, emd_b):
        try:
            outputs = export_pipeline_panel_assets(emdb_id, out_dir=args.out_dir, dpi=args.dpi)
        except FileNotFoundError as exc:
            print(f"[mgta_pair] ERROR EMD-{emdb_id}: {exc}", file=sys.stderr)
            return 2
        panel_dir = args.out_dir / emdb_id
        print(f"[mgta_pair] wrote {len(outputs)} assets -> {panel_dir}/", flush=True)
        for name, path in sorted(outputs.items()):
            print(f"  {name}: {path.name}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
