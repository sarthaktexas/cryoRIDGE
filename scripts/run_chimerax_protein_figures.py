"""Generate ChimeraX protein surface figures and thesis triptychs.

Builds per-map ChimeraX ``.cxc`` scripts (domain coloring + statistic maps), renders
3D surfaces, and composes publication rows (3D + slices + histograms) like the
MonoRes/ResMap overview reference. Also stacks three maps into domain and
statistic triptychs for the thesis.

Requires UCSF ChimeraX for highest-quality 3D surfaces; when ChimeraX is absent the
script still writes ``.cxc`` files and uses matplotlib Cα fallbacks for previews.

Example::

    source .venv/bin/activate
    python scripts/run_chimerax_protein_figures.py
    python scripts/run_chimerax_protein_figures.py --emd-id 49450 --emd-id 23129
    python scripts/run_chimerax_protein_figures.py --chimerax /Applications/ChimeraX-1.8.app/Contents/bin/ChimeraX
    python scripts/run_chimerax_protein_figures.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from cryoem_mrc.chimerax_figures import (
    DEFAULT_DOMAIN_TRIPTYCH_IDS,
    DEFAULT_STATISTIC_TRIPTYCH_IDS,
    STATISTIC_SPECS,
    compose_triptych,
    find_chimerax_executable,
    generate_protein_figures,
)
from cryoem_mrc.repo_paths import OUTPUTS_ROOT, emd_output_dir, sync_thesis_doc_figure


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--emd-id",
        action="append",
        dest="emd_ids",
        help="Repeatable EMDB ID (default: domain + statistic triptych sets)",
    )
    p.add_argument(
        "--mode",
        choices=("all", "domain", "statistics"),
        default="all",
        help="Generate domain rows, statistic rows, or both (default: all)",
    )
    p.add_argument(
        "--statistic",
        action="append",
        dest="statistics",
        help=f"Statistic key(s) to render (default: all available). Choices: {', '.join(STATISTIC_SPECS)}",
    )
    p.add_argument(
        "--chimerax",
        type=Path,
        default=None,
        help="Path to ChimeraX executable (auto-detected when omitted)",
    )
    p.add_argument(
        "--out-root",
        type=Path,
        default=OUTPUTS_ROOT / "chimerax_figures",
        help="Root directory for per-map outputs and triptychs",
    )
    p.add_argument(
        "--preview",
        action="store_true",
        help="Fast ChimeraX renders (step 2, 640px, supersample 1) — use for drafts",
    )
    p.add_argument(
        "--publication",
        action="store_true",
        help="Full-quality ChimeraX renders (step 1, 900px, supersample 3)",
    )
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Write .cxc scripts only; skip ChimeraX invocation",
    )
    p.add_argument(
        "--no-triptych",
        action="store_true",
        help="Skip stacked A/B/C composite figures",
    )
    p.add_argument(
        "--sync-docs",
        action="store_true",
        help="Copy triptychs into docs/figures/ with thesis-friendly names",
    )
    return p.parse_args(argv)


def _modes_from_arg(mode: str) -> tuple[str, ...]:
    if mode == "all":
        return ("domain", "statistics")
    return (mode,)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    modes = _modes_from_arg(args.mode)

    domain_ids = tuple(args.emd_ids) if args.emd_ids else DEFAULT_DOMAIN_TRIPTYCH_IDS
    stat_ids = tuple(args.emd_ids) if args.emd_ids else DEFAULT_STATISTIC_TRIPTYCH_IDS

    exe = None
    if args.chimerax is not None:
        exe = find_chimerax_executable(args.chimerax)
    else:
        exe = find_chimerax_executable()

    if exe is None and not args.dry_run:
        print(
            "[chimerax_figures] ChimeraX not found — writing .cxc scripts and matplotlib 3D fallbacks.",
            file=sys.stderr,
            flush=True,
        )
    elif exe is not None:
        print(f"[chimerax_figures] using ChimeraX: {exe}", flush=True)

    preview = not args.publication
    if preview:
        print("[chimerax_figures] preview mode (step 4, 640px, local /tmp staging)", flush=True)

    args.out_root.mkdir(parents=True, exist_ok=True)
    failures = 0
    outputs_by_id: dict[str, dict[str, Path]] = {}

    run_ids: list[str] = []
    for eid in list(domain_ids) + list(stat_ids):
        eid = str(eid).strip()
        if eid not in run_ids:
            run_ids.append(eid)

    for emdb_id in run_ids:
        out_dir = emd_output_dir(emdb_id) / "chimerax_figures"
        try:
            outputs = generate_protein_figures(
                emdb_id,
                out_dir=out_dir,
                modes=modes,
                statistics=args.statistics,
                chimerax_exe=exe,
                dry_run=args.dry_run,
                dpi=args.dpi,
                preview=preview,
            )
        except (FileNotFoundError, ValueError) as exc:
            failures += 1
            print(f"[chimerax_figures] ERROR EMD-{emdb_id}: {exc}", file=sys.stderr, flush=True)
            continue

        outputs_by_id[emdb_id] = outputs
        print(f"[chimerax_figures] EMD-{emdb_id}: {', '.join(outputs.keys())}", flush=True)

    domain_rows = [
        outputs_by_id[e]["domain_row"]
        for e in domain_ids
        if e in outputs_by_id and "domain_row" in outputs_by_id[e]
    ]
    stat_rows_by_key: dict[str, list[Path]] = {
        key: [
            outputs_by_id[e][f"stat_{key}"]
            for e in stat_ids
            if e in outputs_by_id and f"stat_{key}" in outputs_by_id[e]
        ]
        for key in STATISTIC_SPECS
    }

    if not args.no_triptych:
        if "domain" in modes and domain_rows:
            trip = args.out_root / "chimerax_domain_triptych.png"
            compose_triptych(domain_rows, out_path=trip, title="Domain-colored deposited models")
            print(f"[chimerax_figures] wrote {trip}", flush=True)
            if args.sync_docs:
                sync_thesis_doc_figure(trip, "fig_chimerax_domain_triptych.png")

        if "statistics" in modes:
            for key, rows in stat_rows_by_key.items():
                if not rows:
                    continue
                spec = STATISTIC_SPECS[key]
                trip = args.out_root / f"chimerax_statistic_triptych_{key}.png"
                compose_triptych(
                    rows,
                    out_path=trip,
                    title=f"Map statistic: {spec.label}",
                )
                print(f"[chimerax_figures] wrote {trip}", flush=True)
                if args.sync_docs and key == "local_resolution":
                    sync_thesis_doc_figure(trip, "fig_chimerax_local_resolution_triptych.png")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
