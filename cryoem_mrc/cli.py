"""``halfmap-qc`` command-line interface."""

from __future__ import annotations

import argparse
import sys
from textwrap import dedent

from cryoem_mrc import __version__
from cryoem_mrc.cohort_pipeline import cohort_emdb_ids, main as cohort_main
from cryoem_mrc.repo_paths import COHORT_MANIFEST
from cryoem_mrc.tui import HELP_TEXT, print_help, run_interactive

_COMMANDS = frozenset({"features", "analyze", "reliability", "cohort", "cohort-ids", "interactive", "help"})

_CLI_EPILOG = dedent(
    """
    examples:
      halfmap-qc                              interactive menu (TTY)
      halfmap-qc help                         full command reference
      halfmap-qc features map.mrc --float32 --out features.npz
      halfmap-qc cohort --emd-id 49450 --skip-bfactor
      halfmap-qc cohort --pending --skip-bfactor
      halfmap-qc cohort-ids                   EMDB IDs for SLURM arrays

    install:
      pip install cryoem-halfmap-qc
      pip install "git+https://github.com/sarthaktexas/cryoem-halfmap-qc.git@v0.3.3"

    subcommand help:
      halfmap-qc features --help
      halfmap-qc cohort --help
    """
).strip()


def _features(argv: list[str]) -> int:
    from cryoem_mrc.__main__ import main as features_main

    return features_main(argv)


def _analyze(argv: list[str]) -> int:
    from cryoem_mrc.analysis_driver import main as analysis_main

    return analysis_main(argv)


def _reliability(argv: list[str]) -> int:
    from cryoem_mrc.reliability_driver import main as reliability_main

    return reliability_main(argv)


def _cohort_ids(argv: list[str]) -> int:
    from pathlib import Path

    p = argparse.ArgumentParser(description="Print cohort EMDB IDs (one per line) for SLURM arrays.")
    p.add_argument("--manifest", type=Path, default=COHORT_MANIFEST)
    args = p.parse_args(argv)
    for eid in cohort_emdb_ids(args.manifest):
        print(eid)
    return 0


def _help(_argv: list[str]) -> int:
    print_help()
    return 0


def _interactive(_argv: list[str]) -> int:
    return run_interactive()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="halfmap-qc",
        description=(
            "Half-map QC: density features, windowed half-map correlation, "
            "and reproducibility-based reliability scores for cryo-EM maps."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_CLI_EPILOG,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    sub.add_parser(
        "features",
        help="Extract local density features from a map (avg-half or deposited)",
    ).set_defaults(_run=_features)

    sub.add_parser(
        "analyze",
        help="Half-map metrics and feature vs CC correlations",
    ).set_defaults(_run=_analyze)

    sub.add_parser(
        "reliability",
        help="Reliability score, build zones, and export MRCs for one map",
    ).set_defaults(_run=_reliability)

    sub.add_parser(
        "cohort",
        help="Batch pipeline from cohort/manifest.csv (features → analyze → reliability)",
    ).set_defaults(_run=cohort_main)

    sub.add_parser(
        "cohort-ids",
        help="List EMDB IDs in the active cohort (for parallel cluster jobs)",
    ).set_defaults(_run=_cohort_ids)

    sub.add_parser(
        "interactive",
        help="Interactive menu (same as running halfmap-qc with no arguments on a TTY)",
    ).set_defaults(_run=_interactive)

    sub.add_parser(
        "help",
        help="Print command reference and install notes",
    ).set_defaults(_run=_help)

    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = _build_parser()

    if not argv:
        if sys.stdin.isatty():
            return run_interactive()
        parser.print_help()
        return 0

    if "--version" in argv or "-V" in argv:
        print(f"halfmap-qc {__version__}")
        return 0

    if argv[0] in ("-h", "--help"):
        parser.print_help()
        return 0

    # Legacy shortcut: ``halfmap-qc map.mrc [opts]`` (same as ``halfmap-qc features map.mrc``).
    if argv[0] not in _COMMANDS and str(argv[0]).lower().endswith((".mrc", ".map")):
        return _features(argv)

    if argv[0] not in _COMMANDS:
        parser.print_help(sys.stderr)
        print(f"halfmap-qc: unknown command {argv[0]!r}", file=sys.stderr)
        print("Try: halfmap-qc help", file=sys.stderr)
        return 2

    ns, rest = parser.parse_known_args(argv)
    return ns._run(rest)


if __name__ == "__main__":
    raise SystemExit(main())
