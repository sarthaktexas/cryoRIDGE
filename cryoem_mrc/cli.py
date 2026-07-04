"""``cryoridge`` command-line interface."""

from __future__ import annotations

import argparse
import sys
from textwrap import dedent

from cryoem_mrc import __version__
from cryoem_mrc.tui import HELP_TEXT, print_help, run_interactive

_COMMANDS = frozenset({"features", "analyze", "reliability", "interactive", "help"})

_CLI_EPILOG = dedent(
    """
    examples:
      cryoridge                              two half-maps → MRC outputs (TTY)
      cryoridge help                         full command reference
      cryoridge features map.mrc --float32 --out features.npz
      cryoridge analyze --features features.npz --half1 h1.map --half2 h2.map \\
        --reference ref.map --contour 0.116 --out-dir analysis_out
      cryoridge reliability --reference ref.map --half1 h1.map --half2 h2.map \\
        --features features.npz --contour 0.116 --out-dir reliability_out

    install:
      pip install cryoridge

    subcommand help:
      cryoridge features --help
      cryoridge reliability --help
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


def _help(_argv: list[str]) -> int:
    print_help()
    return 0


def _interactive(_argv: list[str]) -> int:
    return run_interactive()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cryoridge",
        description=(
            "cryoRIDGE — Reliability Inferred from Density Gradient Energy "
            "for cryo-EM half-maps."
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
        "interactive",
        help="Prompt for two half-maps (same as running cryoridge with no arguments on a TTY)",
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
        print(f"cryoridge {__version__}")
        return 0

    if argv[0] in ("-h", "--help"):
        parser.print_help()
        return 0

    # Legacy shortcut: ``cryoridge map.mrc [opts]`` (same as ``cryoridge features map.mrc``).
    if argv[0] not in _COMMANDS and str(argv[0]).lower().endswith((".mrc", ".map")):
        return _features(argv)

    if argv[0] not in _COMMANDS:
        parser.print_help(sys.stderr)
        print(f"cryoridge: unknown command {argv[0]!r}", file=sys.stderr)
        print("Try: cryoridge help", file=sys.stderr)
        return 2

    ns, rest = parser.parse_known_args(argv)
    return ns._run(rest)


if __name__ == "__main__":
    raise SystemExit(main())
