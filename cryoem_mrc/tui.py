"""Interactive terminal menu for ``halfmap-qc`` (stdlib only, no extra deps)."""

from __future__ import annotations

import shlex
import sys
from pathlib import Path
from textwrap import dedent

from cryoem_mrc import __version__
from cryoem_mrc.cohort_pipeline import cohort_emdb_ids, main as cohort_main
from cryoem_mrc.repo_paths import COHORT_MANIFEST

HELP_TEXT = dedent(
    """
    halfmap-qc — half-map reproducibility and local reliability for cryo-EM maps

    INSTALL (PyPI not published yet — use GitHub or a local checkout)
      pip install "git+https://github.com/sarthaktexas/cryoem-halfmap-qc.git@v0.3.2"
      git clone … && cd cryoem-halfmap-qc && pip install -e .

    INTERACTIVE
      halfmap-qc                  launch menu when stdin is a TTY
      halfmap-qc interactive
      halfmap-qc help               print this reference

    COMMANDS
      halfmap-qc features MAP.mrc [--out features.npz] [--float32]
      halfmap-qc analyze --features … --half1 … --half2 … --reference … --contour … --out-dir …
      halfmap-qc reliability --emd-id ID [--contour …] [--features …] [--halfmap-npz …]
      halfmap-qc cohort --pending [--skip-bfactor]
      halfmap-qc cohort --emd-id ID [--skip-bfactor] [--force]
      halfmap-qc cohort-ids [--manifest cohort/manifest.csv]

    Per-command flags: halfmap-qc cohort --help, halfmap-qc features --help, etc.

    Run from the project root where data/ and cohort/manifest.csv live.
    """
).strip()


def print_help() -> None:
    print(HELP_TEXT)


def _prompt(label: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    while True:
        try:
            raw = input(f"{label}{suffix}: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            raise
        if raw:
            return raw
        if default:
            return default
        print("  (required)")


def _prompt_path(label: str, *, must_exist: bool = True) -> Path:
    while True:
        p = Path(_prompt(label)).expanduser()
        if not must_exist or p.is_file():
            return p
        print(f"  not found: {p}")


def _pick_emdb_id(manifest: Path = COHORT_MANIFEST) -> str | None:
    ids = cohort_emdb_ids(manifest)
    if not ids:
        print(f"No cohort entries with local maps (manifest: {manifest})")
        return None
    print("\nCohort maps:")
    for i, eid in enumerate(ids, start=1):
        print(f"  {i:2d}. EMD-{eid}")
    choice = _prompt("EMDB ID or list number", default=ids[0])
    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(ids):
            return ids[idx]
    return choice.strip() or None


def _action_cohort_one() -> int:
    eid = _pick_emdb_id()
    if not eid:
        return 1
    force = _prompt("Re-run if already done? [y/N]", default="n").lower().startswith("y")
    argv = ["--emd-id", eid, "--skip-bfactor"]
    if force:
        argv.append("--force")
    return cohort_main(argv)


def _action_cohort_pending() -> int:
    confirm = _prompt("Run all pending cohort maps? [y/N]", default="n")
    if not confirm.lower().startswith("y"):
        print("Cancelled.")
        return 0
    return cohort_main(["--pending", "--skip-bfactor"])


def _action_features() -> int:
    from cryoem_mrc.cli import _features

    mrc = _prompt_path("Map path (.mrc / .map)")
    out = Path(_prompt("Output .npz", default=str(mrc.with_name(f"{mrc.stem}_features.npz"))))
    return _features([str(mrc), "--float32", "--out", str(out)])


def _action_analyze() -> int:
    from cryoem_mrc.cli import _analyze

    features = _prompt_path("Features .npz")
    half1 = _prompt_path("Half-map 1")
    half2 = _prompt_path("Half-map 2")
    reference = _prompt_path("Reference map")
    contour = _prompt("Contour level", default="0.116")
    out_dir = Path(_prompt("Output directory", default="outputs/analysis"))
    return _analyze(
        [
            "--features",
            str(features),
            "--half1",
            str(half1),
            "--half2",
            str(half2),
            "--reference",
            str(reference),
            "--contour",
            contour,
            "--out-dir",
            str(out_dir),
        ]
    )


def _action_reliability() -> int:
    from cryoem_mrc.cli import _reliability

    eid = _pick_emdb_id()
    if not eid:
        return 1
    contour = _prompt("Contour level", default="0.116")
    return _reliability(["--emd-id", eid, "--contour", contour, "--prune-retired-figures"])


def _action_cohort_ids() -> int:
    from cryoem_mrc.cli import _cohort_ids

    return _cohort_ids([])


def _action_help() -> int:
    print_help()
    return 0


def _action_custom_command() -> int:
    from cryoem_mrc.cli import main as cli_main

    print("Enter flags after the subcommand name.")
    print("Example: cohort --emd-id 49450 --skip-bfactor")
    line = _prompt("subcommand + flags")
    if not line.strip():
        return 0
    return cli_main(shlex.split(line))


def _print_banner() -> None:
    print(f"\n halfmap-qc v{__version__} — interactive mode")
    print(" Run from repo root (data/ + cohort/manifest.csv).\n")


def _print_menu() -> None:
    print(
        dedent(
            """
              1  Cohort pipeline — one EMDB ID
              2  Cohort pipeline — all pending maps
              3  Extract features from a map
              4  Analyze half-maps (CC + correlations)
              5  Export reliability score + build zones
              6  List cohort EMDB IDs
              7  Custom command (type subcommand + flags)
              8  Help / command reference
              0  Exit
            """
        ).strip()
    )


def run_interactive() -> int:
    if not sys.stdin.isatty():
        print("halfmap-qc: interactive mode requires a TTY.", file=sys.stderr)
        print("Try: halfmap-qc help", file=sys.stderr)
        return 1

    handlers = {
        "1": _action_cohort_one,
        "2": _action_cohort_pending,
        "3": _action_features,
        "4": _action_analyze,
        "5": _action_reliability,
        "6": _action_cohort_ids,
        "7": _action_custom_command,
        "8": _action_help,
        "0": None,
    }

    _print_banner()
    while True:
        _print_menu()
        try:
            choice = _prompt("Choice", default="0").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return 0

        if choice == "0":
            print("Bye.")
            return 0

        handler = handlers.get(choice)
        if handler is None:
            print(f"Unknown choice: {choice!r}\n")
            continue

        print()
        try:
            rc = handler()
        except (EOFError, KeyboardInterrupt):
            print("\n(cancelled)\n")
            continue
        print(f"\n→ exit code {rc}\n")
        if rc != 0:
            cont = _prompt("Continue? [Y/n]", default="y")
            if cont.lower().startswith("n"):
                return rc
