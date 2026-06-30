"""Rich interactive terminal UI for ``halfmap-qc``."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from textwrap import dedent

from rich import box
from rich.align import Align
from rich.console import Console, Group
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.status import Status
from rich.text import Text

from cryoem_mrc import __version__
from cryoem_mrc.emringer_cohort import BUILDING_REGIME_MAX_RESOLUTION_A

console = Console()

HELP_TEXT = dedent(
    """
    halfmap-qc — local reliability scores for cryo-EM density maps

    INSTALL
      pip install cryoem-halfmap-qc

    INTERACTIVE
      halfmap-qc                  prompt for two half-maps (TTY)
      halfmap-qc interactive
      halfmap-qc help               print this reference

    NON-INTERACTIVE
      halfmap-qc reliability --reference ref.map --half1 h1.map --half2 h2.map \\
        --features features.npz --contour CONTOUR --out-dir out

    From two half-maps alone, interactive mode averages them, offers auto or
    manual contour, warns when resolution is outside the model-building band,
    and writes ``halfmap_qc_out/{stem}_reliability.mrc`` and ``*_build_zones.mrc``.

    Advanced: halfmap-qc features --help, halfmap-qc reliability --help
    """
).strip()

_BANNER_ART = r"""
    ░░▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░
  ░▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░
 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
  ░▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░
    ░░▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░
"""


def print_help() -> None:
    console.print(Panel(HELP_TEXT, title="[bold]halfmap-qc help[/bold]", border_style="cyan"))


def _banner() -> Panel:
    art = Text.from_markup(f"[bold cyan]{_BANNER_ART}[/bold cyan]")
    title = Text.from_markup(
        f"[bold white]HALFMAP-QC[/bold white]  [dim]v{__version__}[/dim]\n"
        "[italic]reliability & build zones from half-maps[/italic]"
    )
    body = Group(Align.center(art), Align.center(title))
    return Panel(body, border_style="bright_cyan", box=box.DOUBLE, padding=(1, 2))


def _prompt_path(label: str, *, default: str = "") -> Path:
    while True:
        raw = Prompt.ask(f"[bold cyan]{label}[/bold cyan]", default=default)
        if not raw.strip():
            console.print("[yellow]  required[/yellow]")
            continue
        path = Path(raw.strip()).expanduser()
        if not path.is_file():
            console.print(f"[red]  not found:[/red] {path}")
            continue
        return path


def _prompt_contour(suggested: float) -> float:
    console.print(
        f"[dim]Auto contour suggestion:[/dim] [cyan]{suggested:.6g}[/cyan] "
        "(heuristic on averaged half-map)"
    )
    try:
        use_auto = Confirm.ask("Use auto-detected contour?", default=True)
    except (EOFError, KeyboardInterrupt):
        raise
    if use_auto:
        return float(suggested)

    while True:
        try:
            raw = Prompt.ask(
                "Contour level (density units, same as map)",
                default=f"{suggested:.6g}",
            )
            val = float(raw.strip())
            if val <= 0:
                console.print("[red]  contour must be positive[/red]")
                continue
            return val
        except ValueError:
            console.print("[red]  enter a number[/red]")
        except (EOFError, KeyboardInterrupt):
            raise


def _warn_outside_building_regime(resolution_a: float) -> bool:
    """Warn and ask whether to continue. Returns False when the user cancels."""
    console.print()
    console.print(
        Panel(
            "[bold yellow]Low resolution — outside model-building regime[/bold yellow]\n\n"
            f"Estimated global resolution: [bold]{resolution_a:.2f} Å[/bold] "
            "(masked half-map FSC at 0.143)\n"
            f"Build zones are intended for maps finer than "
            f"[bold]≤ {BUILDING_REGIME_MAX_RESOLUTION_A:g} Å[/bold] "
            "(the cohort uses roughly 2.5–4 Å depositions).\n\n"
            "Reliability scores may still be informative, but "
            "[italic]omit / caution / build[/italic] labels should be "
            "interpreted cautiously on coarse maps.",
            border_style="yellow",
        )
    )
    try:
        return Confirm.ask("Continue anyway?", default=True)
    except (EOFError, KeyboardInterrupt):
        raise


def run_interactive() -> int:
    if not sys.stdin.isatty():
        console.print("[red]halfmap-qc:[/red] interactive mode requires a TTY.")
        console.print("Try: [cyan]halfmap-qc help[/cyan]")
        return 1

    console.clear()
    console.print(_banner())
    console.print(
        Panel(
            "[white]Provide two half-maps.[/white] The tool will average them, "
            "let you choose a contour, then write reliability + build-zone MRCs to "
            "[cyan]halfmap_qc_out/[/cyan] next to half-map 1.",
            border_style="green",
        )
    )
    console.print()

    try:
        half1 = _prompt_path("Half-map 1 (.mrc / .map)")
        half2 = _prompt_path("Half-map 2 (.mrc / .map)", default="")
    except (EOFError, KeyboardInterrupt):
        console.print("\n[dim]Cancelled.[/dim]")
        return 130

    out_dir = half1.parent / "halfmap_qc_out"
    console.print(f"[dim]Output directory:[/dim] [cyan]{out_dir}[/cyan]")
    console.print()

    try:
        with Status(
            "[bold cyan]Loading half-maps…[/bold cyan]",
            console=console,
            spinner="dots",
        ):
            from cryoem_mrc.halfmap_run import (
                load_halfmap_pair_context,
                run_halfmap_qc,
                summarize_halfmap_pair,
            )

            context = load_halfmap_pair_context(half1, half2)
            summary = summarize_halfmap_pair(context)

        if math.isfinite(summary.resolution_a):
            console.print(
                f"[dim]Estimated global resolution:[/dim] "
                f"[cyan]{summary.resolution_a:.2f} Å[/cyan] "
                f"(voxel {summary.voxel_size_a:.3f} Å)"
            )
        else:
            console.print("[dim]Estimated global resolution:[/dim] [yellow]unavailable[/yellow]")

        if math.isfinite(summary.resolution_a) and not summary.in_building_regime:
            if not _warn_outside_building_regime(summary.resolution_a):
                console.print("[dim]Cancelled.[/dim]")
                return 0

        console.print()
        contour = _prompt_contour(summary.suggested_contour)
        console.print()

        with Status(
            "[bold cyan]Running pipeline…[/bold cyan] (this may take a few minutes on large maps)",
            console=console,
            spinner="dots",
        ):
            outputs = run_halfmap_qc(
                half1,
                half2,
                out_dir=out_dir,
                contour=contour,
                context=context,
            )
    except (EOFError, KeyboardInterrupt):
        console.print("\n[dim]Cancelled.[/dim]")
        return 130
    except Exception as exc:
        console.print(f"\n[bold red]Failed:[/bold red] {exc}")
        return 1

    rel = outputs["reliability_mrc"]
    zones = outputs["build_zones_mrc"]
    console.print()
    console.print(Panel(
        f"[bold green]Done[/bold green]\n\n"
        f"[dim]contour[/dim]  {outputs['contour']:.6g}\n"
        f"[dim]reliability[/dim]  {rel}\n"
        f"[dim]build zones[/dim]  {zones}",
        title="Outputs",
        border_style="green",
    ))

    try:
        if Confirm.ask("\n[bold]Process another pair?[/bold]", default=False):
            return run_interactive()
    except (EOFError, KeyboardInterrupt):
        pass

    console.print("[dim]Bye.[/dim]")
    return 0
