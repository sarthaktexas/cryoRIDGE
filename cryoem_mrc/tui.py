"""Rich interactive terminal UI for ``cryoridge``."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from textwrap import dedent

import numpy as np

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

DEFAULT_OUTPUT_DIRNAME = "cryoridge_out"

_PATH_DRAG_HINT = (
    "[dim]Drag your .mrc / .map from Finder into this window, "
    "or type the path — then press Enter.[/dim]"
)

_CHIMERAX_STEPS = dedent(
    """
    1. Open [bold]ChimeraX[/bold]
    2. [bold]File → Open[/bold] → choose the map shown below
    3. In [bold]Volume Viewer[/bold], move the [bold]Level[/bold] slider until the
       isosurface encloses the macromolecule (not the whole box)
    4. Copy the [bold]Level[/bold] number and paste it here
    """
).strip()

HELP_TEXT = dedent(
    """
    cryoRIDGE — Reliability Inferred from Density Gradient Energy

    INSTALL
      pip install cryoridge

    INTERACTIVE
      cryoridge                  prompt for two half-maps (TTY)
      cryoridge interactive
      cryoridge help               print this reference

    NON-INTERACTIVE
      cryoridge reliability --reference ref.map --half1 h1.map --half2 h2.map \\
        --features features.npz --contour CONTOUR --out-dir out

    Interactive mode averages half-maps, asks you to set the contour in ChimeraX,
    warns when resolution is outside the model-building band, and writes
    ``cryoridge_out/{stem}_reliability.mrc`` and ``*_build_zones.mrc``.

    On macOS, drag each half-map from Finder into the terminal when prompted.

    Advanced: cryoridge features --help, cryoridge reliability --help
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
    console.print(Panel(HELP_TEXT, title="[bold]cryoRIDGE help[/bold]", border_style="cyan"))


def _banner() -> Panel:
    art = Text.from_markup(f"[bold cyan]{_BANNER_ART}[/bold cyan]")
    title = Text.from_markup(
        f"[bold white]cryoRIDGE[/bold white]  [dim]v{__version__}[/dim]\n"
        "[italic]Reliability Inferred from Density Gradient Energy[/italic]"
    )
    body = Group(Align.center(art), Align.center(title))
    return Panel(body, border_style="bright_cyan", box=box.DOUBLE, padding=(1, 2))


def _prompt_path(label: str, *, default: str = "", required: bool = True) -> Path | None:
    console.print(_PATH_DRAG_HINT)
    while True:
        raw = Prompt.ask(f"[bold cyan]{label}[/bold cyan]", default=default)
        if not raw.strip():
            if not required:
                return None
            console.print("[yellow]  required[/yellow]")
            continue
        path = Path(raw.strip()).expanduser()
        if not path.is_file():
            console.print(f"[red]  not found:[/red] {path}")
            continue
        return path


def _mask_fraction(density: np.ndarray, contour: float) -> float:
    from cryoem_mrc.analysis import build_contour_mask

    return float(build_contour_mask(density, contour).mean())


def _chimerax_contour_panel(*, contour_map: Path) -> Panel:
    body = (
        f"{_CHIMERAX_STEPS}\n\n"
        f"[bold]Map to open:[/bold] [cyan]{contour_map}[/cyan]\n\n"
        "[dim]Use the deposited primary map when you have it (same as EMDB / "
        "validation). Otherwise use avg_half.mrc written next to your half-maps.[/dim]"
    )
    return Panel(body, title="Set contour in ChimeraX", border_style="cyan")


def _prompt_chimerax_contour(density: np.ndarray, *, contour_map: Path) -> float:
    console.print()
    console.print(_chimerax_contour_panel(contour_map=contour_map))
    console.print()

    while True:
        try:
            raw = Prompt.ask(
                "[bold cyan]ChimeraX Volume Viewer Level[/bold cyan] "
                "(density units from the map above)",
            )
            val = float(raw.strip())
            if val <= 0:
                console.print("[red]  contour must be positive[/red]")
                continue
            frac = _mask_fraction(density, val)
            console.print(
                f"[dim]  → {frac:.1%} of grid masked at this level[/dim]"
            )
            if frac == 0:
                console.print(
                    "[yellow]  masks no voxels — lower the level or check you opened "
                    "the correct map[/yellow]"
                )
                continue
            if frac > 0.40:
                console.print(
                    "[yellow]  masks a large fraction of the box — surface may be too loose[/yellow]"
                )
            try:
                if Confirm.ask("Use this contour?", default=True):
                    return val
            except (EOFError, KeyboardInterrupt):
                raise
        except ValueError:
            console.print("[red]  enter a number (the Level from Volume Viewer)[/red]")
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
        console.print("[red]cryoridge:[/red] interactive mode requires a TTY.")
        console.print("Try: [cyan]cryoridge help[/cyan]")
        return 1

    console.clear()
    console.print(_banner())
    console.print(
        Panel(
            "[white]Provide two half-maps[/white] (.mrc or .map).\n\n"
            "When prompted, drag each file from Finder into this terminal window "
            "(macOS pastes the full path), or type the path by hand.\n\n"
            "You will set the analysis contour in [bold]ChimeraX[/bold], then "
            "cryoRIDGE writes reliability + build-zone MRCs to "
            f"[cyan]{DEFAULT_OUTPUT_DIRNAME}/[/cyan] next to half-map 1.",
            border_style="green",
        )
    )
    console.print()

    try:
        half1 = _prompt_path("Half-map 1 (.mrc / .map)")
        half2 = _prompt_path("Half-map 2 (.mrc / .map)", default="")
        primary = _prompt_path(
            "Deposited primary map for ChimeraX contour "
            "(optional — Enter to skip and use avg_half.mrc)",
            default="",
            required=False,
        )
    except (EOFError, KeyboardInterrupt):
        console.print("\n[dim]Cancelled.[/dim]")
        return 130

    assert half1 is not None and half2 is not None

    out_dir = half1.parent / DEFAULT_OUTPUT_DIRNAME
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
                run_cryoridge,
                summarize_halfmap_pair,
                write_avg_half_map,
            )
            from cryoem_mrc.io import load_mrc

            context = load_halfmap_pair_context(half1, half2)
            summary = summarize_halfmap_pair(context)
            avg_path = write_avg_half_map(context, out_dir)

        if primary is not None:
            contour_map = primary
            contour_density = load_mrc(primary, dtype=np.float32)
        else:
            contour_map = avg_path
            contour_density = context.avg

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

        contour = _prompt_chimerax_contour(contour_density, contour_map=contour_map)
        console.print(
            f"[dim]Using ChimeraX contour:[/dim] [cyan]{contour:.6g}[/cyan] "
            f"on [cyan]{contour_map.name}[/cyan]"
        )
        console.print()

        with Status(
            "[bold cyan]Running pipeline…[/bold cyan] (this may take a few minutes on large maps)",
            console=console,
            spinner="dots",
        ):
            outputs = run_cryoridge(
                half1,
                half2,
                out_dir=out_dir,
                contour=contour,
                reference_map=primary,
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
        f"[dim]contour (ChimeraX)[/dim]  {outputs['contour']:.6g}\n"
        f"[dim]mask map[/dim]  {outputs['reference_map']}\n"
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
