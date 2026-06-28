"""Rich interactive terminal UI for ``halfmap-qc``."""

from __future__ import annotations

import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path
from textwrap import dedent

from rich import box
from rich.align import Align
from rich.console import Console, Group
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.rule import Rule
from rich.status import Status
from rich.table import Table
from rich.text import Text

from cryoem_mrc import __version__

console = Console()

HELP_TEXT = dedent(
    """
    halfmap-qc — local reliability scores for cryo-EM density maps

    INSTALL
      pip install cryoem-halfmap-qc

    INTERACTIVE
      halfmap-qc                  launch menu when stdin is a TTY
      halfmap-qc interactive
      halfmap-qc help               print this reference

    COMMANDS
      halfmap-qc features MAP.mrc [--out features.npz] [--float32]
      halfmap-qc analyze --features … --half1 … --half2 … --reference … --contour … --out-dir …
      halfmap-qc reliability --reference … --half1 … --half2 … --features … \\
        --contour … --out-dir …

    Per-command flags: halfmap-qc features --help, halfmap-qc reliability --help, etc.
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


@dataclass
class MapSession:
    """Remember paths between wizard steps."""

    feature_map: Path | None = None
    features_npz: Path | None = None
    half1: Path | None = None
    half2: Path | None = None
    reference: Path | None = None
    contour: str | None = None
    analysis_dir: Path | None = None
    reliability_dir: Path | None = None
    local_res: Path | None = None
    saved_paths: list[str] = field(default_factory=list)

    def remember(self, label: str, path: Path) -> None:
        entry = f"{label}: {path}"
        if entry not in self.saved_paths:
            self.saved_paths.append(entry)


def print_help() -> None:
    console.print(Panel(HELP_TEXT, title="[bold]halfmap-qc help[/bold]", border_style="cyan"))


def _banner() -> Panel:
    art = Text.from_markup(f"[bold cyan]{_BANNER_ART}[/bold cyan]")
    title = Text.from_markup(
        f"[bold white]HALFMAP-QC[/bold white]  [dim]v{__version__}[/dim]\n"
        "[italic]cryo-EM half-map reliability & build zones[/italic]"
    )
    body = Group(Align.center(art), Align.center(title))
    return Panel(body, border_style="bright_cyan", box=box.DOUBLE, padding=(1, 2))


def _session_panel(session: MapSession) -> Panel | None:
    if not any(
        (
            session.features_npz,
            session.half1,
            session.reference,
            session.analysis_dir,
            session.reliability_dir,
        )
    ):
        return None
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column(style="dim")
    table.add_column(style="cyan")
    rows = [
        ("features", session.features_npz),
        ("half-maps", session.half1 and session.half2),
        ("reference", session.reference),
        ("contour", session.contour),
        ("analysis out", session.analysis_dir),
        ("reliability out", session.reliability_dir),
    ]
    for label, value in rows:
        if value:
            if isinstance(value, tuple):
                display = f"{value[0].name} + {value[1].name}"
            elif isinstance(value, Path):
                display = str(value)
            else:
                display = str(value)
            table.add_row(label, display)
    return Panel(table, title="[bold magenta]session[/bold magenta]", border_style="magenta")


def _menu_table() -> Table:
    table = Table(
        title="[bold]What would you like to do?[/bold]",
        box=box.ROUNDED,
        border_style="cyan",
        show_lines=True,
        expand=True,
    )
    table.add_column("Key", style="bold yellow", width=5, justify="center")
    table.add_column("Action", style="white")
    table.add_column("Description", style="dim")
    table.add_row("1", "Full pipeline", "Features → analyze → reliability (guided)")
    table.add_row("2", "Features only", "Extract local density statistics → .npz")
    table.add_row("3", "Analyze", "Half-map CC + feature correlations")
    table.add_row("4", "Reliability", "Scores, build zones, MRC export")
    table.add_row("5", "Help", "Command reference")
    table.add_row("6", "Custom CLI", "Type a subcommand + flags")
    table.add_row("0", "Exit", "Leave interactive mode")
    return table


def _prompt_path(
    label: str,
    *,
    must_exist: bool = True,
    default: str | None = None,
) -> Path:
    while True:
        raw = Prompt.ask(f"[bold cyan]{label}[/bold cyan]", default=default or "")
        if not raw.strip():
            console.print("[yellow]  required[/yellow]")
            continue
        path = Path(raw.strip()).expanduser()
        if must_exist and not path.is_file():
            console.print(f"[red]  not found:[/red] {path}")
            continue
        return path


def _prompt_dir(label: str, *, default: str | None = None) -> Path:
    while True:
        raw = Prompt.ask(f"[bold cyan]{label}[/bold cyan]", default=default or "")
        if not raw.strip():
            console.print("[yellow]  required[/yellow]")
            continue
        path = Path(raw.strip()).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        return path


def _prompt_contour(session: MapSession) -> str:
    default = session.contour or ""
    while True:
        raw = Prompt.ask("[bold cyan]Contour level[/bold cyan]", default=default)
        if raw.strip():
            session.contour = raw.strip()
            return raw.strip()
        console.print("[yellow]  required (depositor-recommended ρ)[/yellow]")


def _run_step(label: str, fn) -> int:
    console.print()
    console.print(Rule(f"[bold]{label}[/bold]", style="cyan"))
    with Status(f"[bold cyan]{label}…[/bold cyan]", console=console, spinner="dots"):
        rc = fn()
    if rc == 0:
        console.print(f"[bold green]✓[/bold green] {label} finished")
    else:
        console.print(f"[bold red]✗[/bold red] {label} failed (exit {rc})")
    return rc


def _action_features(session: MapSession) -> int:
    from cryoem_mrc.cli import _features

    mrc = _prompt_path("Map path (.mrc / .map)", default=str(session.feature_map or ""))
    session.feature_map = mrc
    out_default = str(session.features_npz or mrc.with_name(f"{mrc.stem}_features.npz"))
    out_raw = Prompt.ask("[bold cyan]Output .npz[/bold cyan]", default=out_default)
    out = Path(out_raw).expanduser()
    session.features_npz = out
    session.remember("features", out)

    def run() -> int:
        return _features([str(mrc), "--float32", "--out", str(out)])

    return _run_step("Feature extraction", run)


def _action_analyze(session: MapSession) -> int:
    from cryoem_mrc.cli import _analyze

    features = _prompt_path(
        "Features .npz",
        default=str(session.features_npz or ""),
    )
    half1 = _prompt_path("Half-map 1", default=str(session.half1 or ""))
    half2 = _prompt_path("Half-map 2", default=str(session.half2 or ""))
    reference = _prompt_path("Reference map", default=str(session.reference or ""))
    contour = _prompt_contour(session)
    out_dir = _prompt_dir("Analysis output directory", default=str(session.analysis_dir or "analysis_out"))

    session.features_npz = features
    session.half1 = half1
    session.half2 = half2
    session.reference = reference
    session.analysis_dir = out_dir
    session.remember("analysis", out_dir)

    argv = [
        "--features", str(features),
        "--half1", str(half1),
        "--half2", str(half2),
        "--reference", str(reference),
        "--contour", contour,
        "--out-dir", str(out_dir),
    ]

    def run() -> int:
        return _analyze(argv)

    return _run_step("Half-map analysis", run)


def _action_reliability(session: MapSession) -> int:
    from cryoem_mrc.cli import _reliability

    features = _prompt_path("Features .npz", default=str(session.features_npz or ""))
    half1 = _prompt_path("Half-map 1", default=str(session.half1 or ""))
    half2 = _prompt_path("Half-map 2", default=str(session.half2 or ""))
    reference = _prompt_path("Reference map", default=str(session.reference or ""))
    contour = _prompt_contour(session)
    out_dir = _prompt_dir(
        "Reliability output directory",
        default=str(session.reliability_dir or "reliability_out"),
    )

    session.features_npz = features
    session.half1 = half1
    session.half2 = half2
    session.reference = reference
    session.reliability_dir = out_dir

    argv = [
        "--reference", str(reference),
        "--half1", str(half1),
        "--half2", str(half2),
        "--features", str(features),
        "--contour", contour,
        "--out-dir", str(out_dir),
    ]

    def run() -> int:
        return _reliability(argv)

    return _run_step("Reliability export", run)


def _action_pipeline(session: MapSession) -> int:
    console.print(
        Panel(
            "[white]Guided run:[/white] extract features, analyze half-maps, export reliability.\n"
            "Paths you enter here are remembered for the next steps.",
            title="[bold]Full pipeline[/bold]",
            border_style="green",
        )
    )
    rc = _action_features(session)
    if rc != 0:
        return rc
    if not Confirm.ask("[bold]Continue to analyze?[/bold]", default=True):
        return rc
    rc = _action_analyze(session)
    if rc != 0:
        return rc
    if not Confirm.ask("[bold]Continue to reliability export?[/bold]", default=True):
        return rc
    return _action_reliability(session)


def _action_custom() -> int:
    from cryoem_mrc.cli import main as cli_main

    console.print(
        "[dim]Example:[/dim] reliability --reference ref.map --half1 h1.map --half2 h2.map "
        "--features f.npz --contour 0.116 --out-dir out"
    )
    line = Prompt.ask("[bold cyan]subcommand + flags[/bold cyan]")
    if not line.strip():
        return 0
    return cli_main(shlex.split(line))


def run_interactive() -> int:
    if not sys.stdin.isatty():
        console.print("[red]halfmap-qc:[/red] interactive mode requires a TTY.")
        console.print("Try: [cyan]halfmap-qc help[/cyan]")
        return 1

    session = MapSession()
    handlers = {
        "1": lambda: _action_pipeline(session),
        "2": lambda: _action_features(session),
        "3": lambda: _action_analyze(session),
        "4": lambda: _action_reliability(session),
        "5": lambda: (print_help(), 0)[1],
        "6": _action_custom,
    }

    while True:
        console.clear()
        console.print(_banner())
        panel = _session_panel(session)
        if panel is not None:
            console.print(panel)
        console.print(_menu_table())

        try:
            choice = Prompt.ask(
                "[bold yellow]Choose[/bold yellow]",
                choices=["0", "1", "2", "3", "4", "5", "6"],
                default="0",
                show_choices=False,
            )
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Bye.[/dim]")
            return 0

        if choice == "0":
            console.print("[dim]Bye.[/dim]")
            return 0

        handler = handlers.get(choice)
        if handler is None:
            console.print(f"[red]Unknown choice {choice!r}[/red]")
            continue

        console.print()
        try:
            rc = handler()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[yellow](cancelled)[/yellow]")
            continue

        console.print()
        if rc != 0 and not Confirm.ask("[bold]Stay in the menu?[/bold]", default=True):
            return rc
        Prompt.ask("[dim]Press Enter to continue[/dim]", default="", show_default=False)
