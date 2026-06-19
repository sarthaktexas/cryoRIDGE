"""Schematic figures explaining resolvability metrics vs sliding-window sampling.

Clarifies windowed half-map CC, in-repo local FSC, and BlocRes local resolution:
what each measures, neighborhood size, and how values attach to voxels.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle

from style.nature import apply, label_panel, savefig as save_nature
from style.thesis_palette import (
    THESIS_BLUE,
    THESIS_CYAN,
    THESIS_GREEN,
    THESIS_PURPLE,
    THESIS_RED,
    THESIS_YELLOW,
)


def _draw_sliding_window_panel(ax) -> None:
    """Panel A: output lives on center voxel; value comes from a neighborhood."""
    apply(ax)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.set_aspect("equal")
    ax.axis("off")
    label_panel(ax, "a", x=-0.02, y=1.02)

    n = 11
    xs = np.linspace(1.2, 8.8, n)
    for i, x in enumerate(xs):
        color = THESIS_YELLOW if i == 5 else "#dddddd"
        ec = THESIS_RED if i == 5 else "#888888"
        lw = 1.2 if i == 5 else 0.5
        ax.add_patch(Rectangle((x - 0.22, 1.6), 0.44, 0.9, facecolor=color, edgecolor=ec, linewidth=lw))
    # 5-voxel window bracket
    ax.add_patch(
        Rectangle(
            (xs[3] - 0.22, 1.35),
            xs[7] - xs[3] + 0.44,
            1.4,
            fill=False,
            edgecolor=THESIS_BLUE,
            linewidth=1.0,
            linestyle="--",
        )
    )
    ax.text(5.0, 3.35, "Sliding neighborhood (example: 5 voxels)", ha="center", fontsize=7, color=THESIS_BLUE)
    ax.text(5.0, 0.55, "Each grid position gets one output number", ha="center", fontsize=6.5)
    ax.text(5.0, 0.15, "written at the center voxel (red outline)", ha="center", fontsize=6.5)


def _metric_card(
    ax,
    *,
    panel: str,
    title: str,
    inputs: str,
    neighborhood: str,
    statistic: str,
    output: str,
    question: str,
    axis_label: str,
    accent: str,
    note: str = "",
) -> None:
    apply(ax)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    label_panel(ax, panel, x=-0.04, y=1.02)
    ax.text(0.5, 0.94, title, ha="center", va="top", fontsize=7.5, fontweight="bold", color=accent)

    rows = [
        ("Inputs", inputs),
        ("Neighborhood", neighborhood),
        ("Statistic", statistic),
        ("Map value", output),
        ("Question", question),
    ]
    y = 0.82
    for label, body in rows:
        ax.text(0.06, y, f"{label}:", fontsize=6, fontweight="bold", va="top")
        ax.text(0.06, y - 0.055, body, fontsize=5.8, va="top", wrap=True)
        y -= 0.13 if len(body) < 55 else 0.16

    ax.add_patch(
        FancyBboxPatch(
            (0.05, 0.04),
            0.9,
            0.12,
            boxstyle="round,pad=0.01",
            facecolor=accent,
            edgecolor="none",
            alpha=0.15,
            transform=ax.transAxes,
        )
    )
    ax.text(0.5, 0.10, axis_label, ha="center", va="center", fontsize=6.5, fontweight="bold", color=accent)
    if note:
        ax.text(0.5, 0.01, note, ha="center", va="bottom", fontsize=5.5, color="#555555")


def _draw_comparison_panel(ax) -> None:
    apply(ax)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    label_panel(ax, "e", x=-0.02, y=1.02)

    headers = ["Metric", "Needs halves?", "Neighborhood", "Domain", "Dense map?", "Coupled to Q?"]
    rows = [
        ["Windowed half-map CC", "Yes", "5³ box", "Real-space correlation", "Yes (every voxel)", "Weak"],
        ["In-repo local FSC", "Yes", "17^3 patch, stride 4", "Fourier FSC to A", "Yes (interpolated)", "Weak"],
        ["BlocRes local res", "Yes", "Tool patches", "Fourier FSC to A", "Yes", "Weak"],
        ["Constraint V (zones)", "No (avg only)", "5^3 box on grad^2", "Real-space gradient", "Yes (every voxel)", "Strong"],
    ]
    col_x = [0.02, 0.24, 0.38, 0.54, 0.72, 0.88]
    y = 0.92
    for j, h in enumerate(headers):
        ax.text(col_x[j], y, h, fontsize=5.8, fontweight="bold", va="top")
    y -= 0.08
    for i, row in enumerate(rows):
        fc = "#f8f8f8" if i % 2 == 0 else "#ffffff"
        ax.add_patch(Rectangle((0.01, y - 0.11), 0.98, 0.12, facecolor=fc, edgecolor="#cccccc", linewidth=0.3, transform=ax.transAxes))
        for j, cell in enumerate(row):
            weight = "bold" if j == 0 else "normal"
            color = THESIS_GREEN if cell == "Strong" else ("#666666" if cell == "Weak" else "black")
            ax.text(col_x[j], y - 0.02, cell, fontsize=5.5, va="top", fontweight=weight, color=color)
        y -= 0.13

    ax.text(
        0.5,
        0.06,
        "Resolvability family (CC, local FSC, BlocRes) answers whether signal is reproducible / resolved. "
        "Constraint V answers placement difficulty — use both axes during building.",
        ha="center",
        va="center",
        fontsize=6,
        wrap=True,
    )


def plot_resolvability_metrics_schematic(*, dpi: int = 300) -> plt.Figure:
    """Multi-panel schematic for thesis methods (CC vs local FSC vs local res)."""
    fig = plt.figure(figsize=(7.2, 8.2))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.35, 1.0], hspace=0.45, wspace=0.35)

    ax_a = fig.add_subplot(gs[0, :])
    _draw_sliding_window_panel(ax_a)

    ax_b = fig.add_subplot(gs[1, 0])
    _metric_card(
        ax_b,
        panel="b",
        title="Windowed half-map CC",
        inputs="Gold-standard half-maps h1, h2",
        neighborhood="5^3 uniform cube centered on each voxel",
        statistic="Pearson correlation of intensities inside the cube",
        output="CC in [-1, 1] at every voxel (dense sliding window)",
        question="Do the two independent reconstructions agree here?",
        axis_label="Resolvability gate",
        accent=THESIS_CYAN,
        note="rho(CC, local FSC) ~ -0.9 on anchor",
    )

    ax_c = fig.add_subplot(gs[1, 1])
    _metric_card(
        ax_c,
        panel="c",
        title="In-repo local FSC",
        inputs="Same half-maps h1, h2",
        neighborhood="17^3 patch with Hann taper; centers every 4 voxels (stride)",
        statistic="Shell-averaged 1D FSC; resolution where FSC < 0.143",
        output="Local resolution (A) interpolated onto all voxels",
        question="What is the highest frequency still consistent between halves?",
        axis_label="Resolvability (frequency view)",
        accent=THESIS_BLUE,
        note="Sparse patch grid, then interpolated map",
    )

    ax_d = fig.add_subplot(gs[1, 2])
    _metric_card(
        ax_d,
        panel="d",
        title="BlocRes local resolution",
        inputs="Half-maps (Bsoft BlocRes)",
        neighborhood="BlocRes patches + FSC window (external tool)",
        statistic="FSC threshold to A (field-standard local res)",
        output="A resolution per voxel (depositor-style heatmap)",
        question="Where does the map resolve at the chosen FSC cutoff?",
        axis_label="Resolvability (external comparator)",
        accent=THESIS_PURPLE,
        note="Weakly coupled to Q and to constraint V",
    )

    ax_e = fig.add_subplot(gs[2, :])
    _draw_comparison_panel(ax_e)

    fig.suptitle(
        "Resolvability readouts share half-maps but differ in neighborhood and domain",
        fontsize=8,
        y=0.995,
    )
    fig.subplots_adjust(top=0.96, bottom=0.04, left=0.06, right=0.98)
    return fig


def export_resolvability_metrics_schematic(out_path: str | Path, *, dpi: int = 300) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plot_resolvability_metrics_schematic(dpi=dpi)
    save_nature(fig, out_path, dpi=dpi)
    plt.close(fig)
    return out_path
