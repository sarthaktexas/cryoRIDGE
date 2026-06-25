"""Nature-journal matplotlib styling: rcParams, palettes, and figure helpers."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap, LinearSegmentedColormap


_NATURE_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7,
    "axes.titlesize": 8,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.minor.width": 0.5,
    "ytick.minor.width": 0.5,
    "lines.linewidth": 0.75,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "axes.spines.top": False,
    "axes.spines.right": False,
}

mpl.rcParams.update(_NATURE_RC)

# macOS Helvetica/Arial often trigger benign fontTools table-parse warnings on save.
logging.getLogger("fontTools").setLevel(logging.ERROR)

# Thesis figure palette (user-defined; used for cohort scatters, bars, domain bands).
_CATEGORICAL = [
    "#E8303A",
    "#30C8E8",
    "#3BBF6A",
    "#4B6FD4",
    "#8B84D7",  # purple
    "#F0A8C8",
    "#F4A0A8",
    "#F5C518",  # yellow / gold
]

# Named swatches for semantic reuse in bar charts and sign-coded panels.
THESIS_RED = _CATEGORICAL[0]
THESIS_CYAN = _CATEGORICAL[1]
THESIS_GREEN = _CATEGORICAL[2]
THESIS_BLUE = _CATEGORICAL[3]
THESIS_PURPLE = _CATEGORICAL[4]
THESIS_PINK = _CATEGORICAL[5]
THESIS_SALMON = _CATEGORICAL[6]
THESIS_YELLOW = _CATEGORICAL[7]

# Correlation heatmaps: royal blue (negative) → white → red (positive).
_DIVERGING = LinearSegmentedColormap.from_list(
    "thesis_diverging",
    [THESIS_BLUE, "#FFFFFF", THESIS_RED],
)

# Coupling / magnitude sequential: yellow → salmon → red.
_SEQUENTIAL = LinearSegmentedColormap.from_list(
    "thesis_sequential",
    ["#FFF9E6", THESIS_YELLOW, THESIS_SALMON, THESIS_RED],
)

# Reliability readouts: red (low CC / blurry) → yellow → green (high CC / sharp).
_RELIABILITY_CC = LinearSegmentedColormap.from_list(
    "thesis_reliability_cc",
    [THESIS_RED, THESIS_YELLOW, THESIS_GREEN],
)
_RELIABILITY_LOCRES = LinearSegmentedColormap.from_list(
    "thesis_reliability_locres",
    [THESIS_GREEN, THESIS_YELLOW, THESIS_RED],
)

PALETTES: dict[str, list[str] | Colormap] = {
    "categorical": _CATEGORICAL,
    "diverging": _DIVERGING,
    "sequential": _SEQUENTIAL,
    "reliability_cc": _RELIABILITY_CC,
    "reliability_locres": _RELIABILITY_LOCRES,
}

RELIABILITY_CMAP_CC = _RELIABILITY_CC
RELIABILITY_CMAP_LOCRES = _RELIABILITY_LOCRES


def apply(ax: plt.Axes) -> None:
    """Strip top/right spines and set tick params to match Nature style."""
    if hasattr(ax, "spines"):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(
            axis="both",
            which="major",
            labelsize=6,
            width=0.5,
            length=2,
        )
        ax.tick_params(axis="both", which="minor", width=0.5)
    else:
        # Axes3D: no spine API; tick styling only.
        ax.tick_params(labelsize=6, width=0.5, length=2)


def label_panel(ax: plt.Axes, letter: str, *, x: float = -0.1, y: float = 1.05) -> None:
    """Bold panel label (a, b, c…) at upper-left in Nature position."""
    ax.text(
        x,
        y,
        letter,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="bottom",
        ha="right",
    )


def label_panel_3d(ax, letter: str) -> None:
    """Panel label for mplot3d axes (text2D in axes coordinates)."""
    ax.text2D(
        -0.08,
        1.06,
        letter,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="bottom",
        ha="right",
    )


WORD_PNG_DPI = 600


def _has_3d_axes(fig: plt.Figure) -> bool:
    """True when any subplot is an mplot3d Axes3D instance."""
    for ax in fig.get_axes():
        if getattr(ax, "name", None) == "3d":
            return True
        if not hasattr(ax, "spines"):
            return True
    return False


def savefig(
    fig: plt.Figure,
    path: str | Path,
    dpi: int = WORD_PNG_DPI,
    **kwargs,
) -> None:
    """
    Export figures for thesis / publication.

    - **2D figures:** vector PDF + 600 dpi PNG (Word).
    - **3D figures (Axes3D):** 600 dpi PNG only (mplot3d PDF is unwieldy).

    ``dpi`` is accepted for API compatibility; PNG is always written at ``WORD_PNG_DPI``.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    stem = out.with_suffix("")
    save_kw = dict(bbox_inches="tight", facecolor="white", **kwargs)
    if not _has_3d_axes(fig):
        fig.savefig(stem.with_suffix(".pdf"), **save_kw)
    fig.savefig(stem.with_suffix(".png"), dpi=WORD_PNG_DPI, **save_kw)


__all__ = [
    "PALETTES",
    "RELIABILITY_CMAP_CC",
    "RELIABILITY_CMAP_LOCRES",
    "THESIS_BLUE",
    "THESIS_CYAN",
    "THESIS_GREEN",
    "THESIS_PINK",
    "THESIS_PURPLE",
    "THESIS_RED",
    "THESIS_SALMON",
    "THESIS_YELLOW",
    "WORD_PNG_DPI",
    "apply",
    "label_panel",
    "label_panel_3d",
    "savefig",
]
