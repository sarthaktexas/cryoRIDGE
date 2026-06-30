"""Matplotlib rcParams and small helpers for publication figures."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

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
logging.getLogger("fontTools").setLevel(logging.ERROR)

WORD_PNG_DPI = 600


def apply(ax: plt.Axes) -> None:
    """Strip top/right spines and set tick params."""
    if hasattr(ax, "spines"):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", which="major", labelsize=6, width=0.5, length=2)
        ax.tick_params(axis="both", which="minor", width=0.5)
    else:
        ax.tick_params(labelsize=6, width=0.5, length=2)


def label_panel(ax: plt.Axes, letter: str, *, x: float = -0.1, y: float = 1.05) -> None:
    """Bold panel label (a, b, c…) at upper-left."""
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
    """Panel label for mplot3d axes."""
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


def _has_3d_axes(fig: plt.Figure) -> bool:
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
    """Write PDF (2D only) and 600 dpi PNG."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    stem = out.with_suffix("")
    save_kw = dict(bbox_inches="tight", facecolor="white", **kwargs)
    if not _has_3d_axes(fig):
        fig.savefig(stem.with_suffix(".pdf"), **save_kw)
    fig.savefig(stem.with_suffix(".png"), dpi=WORD_PNG_DPI, **save_kw)


__all__ = [
    "WORD_PNG_DPI",
    "apply",
    "label_panel",
    "label_panel_3d",
    "savefig",
]
