"""Thesis color palette and matplotlib colormaps.

Full 16-color swatch book plus curated picks for categorical plots, reliability
readouts, and build / caution / omit zones. Import colormap constants from here;
use ``style.nature`` for rcParams and figure helpers (``apply``, ``savefig``).
"""

from __future__ import annotations

from matplotlib.colors import Colormap, LinearSegmentedColormap

# ---------------------------------------------------------------------------
# Master swatch book (user palette — all 16 hex codes)
# ---------------------------------------------------------------------------

MASTER_PALETTE: tuple[str, ...] = (
    "#00A8FA",  # bright blue
    "#00B241",  # deep green
    "#00C0D8",  # cyan
    "#30C8E8",  # light cyan
    "#3BBF6A",  # green
    "#4B6FD4",  # royal blue
    "#8B84D7",  # lavender
    "#BA3EC3",  # magenta
    "#CADE00",  # lime
    "#E8303A",  # red
    "#F0A8C8",  # pink
    "#F4A0A8",  # salmon
    "#F5C518",  # gold
    "#FF2727",  # bright red
    "#FF4500",  # orange-red
    "#FFAE03",  # amber
)

# ---------------------------------------------------------------------------
# Semantic swatches (best picks for recurring thesis semantics)
# ---------------------------------------------------------------------------

ZONE_OMIT = "#E8303A"
ZONE_CAUTION = "#F5C518"
ZONE_BUILD = "#3BBF6A"

BUILD_ZONE_COLORS: dict[int, str] = {
    0: ZONE_OMIT,
    1: ZONE_CAUTION,
    2: ZONE_BUILD,
}

THESIS_RED = ZONE_OMIT
THESIS_YELLOW = ZONE_CAUTION
THESIS_GREEN = ZONE_BUILD
THESIS_BLUE = "#4B6FD4"
THESIS_CYAN = "#30C8E8"
THESIS_PURPLE = "#8B84D7"
THESIS_MAGENTA = "#BA3EC3"
THESIS_PINK = "#F0A8C8"
THESIS_SALMON = "#F4A0A8"
THESIS_AMBER = "#FFAE03"
THESIS_ORANGE = "#FF4500"
THESIS_BRIGHT_BLUE = "#00A8FA"
THESIS_DEEP_GREEN = "#00B241"
THESIS_CYAN_DARK = "#00C0D8"
THESIS_LIME = "#CADE00"
THESIS_BRIGHT_RED = "#FF2727"

# Eight maximally distinct hues for multi-series bar/scatter plots.
_CATEGORICAL: list[str] = [
    THESIS_RED,
    THESIS_BLUE,
    THESIS_GREEN,
    THESIS_MAGENTA,
    THESIS_YELLOW,
    THESIS_CYAN_DARK,
    THESIS_PURPLE,
    THESIS_ORANGE,
]

# Correlation heatmaps: royal blue (negative) → white → red (positive).
_DIVERGING = LinearSegmentedColormap.from_list(
    "thesis_diverging",
    [THESIS_BLUE, "#FFFFFF", THESIS_RED],
)

# Coupling / magnitude: pale → amber → salmon → red.
_SEQUENTIAL = LinearSegmentedColormap.from_list(
    "thesis_sequential",
    ["#FFF9E6", THESIS_AMBER, THESIS_SALMON, THESIS_RED],
)

# Windowed half-map CC: low (unreliable) → high (reliable).
_RELIABILITY_CC = LinearSegmentedColormap.from_list(
    "thesis_reliability_cc",
    [THESIS_RED, THESIS_AMBER, THESIS_YELLOW, THESIS_GREEN],
)

# BlocRes local resolution: sharp (green) → blurry (red).
_RELIABILITY_LOCRES = LinearSegmentedColormap.from_list(
    "thesis_reliability_locres",
    [THESIS_GREEN, THESIS_YELLOW, THESIS_RED],
)

# In-mask percentile reliability score: low (white) → high (purple).
_RELIABILITY_SCORE = LinearSegmentedColormap.from_list(
    "thesis_reliability_score",
    ["#FFFFFF", "#D8D0F0", THESIS_PURPLE, THESIS_MAGENTA],
)

# Locres vs reliability disagreement: agree (white) → disagree (red).
_RELIABILITY_DISAGREEMENT = LinearSegmentedColormap.from_list(
    "thesis_reliability_disagreement",
    ["#FFFFFF", "#F4A0A8", THESIS_RED, THESIS_BRIGHT_RED],
)

PALETTES: dict[str, list[str] | Colormap] = {
    "master": list(MASTER_PALETTE),
    "categorical": _CATEGORICAL,
    "diverging": _DIVERGING,
    "sequential": _SEQUENTIAL,
    "reliability_cc": _RELIABILITY_CC,
    "reliability_locres": _RELIABILITY_LOCRES,
    "reliability_score": _RELIABILITY_SCORE,
    "reliability_disagreement": _RELIABILITY_DISAGREEMENT,
}

RELIABILITY_CMAP_CC = _RELIABILITY_CC
RELIABILITY_CMAP_LOCRES = _RELIABILITY_LOCRES
RELIABILITY_CMAP_SCORE = _RELIABILITY_SCORE
RELIABILITY_CMAP_DISAGREEMENT = _RELIABILITY_DISAGREEMENT

__all__ = [
    "BUILD_ZONE_COLORS",
    "MASTER_PALETTE",
    "PALETTES",
    "RELIABILITY_CMAP_CC",
    "RELIABILITY_CMAP_DISAGREEMENT",
    "RELIABILITY_CMAP_LOCRES",
    "RELIABILITY_CMAP_SCORE",
    "THESIS_AMBER",
    "THESIS_BLUE",
    "THESIS_BRIGHT_BLUE",
    "THESIS_BRIGHT_RED",
    "THESIS_CYAN",
    "THESIS_CYAN_DARK",
    "THESIS_DEEP_GREEN",
    "THESIS_GREEN",
    "THESIS_LIME",
    "THESIS_MAGENTA",
    "THESIS_ORANGE",
    "THESIS_PINK",
    "THESIS_PURPLE",
    "THESIS_RED",
    "THESIS_SALMON",
    "THESIS_YELLOW",
    "ZONE_BUILD",
    "ZONE_CAUTION",
    "ZONE_OMIT",
]
