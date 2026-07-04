"""Figure color palette and matplotlib colormaps for cryoridge outputs."""

from __future__ import annotations

from matplotlib.colors import Colormap, LinearSegmentedColormap

MASTER_PALETTE: tuple[str, ...] = (
    "#00A8FA",
    "#00B241",
    "#00C0D8",
    "#30C8E8",
    "#3BBF6A",
    "#4B6FD4",
    "#8B84D7",
    "#BA3EC3",
    "#CADE00",
    "#E8303A",
    "#F0A8C8",
    "#F4A0A8",
    "#F5C518",
    "#FF2727",
    "#FF4500",
    "#FFAE03",
)

ZONE_OMIT = "#E8303A"
ZONE_CAUTION = "#F5C518"
ZONE_BUILD = "#3BBF6A"

BUILD_ZONE_COLORS: dict[int, str] = {
    0: ZONE_OMIT,
    1: ZONE_CAUTION,
    2: ZONE_BUILD,
}

PALETTE_BLUE = "#4B6FD4"
PALETTE_CYAN = "#30C8E8"
PALETTE_PURPLE = "#8B84D7"
PALETTE_MAGENTA = "#BA3EC3"
PALETTE_PINK = "#F0A8C8"
PALETTE_SALMON = "#F4A0A8"
PALETTE_AMBER = "#FFAE03"
PALETTE_ORANGE = "#FF4500"
PALETTE_BRIGHT_BLUE = "#00A8FA"
PALETTE_DEEP_GREEN = "#00B241"
PALETTE_CYAN_DARK = "#00C0D8"
PALETTE_LIME = "#CADE00"
PALETTE_BRIGHT_RED = "#FF2727"

_CATEGORICAL: list[str] = [
    ZONE_OMIT,
    PALETTE_BLUE,
    ZONE_BUILD,
    PALETTE_MAGENTA,
    ZONE_CAUTION,
    PALETTE_CYAN_DARK,
    PALETTE_PURPLE,
    PALETTE_ORANGE,
]

_DIVERGING = LinearSegmentedColormap.from_list(
    "halfmap_diverging",
    [PALETTE_BLUE, "#FFFFFF", ZONE_OMIT],
)

_SEQUENTIAL = LinearSegmentedColormap.from_list(
    "halfmap_sequential",
    ["#FFF9E6", PALETTE_AMBER, PALETTE_SALMON, ZONE_OMIT],
)

_RELIABILITY_CC = LinearSegmentedColormap.from_list(
    "halfmap_reliability_cc",
    [ZONE_OMIT, PALETTE_AMBER, ZONE_CAUTION, ZONE_BUILD],
)

_RELIABILITY_LOCRES = LinearSegmentedColormap.from_list(
    "halfmap_reliability_locres",
    [ZONE_BUILD, ZONE_CAUTION, ZONE_OMIT],
)

_RELIABILITY_SCORE = LinearSegmentedColormap.from_list(
    "halfmap_reliability_score",
    ["#FFFFFF", "#D8D0F0", PALETTE_PURPLE, PALETTE_MAGENTA],
)

_RELIABILITY_DISAGREEMENT = LinearSegmentedColormap.from_list(
    "halfmap_reliability_disagreement",
    ["#FFFFFF", "#F4A0A8", ZONE_OMIT, PALETTE_BRIGHT_RED],
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
    "PALETTE_AMBER",
    "PALETTE_BLUE",
    "PALETTE_BRIGHT_BLUE",
    "PALETTE_BRIGHT_RED",
    "PALETTE_CYAN",
    "PALETTE_CYAN_DARK",
    "PALETTE_DEEP_GREEN",
    "PALETTE_LIME",
    "PALETTE_MAGENTA",
    "PALETTE_ORANGE",
    "PALETTE_PINK",
    "PALETTE_PURPLE",
    "PALETTE_SALMON",
    "ZONE_BUILD",
    "ZONE_CAUTION",
    "ZONE_OMIT",
    "RELIABILITY_CMAP_CC",
    "RELIABILITY_CMAP_DISAGREEMENT",
    "RELIABILITY_CMAP_LOCRES",
    "RELIABILITY_CMAP_SCORE",
]
