"""Tests for style.palette."""

from __future__ import annotations

from style.palette import (
    BUILD_ZONE_COLORS,
    RELIABILITY_CMAP_CC,
    RELIABILITY_CMAP_SCORE,
    ZONE_BUILD,
    ZONE_CAUTION,
    ZONE_OMIT,
)


def test_build_zone_colors() -> None:
    assert BUILD_ZONE_COLORS[0] == ZONE_OMIT
    assert BUILD_ZONE_COLORS[1] == ZONE_CAUTION
    assert BUILD_ZONE_COLORS[2] == ZONE_BUILD


def test_reliability_colormaps_registered() -> None:
    assert RELIABILITY_CMAP_CC.name == "halfmap_reliability_cc"
    assert RELIABILITY_CMAP_SCORE.name == "halfmap_reliability_score"
