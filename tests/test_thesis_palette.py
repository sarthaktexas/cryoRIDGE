"""Tests for style.thesis_palette."""

from __future__ import annotations

from style.thesis_palette import (
    BUILD_ZONE_COLORS,
    MASTER_PALETTE,
    PALETTES,
    RELIABILITY_CMAP_CC,
    RELIABILITY_CMAP_SCORE,
    ZONE_BUILD,
    ZONE_CAUTION,
    ZONE_OMIT,
)


def test_master_palette_has_sixteen_swatches() -> None:
    assert len(MASTER_PALETTE) == 16
    assert all(c.startswith("#") and len(c) == 7 for c in MASTER_PALETTE)


def test_build_zone_semantics() -> None:
    assert BUILD_ZONE_COLORS == {0: ZONE_OMIT, 1: ZONE_CAUTION, 2: ZONE_BUILD}
    assert ZONE_OMIT == "#E8303A"
    assert ZONE_CAUTION == "#F5C518"
    assert ZONE_BUILD == "#3BBF6A"


def test_categorical_subset_of_master() -> None:
    cat = PALETTES["categorical"]
    assert isinstance(cat, list)
    assert len(cat) == 8
    assert all(c in MASTER_PALETTE for c in cat)


def test_reliability_colormaps_registered() -> None:
    assert RELIABILITY_CMAP_CC.name == "thesis_reliability_cc"
    assert RELIABILITY_CMAP_SCORE.name == "thesis_reliability_score"
