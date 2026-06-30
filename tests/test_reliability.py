"""Tests for cryoem_mrc.reliability."""

from __future__ import annotations

import numpy as np

from cryoem_mrc.reliability import (
    BUILD_ZONE_COLORS,
    BUILD_ZONE_LABELS,
    build_zone_colormap,
    classify_build_zones,
    compute_reliability_maps,
    percentile_rank_in_mask,
)


def test_percentile_rank_in_mask_monotone() -> None:
    vol = np.arange(27, dtype=np.float32).reshape(3, 3, 3)
    mask = vol > 0
    ranks = percentile_rank_in_mask(vol, mask)
    assert ranks[mask].min() > 0
    assert ranks[mask].max() <= 1.0
    assert ranks[~mask].max() == 0.0


def test_build_zone_colors_match_labels() -> None:
    assert BUILD_ZONE_LABELS == {0: "omit", 1: "caution", 2: "build"}
    assert BUILD_ZONE_COLORS[0] == "#E8303A"
    assert BUILD_ZONE_COLORS[1] == "#F5C518"
    assert BUILD_ZONE_COLORS[2] == "#3BBF6A"
    cmap = build_zone_colormap()
    assert tuple(cmap.colors) == tuple(BUILD_ZONE_COLORS[z] for z in (0, 1, 2))


def test_build_zones_three_labels() -> None:
    score = np.linspace(0, 1, 1000, dtype=np.float32).reshape(10, 10, 10)
    mask = np.ones_like(score, dtype=bool)
    z = classify_build_zones(score, mask)
    assert set(np.unique(z[mask])) == {0, 1, 2}


def test_reliability_maps_keys() -> None:
    rho = np.random.default_rng(0).standard_normal((16, 16, 16)).astype(np.float32)
    mask = np.ones(rho.shape, dtype=bool)
    out = compute_reliability_maps(rho, mask=mask)
    assert "reliability_score" in out
    assert "reliability_smoothness" in out
    assert out["reliability_score"][mask].min() > 0
