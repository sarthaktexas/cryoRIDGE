"""Tests for feature pipeline embed/crop edge cases."""

from __future__ import annotations

import numpy as np

from cryoem_mrc.mask_bbox import VolumeBbox, embed_array
from cryoem_mrc.pipeline import _embed_volume_features


def test_embed_volume_features_skips_1d_metadata() -> None:
    full_shape = (10, 10, 10)
    bbox = VolumeBbox(1, 9, 1, 9, 1, 9)
    block = {
        "density_normalized": np.ones(bbox.shape, dtype=np.float32),
        "multiscale_sigmas": np.array([0.5, 1.0, 2.0], dtype=np.float64),
    }
    raw_full = np.zeros(full_shape, dtype=np.float32)
    out = _embed_volume_features(full_shape, bbox, block, raw_full=raw_full)
    assert out["density_raw"].shape == full_shape
    assert out["multiscale_sigmas"].shape == (3,)
    assert out["density_normalized"].shape == full_shape
    assert float(out["density_normalized"][5, 5, 5]) == 1.0
