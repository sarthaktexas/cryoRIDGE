"""Tests for thesis.tv_curvature density-derived T/V maps and block correlations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from thesis.tv_curvature import (
    TV_FEATURE_KEYS,
    density_tv_curvature_maps,
    tv_block_correlations,
)


def _gaussian_blob(shape: tuple[int, int, int], center, sigma: float, amp: float = 1.0) -> np.ndarray:
    z, y, x = np.ogrid[: shape[0], : shape[1], : shape[2]]
    cz, cy, cx = center
    r2 = (z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2
    return amp * np.exp(-r2 / (2.0 * sigma**2))


def test_maps_keys_shape_and_signs() -> None:
    rho = _gaussian_blob((32, 32, 32), (16, 16, 16), sigma=3.0).astype(np.float32)
    maps = density_tv_curvature_maps(rho, chunk_z=12)
    for key in ("T_laplacian", "T_laplacian_abs", "T_vonweizsacker", "V_curvature"):
        assert maps[key].shape == rho.shape
        assert np.isfinite(maps[key]).all()

    c = (16, 16, 16)
    # Laplacian is negative at a density peak (concave); curvature strength positive.
    assert maps["T_laplacian"][c] < 0.0
    assert maps["T_laplacian_abs"][c] >= 0.0
    assert maps["V_curvature"][c] > 0.0
    # |∇ρ|² vanishes at the peak (zero gradient) but is positive on the flanks.
    assert maps["T_vonweizsacker"][c] < maps["T_vonweizsacker"][16, 16, 19]


def test_chunked_matches_full_volume() -> None:
    rng = np.random.default_rng(0)
    rho = rng.standard_normal((30, 28, 26)).astype(np.float64)
    full = density_tv_curvature_maps(rho, chunk_z=None)
    chunked = density_tv_curvature_maps(rho, chunk_z=8)
    for key in full:
        assert np.allclose(full[key], chunked[key], atol=1e-9)


def test_sharper_blob_has_larger_curvature() -> None:
    """V (squared curvature) ∝ 1/σ⁴ for a Gaussian — sharper means larger V."""
    sharp = _gaussian_blob((40, 40, 40), (20, 20, 20), sigma=2.0)
    broad = _gaussian_blob((40, 40, 40), (20, 20, 20), sigma=5.0)
    v_sharp = density_tv_curvature_maps(sharp, chunk_z=None)["V_curvature"][20, 20, 20]
    v_broad = density_tv_curvature_maps(broad, chunk_z=None)["V_curvature"][20, 20, 20]
    assert v_sharp > v_broad


def test_block_correlations_recover_planted_structure() -> None:
    """
    Synthetic residue table where V tracks B (negatively) and T tracks resolution:
    the block result should put the strong couplings on the diagonal.
    """
    rng = np.random.default_rng(1)
    n = 200
    b = rng.uniform(20.0, 120.0, size=n)
    res = rng.uniform(2.0, 5.0, size=n)
    df = pd.DataFrame(
        {
            "in_contour_mask": np.ones(n, dtype=bool),
            "b_factor": b,
            "local_resolution": res,
            "V_curvature": 1.0 / (b**2) + 1e-5 * rng.standard_normal(n),
            "T_vonweizsacker": 1.0 / res + 1e-3 * rng.standard_normal(n),
            "T_laplacian_abs": 1.0 / res + 1e-3 * rng.standard_normal(n),
        }
    )
    result = tv_block_correlations(df, emdb_id="test")
    rec = result.flat_record()
    # V↔B strong negative; T↔resolution strong negative; off-diagonal weak.
    assert rec["rho__V_curvature__vs__b_factor"] < -0.9
    assert rec["rho__T_vonweizsacker__vs__local_resolution"] < -0.9
    assert abs(rec["rho__V_curvature__vs__local_resolution"]) < 0.3
    assert abs(rec["rho__T_vonweizsacker__vs__b_factor"]) < 0.3


def test_feature_keys_constant() -> None:
    assert TV_FEATURE_KEYS == ("T_laplacian_abs", "T_vonweizsacker", "V_curvature")
