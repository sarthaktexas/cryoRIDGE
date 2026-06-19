"""Tests for avg-vs-primary Guinier sharpening comparison."""

from __future__ import annotations

import numpy as np

from cryoem_mrc.guinier_sharpening import (
    R_MIN_A_DEFAULT,
    apply_global_bfactor_sharpen,
    compare_guinier_b_avg_vs_primary,
    estimate_global_guinier_b,
)


def test_compare_guinier_sharpened_primary_more_negative() -> None:
    rng = np.random.default_rng(0)
    vol = rng.normal(0.0, 1.0, (32, 32, 32)).astype(np.float64)
    vox = (1.0, 1.0, 1.0)
    mask = np.ones(vol.shape, dtype=bool)
    r_max = 4.0

    b_sharp = -40.0
    sharp = apply_global_bfactor_sharpen(vol, vox, b_sharp, r_min_a=R_MIN_A_DEFAULT, r_max_a=r_max)

    est = compare_guinier_b_avg_vs_primary(
        vol, sharp, vox, r_min_a=R_MIN_A_DEFAULT, r_max_a=r_max, mask=mask
    )
    b_avg = estimate_global_guinier_b(vol, vox, r_min_a=R_MIN_A_DEFAULT, r_max_a=r_max, mask=mask).b_factor
    b_pri = estimate_global_guinier_b(
        sharp, vox, r_min_a=R_MIN_A_DEFAULT, r_max_a=r_max, mask=mask
    ).b_factor

    assert np.isclose(est.b_avg_guinier, b_avg)
    assert np.isclose(est.b_primary_guinier, b_pri)
    assert np.isclose(est.b_sharpening_delta, b_pri - b_avg)
    assert est.reported_style_sharpening_b == est.b_sharpening_delta
