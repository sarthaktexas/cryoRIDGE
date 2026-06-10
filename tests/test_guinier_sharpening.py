"""Tests for Guinier B estimation and sharpening."""

from __future__ import annotations

import unittest

import numpy as np

from cryoem_mrc.guinier_sharpening import (
    apply_global_bfactor_sharpen,
    apply_synthetic_bfactor_blur,
    estimate_global_guinier_b,
    estimate_local_guinier_b_map,
    fit_guinier_b_shells,
    masked_map_ccc,
)


def _gaussian_blob(shape: tuple[int, int, int], sigma_vox: float) -> np.ndarray:
    nz, ny, nx = shape
    zz, yy, xx = np.mgrid[0:nz, 0:ny, 0:nx]
    cz, cy, cx = nz // 2, ny // 2, nx // 2
    r2 = (zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2
    return np.exp(-r2 / (2.0 * sigma_vox**2)).astype(np.float64)


class TestGuinierFit(unittest.TestCase):
    def test_known_slope_recovery(self) -> None:
        slope = -20.0
        shell_r = np.arange(1, 40, dtype=np.float64)
        s2 = (shell_r / (64.0 * 1.0)) ** 2
        amp = np.exp(5.0 + slope * s2)
        fit = fit_guinier_b_shells(
            shell_r,
            amp,
            voxel_size_a=1.0,
            n_ref=64.0,
            r_min_a=15.0,
            r_max_a=3.0,
        )
        self.assertAlmostEqual(fit.slope, slope, delta=0.5)
        self.assertAlmostEqual(fit.b_factor, 4.0 * slope, delta=2.0)

    def test_synthetic_blur_yields_finite_guinier_b(self) -> None:
        shape = (64, 64, 64)
        vox = (1.0, 1.0, 1.0)
        sharp = _gaussian_blob(shape, sigma_vox=2.0)
        blurred = apply_synthetic_bfactor_blur(sharp, vox, 80.0, r_max_a=3.5)
        mask = np.ones(shape, dtype=bool)
        fit = estimate_global_guinier_b(
            blurred,
            vox,
            r_min_a=15.0,
            r_max_a=3.5,
            mask=mask,
        )
        self.assertTrue(np.isfinite(fit.b_factor))
        self.assertGreater(fit.r_squared, 0.9)

    def test_global_sharpen_roundtrip_ccc(self) -> None:
        shape = (64, 64, 64)
        vox = (1.0, 1.0, 1.0)
        target = _gaussian_blob(shape, sigma_vox=2.0)
        b_true = 60.0
        blurred = apply_synthetic_bfactor_blur(target, vox, b_true, r_max_a=3.5)
        fit = estimate_global_guinier_b(blurred, vox, r_min_a=15.0, r_max_a=3.5)
        restored = apply_global_bfactor_sharpen(
            blurred, vox, fit.b_factor, r_min_a=15.0, r_max_a=3.5
        )
        mask = np.ones(shape, dtype=bool)
        ccc = masked_map_ccc(restored, target.astype(np.float32), mask)
        self.assertGreater(ccc, 0.85)

    def test_local_b_map_finite_in_mask(self) -> None:
        shape = (64, 64, 64)
        vox = (1.0, 1.0, 1.0)
        vol = _gaussian_blob(shape, sigma_vox=2.0)
        mask = np.zeros(shape, dtype=bool)
        mask[16:48, 16:48, 16:48] = True
        bmap = estimate_local_guinier_b_map(
            vol,
            voxel_size_zyx=vox,
            r_min_a=15.0,
            r_max_a=3.5,
            patch_size=17,
            stride=8,
            mask=mask,
        )
        inside = bmap[mask]
        inside = inside[np.isfinite(inside)]
        self.assertGreater(inside.size, 10)
        self.assertTrue(np.all(np.isfinite(inside)))


if __name__ == "__main__":
    unittest.main()
