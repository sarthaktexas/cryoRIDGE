"""Tests for cohort path helpers and density source."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cryoem_mrc.density_source import rho_normalized_for_reliability
from cryoem_mrc.repo_paths import avg_features_npz_path, find_features_npz, primary_features_npz_path


def test_avg_features_npz_path() -> None:
    path = avg_features_npz_path(Path("data/foo"), "49450", 0.116)
    assert path.name == "emd_49450_avg_features_t0116.npz"


def test_primary_features_npz_path() -> None:
    path = primary_features_npz_path(Path("data/foo"), "49450", 0.116)
    assert path.name == "emd_49450_features_t0116.npz"


def test_find_features_npz_prefers_avg_by_default(tmp_path: Path) -> None:
    primary = tmp_path / "emd_49450_features_t0116.npz"
    avg = tmp_path / "emd_49450_avg_features_t0116.npz"
    primary.write_bytes(b"")
    avg.write_bytes(b"")
    assert find_features_npz(tmp_path, "49450", 0.116) == avg


def test_find_features_npz_primary_mode(tmp_path: Path) -> None:
    primary = tmp_path / "emd_49450_features_t0116.npz"
    avg = tmp_path / "emd_49450_avg_features_t0116.npz"
    primary.write_bytes(b"")
    avg.write_bytes(b"")
    assert find_features_npz(tmp_path, "49450", 0.116, density_source="primary") == primary


def test_rho_normalized_avg_half() -> None:
    h1 = np.ones((4, 4, 4), dtype=np.float32)
    h2 = np.ones((4, 4, 4), dtype=np.float32) * 3.0
    rho = rho_normalized_for_reliability(source="avg_half", half1=h1, half2=h2)
    assert rho.shape == h1.shape
    assert np.allclose(rho.mean(), 0.0, atol=1e-5)
