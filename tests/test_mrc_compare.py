"""Tests for MRC file comparison."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import mrcfile
import numpy as np

from cryoem_mrc.mrc_compare import compare_mrc_files


def _write_mrc(path: Path, data: np.ndarray) -> None:
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(np.asarray(data, dtype=np.float32))


class TestMrcCompare(unittest.TestCase):
    def test_byte_identical_files(self) -> None:
        data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = root / "a.mrc"
            b = root / "b.mrc"
            _write_mrc(a, data)
            _write_mrc(b, data)
            report = compare_mrc_files(a, b)
            self.assertTrue(report.byte_identical)
            self.assertTrue(report.same)

    def test_same_values_different_bytes(self) -> None:
        data = np.ones((4, 4, 4), dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = root / "a.mrc"
            b = root / "b.mrc"
            _write_mrc(a, data)
            with mrcfile.new(b, overwrite=True) as mrc:
                mrc.set_data(data.copy())
                mrc.add_label("copied")
            report = compare_mrc_files(a, b)
            self.assertFalse(report.byte_identical)
            self.assertTrue(report.data_identical)
            self.assertTrue(report.same)

    def test_different_voxel_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = root / "a.mrc"
            b = root / "b.mrc"
            _write_mrc(a, np.zeros((3, 3, 3), dtype=np.float32))
            _write_mrc(b, np.ones((3, 3, 3), dtype=np.float32))
            report = compare_mrc_files(a, b)
            self.assertFalse(report.same)
            self.assertFalse(report.data_identical)
            self.assertIsNotNone(report.max_abs_diff)
            self.assertEqual(report.max_abs_diff, 1.0)

    def test_allclose_tolerance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = root / "a.mrc"
            b = root / "b.mrc"
            base = np.linspace(0.0, 1.0, 27, dtype=np.float32).reshape(3, 3, 3)
            _write_mrc(a, base)
            _write_mrc(b, base + 1e-7)
            strict = compare_mrc_files(a, b)
            loose = compare_mrc_files(a, b, rtol=1e-5, atol=1e-5)
            self.assertFalse(strict.same)
            self.assertTrue(loose.same)
