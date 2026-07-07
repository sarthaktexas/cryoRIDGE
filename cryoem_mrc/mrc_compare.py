"""Compare two MRC/MAP volumes for byte identity, grid alignment, and voxel values."""

from __future__ import annotations

import filecmp
import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .map_grid import GridAlignmentReport, load_map_grid, verify_grid_alignment


@dataclass
class MrcCompareReport:
    """Result of comparing two on-disk MRC files."""

    path_a: Path
    path_b: Path
    byte_identical: bool | None = None
    byte_hash_a: str | None = None
    byte_hash_b: str | None = None
    grid: GridAlignmentReport | None = None
    shape_match: bool | None = None
    data_identical: bool | None = None
    data_allclose: bool | None = None
    max_abs_diff: float | None = None
    mean_abs_diff: float | None = None
    messages: list[str] = field(default_factory=list)

    @property
    def same(self) -> bool:
        """True when the chosen checks all pass."""
        if self.byte_identical:
            return True
        if self.data_identical is True:
            if self.grid is None or self.grid.ok:
                return True
        if self.data_allclose is True:
            if self.grid is None or self.grid.ok:
                return True
        return False

    def summary_lines(self) -> list[str]:
        lines = [f"A: {self.path_a}", f"B: {self.path_b}"]
        if self.byte_identical is not None:
            lines.append(
                "Byte-identical: yes"
                if self.byte_identical
                else "Byte-identical: no"
            )
        if self.byte_hash_a and self.byte_hash_b and not self.byte_identical:
            lines.append(f"SHA-256 A: {self.byte_hash_a}")
            lines.append(f"SHA-256 B: {self.byte_hash_b}")
        if self.grid is not None:
            lines.append(
                "Grid aligned: yes" if self.grid.ok else "Grid aligned: no"
            )
            lines.extend(f"  {msg}" for msg in self.grid.messages)
        if self.shape_match is not None and not self.shape_match:
            lines.append("Voxel array shape: mismatch")
        if self.data_identical is not None:
            lines.append(
                "Voxel data identical: yes"
                if self.data_identical
                else "Voxel data identical: no"
            )
        if self.data_allclose is not None and not self.data_identical:
            lines.append(
                "Voxel data allclose: yes"
                if self.data_allclose
                else "Voxel data allclose: no"
            )
        if self.max_abs_diff is not None:
            lines.append(f"Max |Δ|: {self.max_abs_diff:.6g}")
        if self.mean_abs_diff is not None:
            lines.append(f"Mean |Δ|: {self.mean_abs_diff:.6g}")
        for msg in self.messages:
            lines.append(msg)
        lines.append("SAME" if self.same else "DIFFERENT")
        return lines


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _compare_voxel_arrays(
    a: np.ndarray,
    b: np.ndarray,
    *,
    rtol: float,
    atol: float,
    chunk_elements: int,
) -> tuple[bool, bool, float | None, float | None]:
    """Return (identical, allclose, max_abs_diff, mean_abs_diff)."""
    if a.shape != b.shape:
        return False, False, None, None

    flat_a = np.asarray(a).ravel()
    flat_b = np.asarray(b).ravel()
    if flat_a.size == 0:
        return True, True, 0.0, 0.0

    identical = True
    allclose = True
    max_abs = 0.0
    total_abs = 0.0
    count = 0

    for start in range(0, flat_a.size, chunk_elements):
        stop = min(flat_a.size, start + chunk_elements)
        slab_a = flat_a[start:stop]
        slab_b = flat_b[start:stop]
        if identical and not np.array_equal(slab_a, slab_b):
            identical = False
        if allclose and not np.allclose(slab_a, slab_b, rtol=rtol, atol=atol):
            allclose = False
        diff = np.abs(slab_a.astype(np.float64) - slab_b.astype(np.float64))
        max_abs = max(max_abs, float(diff.max(initial=0.0)))
        total_abs += float(diff.sum())
        count += diff.size
        if not identical and not allclose:
            break

    mean_abs = total_abs / count if count else 0.0
    return identical, allclose, max_abs, mean_abs


def compare_mrc_files(
    path_a: str | Path,
    path_b: str | Path,
    *,
    check_bytes: bool = True,
    hash_bytes: bool = False,
    check_grid: bool = True,
    check_data: bool = True,
    rtol: float = 0.0,
    atol: float = 0.0,
    voxel_rtol: float = 1e-3,
    voxel_atol: float = 1e-4,
    origin_atol: float = 1e-2,
    chunk_elements: int = 16_777_216,
) -> MrcCompareReport:
    """
    Compare two MRC files.

    By default this checks:
    1. Byte identity (fast path via :func:`filecmp.cmp`)
    2. Grid metadata (shape, voxel size, origin, axis order)
    3. Voxel values (exact match; use ``rtol`` / ``atol`` for float tolerance)

    When files are byte-identical, grid and voxel checks are skipped.
    """
    a_path = Path(path_a)
    b_path = Path(path_b)
    if not a_path.is_file():
        raise FileNotFoundError(a_path)
    if not b_path.is_file():
        raise FileNotFoundError(b_path)

    report = MrcCompareReport(path_a=a_path, path_b=b_path)

    if check_bytes:
        report.byte_identical = filecmp.cmp(a_path, b_path, shallow=False)
        if report.byte_identical:
            report.messages.append("Files are byte-identical.")
            return report
        if hash_bytes:
            report.byte_hash_a = _sha256(a_path)
            report.byte_hash_b = _sha256(b_path)

    grid_a = load_map_grid(a_path, dtype=np.float32)
    grid_b = load_map_grid(b_path, dtype=np.float32)

    if check_grid:
        report.grid = verify_grid_alignment(
            grid_a,
            grid_b,
            voxel_rtol=voxel_rtol,
            voxel_atol=voxel_atol,
            origin_atol=origin_atol,
        )

    if check_data:
        report.shape_match = grid_a.shape_zyx == grid_b.shape_zyx
        if report.shape_match:
            identical, close, max_abs, mean_abs = _compare_voxel_arrays(
                grid_a.data,
                grid_b.data,
                rtol=rtol,
                atol=atol,
                chunk_elements=chunk_elements,
            )
            report.data_identical = identical
            report.data_allclose = close
            report.max_abs_diff = max_abs
            report.mean_abs_diff = mean_abs
        else:
            report.data_identical = False
            report.data_allclose = False
            report.messages.append(
                f"Shape mismatch: {grid_a.shape_zyx} vs {grid_b.shape_zyx}"
            )

    return report
