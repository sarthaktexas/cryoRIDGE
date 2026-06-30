"""Sliding-window half-map agreement metrics on a shared grid."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

from style.figures import apply, label_panel, savefig as save_nature

from .io import save_volume_like_reference

WINDOWED_HALFMAP_CORRELATION_KEY = "windowed_halfmap_correlation"
LEGACY_HALFMAP_CORRELATION_KEY = "local_cross_correlation"
WINDOWED_HALFMAP_CORRELATION_LABEL = "windowed half-map correlation"
WINDOWED_HALFMAP_CORRELATION_MRC_NAME = "halfmap_windowed_halfmap_correlation.mrc"
_LEGACY_MRC_NAMES = (
    "half_repro_windowed_halfmap_correlation.mrc",
    "half_repro_local_cross_correlation.mrc",
)


def _uf(x: np.ndarray, size: int) -> np.ndarray:
    return ndimage.uniform_filter(x.astype(np.float64), size=size, mode="nearest")


def half_map_local_metrics(
    half1: np.ndarray,
    half2: np.ndarray,
    *,
    window: int = 5,
    eps: float = 1e-12,
) -> dict[str, np.ndarray]:
    """
    Per-voxel neighborhood statistics (cubic window of side ``window``).

    Returns (Z, Y, X) arrays:

    - ``windowed_halfmap_correlation``: Pearson correlation between the two halves
      in a sliding cubic window
    - ``local_mean_squared_difference``: mean of ``(h1 - h2)²``
    - ``local_variance_difference``: variance of ``(h1 - h2)``
    - ``local_reproducibility_snr``: ``0.5 * (|mean(h1)| + |mean(h2)|) / (std(h1-h2) + eps)``
    """
    if half1.shape != half2.shape:
        raise ValueError(f"Shape mismatch: {half1.shape} vs {half2.shape}")
    a = np.asarray(half1, dtype=np.float64)
    b = np.asarray(half2, dtype=np.float64)
    w = int(window)
    if w < 1:
        raise ValueError("window must be positive")

    m1 = _uf(a, w)
    m2 = _uf(b, w)
    m1sq = _uf(a * a, w)
    m2sq = _uf(b * b, w)
    mab = _uf(a * b, w)
    v1 = np.maximum(m1sq - m1 * m1, 0.0)
    v2 = np.maximum(m2sq - m2 * m2, 0.0)
    cov = mab - m1 * m2
    denom = np.sqrt(v1 * v2) + eps
    local_cc = cov / denom

    diff = a - b
    local_mse = _uf(diff * diff, w)
    md = _uf(diff, w)
    local_var_diff = np.maximum(_uf(diff * diff, w) - md * md, 0.0)
    std_d = np.sqrt(local_var_diff + eps)
    combined = 0.5 * (np.abs(m1) + np.abs(m2))
    snr_like = combined / (std_d + eps)

    dt = np.result_type(half1.dtype, half2.dtype)
    return {
        WINDOWED_HALFMAP_CORRELATION_KEY: local_cc.astype(dt, copy=False),
        "local_mean_squared_difference": local_mse.astype(dt, copy=False),
        "local_variance_difference": local_var_diff.astype(dt, copy=False),
        "local_reproducibility_snr": snr_like.astype(dt, copy=False),
    }


def normalize_halfmap_metric_keys(metrics: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Return metrics with the canonical windowed-halfmap-correlation key."""
    out = dict(metrics)
    if (
        LEGACY_HALFMAP_CORRELATION_KEY in out
        and WINDOWED_HALFMAP_CORRELATION_KEY not in out
    ):
        out[WINDOWED_HALFMAP_CORRELATION_KEY] = out.pop(LEGACY_HALFMAP_CORRELATION_KEY)
    return out


def load_windowed_halfmap_correlation(npz: Mapping[str, np.ndarray]) -> np.ndarray:
    """Load windowed half-map correlation from an NPZ dict."""
    if WINDOWED_HALFMAP_CORRELATION_KEY in npz:
        return np.asarray(npz[WINDOWED_HALFMAP_CORRELATION_KEY], dtype=np.float32)
    if LEGACY_HALFMAP_CORRELATION_KEY in npz:
        return np.asarray(npz[LEGACY_HALFMAP_CORRELATION_KEY], dtype=np.float32)
    raise KeyError(f"NPZ missing {WINDOWED_HALFMAP_CORRELATION_KEY!r}")


def resolve_windowed_halfmap_correlation_mrc(metrics_dir: Path) -> Path | None:
    """Return exported windowed-correlation MRC if present."""
    metrics_dir = Path(metrics_dir)
    candidates = (WINDOWED_HALFMAP_CORRELATION_MRC_NAME, *_LEGACY_MRC_NAMES)
    for name in candidates:
        path = metrics_dir / name
        if path.is_file():
            return path
    return None


def save_half_map_metrics_mrc(
    metrics: dict[str, np.ndarray],
    reference_mrc_path: str | Path,
    out_dir: str | Path,
    *,
    prefix: str = "halfmap_",
) -> dict[str, Path]:
    """Write each metric volume as MRC on the same grid as ``reference_mrc_path``."""
    ref = Path(reference_mrc_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for name, vol in metrics.items():
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
        path = out_dir / f"{prefix}{safe}.mrc"
        save_volume_like_reference(ref, vol, path, extra_label=name[:80])
        written[name] = path
    return written


def plot_half_map_metric_distributions(
    metrics: dict[str, np.ndarray],
    *,
    max_samples: int = 500_000,
    bins: int = 80,
    figsize: tuple[float, float] = (10, 8),
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """Histogram each metric (uniform random subsample of voxels for large maps)."""
    rng = np.random.default_rng(0)
    names = list(metrics.keys())
    n = len(names)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    flat_ax = axes.ravel()
    for i, name in enumerate(names):
        ax = flat_ax[i]
        apply(ax)
        v = np.asarray(metrics[name]).ravel()
        if v.size > max_samples:
            idx = rng.choice(v.size, size=max_samples, replace=False)
            v = v[idx]
        v = v[np.isfinite(v)]
        ax.hist(v, bins=bins, density=True, color="steelblue", alpha=0.85)
        ax.set_title(name)
        ax.set_ylabel("density")
        label_panel(ax, chr(ord("a") + i))
    for j in range(len(names), len(flat_ax)):
        flat_ax[j].set_visible(False)
    fig.tight_layout()
    if save_path is not None:
        save_nature(fig, save_path)
    if show:
        plt.show()
    return fig
