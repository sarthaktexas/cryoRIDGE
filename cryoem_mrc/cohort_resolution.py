"""Shared global-resolution bin definitions for cohort stratification."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ResolutionBin:
    lo: float
    hi: float
    key: str
    label: str


# Standard bins used in cohort figures and placement-utility summaries.
COHORT_RESOLUTION_BINS: tuple[ResolutionBin, ...] = (
    ResolutionBin(0.0, 2.5, "le_2.5", "≤2.5 Å"),
    ResolutionBin(2.5, 4.0, "2.5_4", "2.5–4 Å"),
    ResolutionBin(4.0, 6.0, "4_6", "4–6 Å"),
    ResolutionBin(6.0, float("inf"), "gt_6", ">6 Å"),
)

RESOLUTION_BIN_ORDER: tuple[str, ...] = tuple(b.label for b in COHORT_RESOLUTION_BINS)

DEFAULT_CUTOFFS_A: tuple[float, ...] = (3.0, 3.5, 4.0, 4.5, 5.0, 6.0)


def resolution_bin_label(res_a: float) -> str:
    """Human-readable bin label for a deposited global resolution (Å)."""
    if not np.isfinite(res_a):
        return "unknown"
    for b in COHORT_RESOLUTION_BINS:
        if b.lo <= res_a < b.hi:
            return b.label
    return "unknown"


def resolution_bin_key(res_a: float) -> str | None:
    """Machine key (e.g. ``2.5_4``) for a resolution, or ``None`` if unknown."""
    if not np.isfinite(res_a):
        return None
    for b in COHORT_RESOLUTION_BINS:
        if b.lo <= res_a < b.hi:
            return b.key
    return None


def _finite_pairs(pairs: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    return [(g, rho) for g, rho in pairs if np.isfinite(g) and np.isfinite(rho)]


def median_rho_by_resolution_bin(
    pairs: Sequence[tuple[float, float]],
    *,
    bins: Sequence[ResolutionBin] = COHORT_RESOLUTION_BINS,
    prefix: str = "median_spearman",
    metric: str = "q_vs_v",
) -> dict[str, float]:
    """Median Spearman ρ per resolution bin; keys like ``median_spearman_q_vs_v_2.5_4``."""
    arr = _finite_pairs(pairs)
    out: dict[str, float] = {}
    for b in bins:
        vals = [rho for g, rho in arr if b.lo <= g < b.hi]
        if vals:
            out[f"{prefix}_{metric}_{b.key}"] = float(np.median(vals))
    return out


def summarize_resolution_bins(
    pairs: Sequence[tuple[float, float]],
    *,
    bins: Sequence[ResolutionBin] = COHORT_RESOLUTION_BINS,
) -> list[dict[str, float | int | str]]:
    """Per-bin n, median, mean, min, max for reporting tables."""
    arr = _finite_pairs(pairs)
    rows: list[dict[str, float | int | str]] = []
    for b in bins:
        vals = [rho for g, rho in arr if b.lo <= g < b.hi]
        if not vals:
            continue
        rh = np.asarray(vals, dtype=np.float64)
        rows.append(
            {
                "bin_label": b.label,
                "bin_key": b.key,
                "lo_a": b.lo,
                "hi_a": b.hi if np.isfinite(b.hi) else float("nan"),
                "n": len(vals),
                "median_rho": float(np.median(rh)),
                "mean_rho": float(np.mean(rh)),
                "min_rho": float(np.min(rh)),
                "max_rho": float(np.max(rh)),
            }
        )
    return rows


def sweep_resolution_bins(
    pairs: Sequence[tuple[float, float]],
    *,
    width: float = 0.5,
    lo: float = 2.0,
    hi: float = 6.0,
) -> list[dict[str, float | int | str]]:
    """Fixed-width resolution windows (e.g. 0.5 Å) for sensitivity plots."""
    arr = _finite_pairs(pairs)
    rows: list[dict[str, float | int | str]] = []
    edge = lo
    while edge < hi:
        edge_hi = edge + width
        vals = [rho for g, rho in arr if edge <= g < edge_hi]
        if vals:
            rh = np.asarray(vals, dtype=np.float64)
            rows.append(
                {
                    "bin_label": f"{edge:.1f}–{edge_hi:.1f} Å",
                    "lo_a": edge,
                    "hi_a": edge_hi,
                    "n": len(vals),
                    "median_rho": float(np.median(rh)),
                    "mean_rho": float(np.mean(rh)),
                }
            )
        edge = edge_hi
    return rows


def cutoff_median_table(
    pairs: Sequence[tuple[float, float]],
    *,
    cutoffs: Sequence[float] = DEFAULT_CUTOFFS_A,
) -> list[dict[str, float | int]]:
    """
    For each cutoff C: median ρ among maps with res ≤ C vs res > C.

    Shows where cohort signal halves as resolution ceiling is raised.
    """
    arr = _finite_pairs(pairs)
    rows: list[dict[str, float | int]] = []
    for cutoff in cutoffs:
        in_bin = [rho for g, rho in arr if g <= cutoff]
        out_bin = [rho for g, rho in arr if g > cutoff]
        rows.append(
            {
                "cutoff_a": float(cutoff),
                "n_le_cutoff": len(in_bin),
                "median_rho_le_cutoff": float(np.median(in_bin)) if in_bin else float("nan"),
                "n_gt_cutoff": len(out_bin),
                "median_rho_gt_cutoff": float(np.median(out_bin)) if out_bin else float("nan"),
            }
        )
    return rows
