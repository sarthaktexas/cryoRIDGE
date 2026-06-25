"""Conformation-pair coupling layout scores (no matplotlib)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .conformation_pair import (
    compute_domain_mean_coupling,
    compute_per_residue_ca_rmsd,
    get_domain_assignments,
    get_domain_regions_for_pair,
    interior_residue_indices,
)

DEFAULT_COUPLING_LAYOUT_THRESHOLD = 0.08
DEFAULT_QSCORE_COUPLING_LAYOUT_THRESHOLD = DEFAULT_COUPLING_LAYOUT_THRESHOLD


def sorted_conformation_motion(
    pairs: Sequence[tuple[object, object]],
    *,
    in_mask_both: bool = True,
) -> tuple[list[tuple[object, object]], np.ndarray, np.ndarray, list[str]] | None:
    """Matched in-mask pairs with per-residue Cα RMSD and Δreliability (B − A)."""
    use, rmsd = compute_per_residue_ca_rmsd(pairs, in_mask_both=in_mask_both)
    if len(use) < 10:
        return None
    drel = np.array([b.reliability_score - a.reliability_score for a, b in use], dtype=np.float64)
    chains = [a.chain for a, _ in use]
    return use, rmsd, drel, chains


def _local_profile_cross_corr_matrix(
    a: np.ndarray,
    b: np.ndarray,
    *,
    half_window: int,
) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = int(a.size)
    w = 2 * half_window + 1
    if n < w:
        return np.full((n, n), np.nan, dtype=np.float64)

    win_a = np.full((n, w), np.nan, dtype=np.float64)
    win_b = np.full((n, w), np.nan, dtype=np.float64)
    for i in range(n):
        i0, i1 = i - half_window, i + half_window + 1
        if i0 >= 0 and i1 <= n:
            win_a[i] = a[i0:i1]
            win_b[i] = b[i0:i1]

    full_win = np.all(np.isfinite(win_a), axis=1) & np.all(np.isfinite(win_b), axis=1)
    mean_a = np.full((n, 1), np.nan, dtype=np.float64)
    mean_b = np.full((n, 1), np.nan, dtype=np.float64)
    std_a = np.full((n, 1), np.nan, dtype=np.float64)
    std_b = np.full((n, 1), np.nan, dtype=np.float64)
    if full_win.any():
        wa = win_a[full_win]
        wb = win_b[full_win]
        mean_a[full_win] = wa.mean(axis=1, keepdims=True)
        mean_b[full_win] = wb.mean(axis=1, keepdims=True)
        std_a[full_win] = wa.std(axis=1, ddof=0, keepdims=True)
        std_b[full_win] = wb.std(axis=1, ddof=0, keepdims=True)
    valid = (
        full_win
        & np.isfinite(std_a[:, 0])
        & np.isfinite(std_b[:, 0])
        & (std_a[:, 0] > 0)
        & (std_b[:, 0] > 0)
    )
    za = np.where(valid[:, None], (win_a - mean_a) / std_a, 0.0)
    zb = np.where(valid[:, None], (win_b - mean_b) / std_b, 0.0)
    corr = (za @ zb.T) / w
    corr[~valid, :] = np.nan
    corr[:, ~valid] = np.nan
    return corr


def _coupling_interior_slice(
    corr: np.ndarray,
    rmsd: np.ndarray,
    drel: np.ndarray,
    use: list[tuple[object, object]],
    *,
    half_window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[object, object]], np.ndarray]:
    idx = interior_residue_indices(len(use), half_window)
    if idx.size < 10:
        idx = np.arange(len(use), dtype=int)
    sub_use = [use[int(i)] for i in idx]
    return (
        corr[np.ix_(idx, idx)],
        rmsd[idx],
        drel[idx],
        sub_use,
        idx,
    )


def compute_conformation_coupling(
    pairs: Sequence[tuple[object, object]],
    *,
    in_mask_both: bool = True,
    half_window: int | None = None,
) -> dict[str, object] | None:
    packed = sorted_conformation_motion(pairs, in_mask_both=in_mask_both)
    if packed is None:
        return None
    use, rmsd, drel, chains = packed
    n = len(use)
    hw = half_window if half_window is not None else max(5, min(21, n // 25))
    corr = _local_profile_cross_corr_matrix(rmsd, drel, half_window=hw)
    row_mean_abs = np.full(n, np.nan, dtype=np.float64)
    has_finite = np.any(np.isfinite(corr), axis=1)
    if has_finite.any():
        row_mean_abs[has_finite] = np.nanmean(np.abs(corr[has_finite]), axis=1)
    corr_i, rmsd_i, drel_i, use_i, idx = _coupling_interior_slice(
        corr, rmsd, drel, use, half_window=hw
    )
    return {
        "use": use,
        "rmsd": rmsd,
        "drel": drel,
        "chains": chains,
        "corr": corr,
        "row_mean_abs": row_mean_abs,
        "half_window": hw,
        "interior_use": use_i,
        "interior_corr": corr_i,
        "interior_rmsd": rmsd_i,
        "interior_drel": drel_i,
        "interior_indices": idx,
    }


def _hierarchical_cluster(corr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import pdist

    n = int(corr.shape[0])
    if n < 3:
        return np.arange(n, dtype=int), np.zeros((0, 4), dtype=np.float64)
    profiles = np.nan_to_num(corr, nan=0.0)
    d = pdist(profiles, metric="euclidean")
    if not np.all(np.isfinite(d)) or d.size == 0:
        return np.arange(n, dtype=int), np.zeros((0, 4), dtype=np.float64)
    z = linkage(d, method="average")
    return np.asarray(leaves_list(z), dtype=int), z


def compute_coupling_cluster_separation_score(
    corr: np.ndarray,
    *,
    k_max: int = 6,
) -> tuple[float, np.ndarray, np.ndarray]:
    from scipy.cluster.hierarchy import fcluster

    order, z = _hierarchical_cluster(corr)
    n = int(corr.shape[0])
    if n < 10 or z.shape[0] < 2:
        return 0.0, order, z

    best = 0.0
    for k in range(2, min(k_max + 1, max(3, n // 50 + 2))):
        labels = fcluster(z, t=k, criterion="maxclust")
        same_abs: list[float] = []
        diff_abs: list[float] = []
        for i in range(n):
            for j in range(n):
                val = float(corr[i, j])
                if not np.isfinite(val):
                    continue
                mag = abs(val)
                if int(labels[i]) == int(labels[j]):
                    same_abs.append(mag)
                else:
                    diff_abs.append(mag)
        if not same_abs or not diff_abs:
            continue
        ms = float(np.mean(same_abs))
        md = float(np.mean(diff_abs))
        best = max(best, float((ms - md) / (ms + md + 1e-9)))

    return best, order, z


def compute_diagonal_coupling_contrast_score(corr: np.ndarray) -> float:
    n = int(corr.shape[0])
    if n < 3:
        return 0.0
    diag = np.abs(np.diag(corr))
    diag = diag[np.isfinite(diag)]
    off = corr.astype(np.float64, copy=True)
    np.fill_diagonal(off, np.nan)
    off_abs = np.abs(off[np.isfinite(off)])
    if diag.size == 0 or off_abs.size == 0:
        return 0.0
    ms = float(diag.mean())
    md = float(off_abs.mean())
    return float((ms - md) / (ms + md + 1e-9))


def compute_domain_coupling_block_score(
    corr: np.ndarray,
    assignments: dict[str, list[int]],
    domain_order: Sequence[str],
) -> float:
    mat, names = compute_domain_mean_coupling(
        corr, assignments, domain_order=domain_order, metric="mean_abs"
    )
    if len(names) < 2:
        return 0.0
    within: list[float] = []
    across: list[float] = []
    for i in range(len(names)):
        for j in range(len(names)):
            val = float(mat[i, j])
            if not np.isfinite(val):
                continue
            (within if i == j else across).append(val)
    if not within or not across:
        return 0.0
    ms = float(np.mean(within))
    md = float(np.mean(across))
    return float((ms - md) / (ms + md + 1e-9))


def compute_coupling_layout_scores(
    corr: np.ndarray,
    *,
    emdb_a: str | None = None,
    emdb_b: str | None = None,
    interior_use: Sequence[tuple[object, object]] | None = None,
) -> dict[str, float]:
    diag = compute_diagonal_coupling_contrast_score(corr)
    hierarchical, _, _ = compute_coupling_cluster_separation_score(corr)
    domain = float("nan")
    if emdb_a and emdb_b and interior_use is not None:
        regions = get_domain_regions_for_pair(emdb_a, emdb_b)
        if regions:
            domain_order = [reg.name for reg in regions]
            assignments = get_domain_assignments(interior_use, regions)
            domain = compute_domain_coupling_block_score(corr, assignments, domain_order)
    if np.isfinite(domain):
        layout = max(float(domain), diag)
    else:
        layout = diag
    return {
        "diagonal_coupling_score": diag,
        "domain_coupling_score": domain,
        "coupling_layout_score": layout,
        "hierarchical_cluster_score": float(hierarchical),
    }


def compute_qscore_coupling_layout_scores(
    corr: np.ndarray,
    *,
    emdb_a: str | None = None,
    emdb_b: str | None = None,
    interior_use: Sequence[tuple[object, object]] | None = None,
) -> dict[str, float]:
    return compute_coupling_layout_scores(
        corr, emdb_a=emdb_a, emdb_b=emdb_b, interior_use=interior_use
    )


def select_conformation_pair_figure_layout(
    separation_score: float,
    *,
    threshold: float = DEFAULT_COUPLING_LAYOUT_THRESHOLD,
    layout: str = "auto",
) -> str:
    if layout == "block":
        return "block"
    if layout == "domain":
        return "domain"
    if separation_score >= threshold:
        return "block"
    return "domain"
