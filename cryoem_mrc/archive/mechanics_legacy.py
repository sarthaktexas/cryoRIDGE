"""Gradient-energy reliability maps and legacy rho-only mechanics (archive comparisons)."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from cryoem_mrc.local_stats import gradient_magnitude, local_laplacian, local_mean


def _uf(x: np.ndarray, size: int) -> np.ndarray:
    return ndimage.uniform_filter(np.asarray(x, dtype=np.float64), size=int(size), mode="nearest")


def _match_dtype(volume: np.ndarray, arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr, dtype=volume.dtype)


def _maybe_window(arr: np.ndarray, window: int) -> np.ndarray:
    w = int(window)
    if w <= 1:
        return np.asarray(arr, dtype=np.float64)
    return _uf(arr, w)


def windowed_smoothness(
    rho: np.ndarray,
    *,
    window: int = 5,
    kappa: float = 1.0,
) -> np.ndarray:
    """
    Windowed gradient energy: (kappa/2) * mean_W(||grad rho||^2).

    ``rho`` must be globally z-scored half-map average (``density_normalized``).
    """
    v = np.asarray(rho)
    grad_sq = gradient_magnitude(v) ** 2
    raw = 0.5 * float(kappa) * grad_sq
    return _match_dtype(v, _maybe_window(raw, window))


def rigidity_like_from_energy(energy: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """Map non-negative energy to (0, 1]: higher = lower energy / more rigid-like."""
    e = np.asarray(energy, dtype=np.float64)
    out = 1.0 / (1.0 + np.maximum(e, 0.0))
    return out.astype(np.asarray(energy).dtype, copy=False)


def lagrangian_density(
    rho: np.ndarray,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
    window: int = 5,
) -> dict[str, np.ndarray]:
    """Legacy rho-only smoothness functional (archive comparisons only)."""
    v = np.asarray(rho)
    grad = gradient_magnitude(v)
    grad_sq = grad * grad
    mean_w = local_mean(v, size=window)
    dev_sq = (v - mean_w) ** 2
    a, b = float(alpha), float(beta)
    kinetic = _match_dtype(v, 0.5 * a * grad_sq)
    potential = _match_dtype(v, 0.5 * b * dev_sq)
    total = _match_dtype(v, kinetic + potential)
    return {
        "kinetic_energy": kinetic,
        "potential_energy": potential,
        "lagrangian_density": total,
    }


def euler_lagrange_residual(
    rho: np.ndarray,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
    window: int = 5,
) -> np.ndarray:
    """Euler-Lagrange residual for the legacy rho-only Lagrangian (archive only)."""
    v = np.asarray(rho)
    lap = local_laplacian(v)
    mean_w = local_mean(v, size=window)
    a, b = float(alpha), float(beta)
    return _match_dtype(v, -a * lap + b * (v - mean_w))


def hamiltonian_density(
    rho: np.ndarray,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
    window: int = 5,
    eps: float = 1e-12,
) -> dict[str, np.ndarray]:
    """Legacy rho-only Hamiltonian (archive comparisons only)."""
    v = np.asarray(rho)
    grad = gradient_magnitude(v)
    grad_sq = grad * grad
    mean_w = local_mean(v, size=window)
    dev_sq = (v - mean_w) ** 2
    a, b = float(alpha), float(beta)
    kinetic = _match_dtype(v, 0.5 * grad_sq / (a + eps))
    potential = _match_dtype(v, 0.5 * b * dev_sq)
    total = _match_dtype(v, kinetic + potential)
    return {
        "hamiltonian_kinetic": kinetic,
        "hamiltonian_potential": potential,
        "hamiltonian": total,
    }


def compute_mechanics_headlines(
    rho: np.ndarray,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
    window: int = 5,
    kappa: float = 1.0,
) -> dict[str, np.ndarray]:
    """Headline mechanics scores for rigidity comparison (minimal peak memory)."""
    v = np.asarray(rho)
    a, b = float(alpha), float(beta)
    w = int(window)
    eps = np.float32(1e-6) if v.dtype == np.float32 else 1e-12

    smoothness = windowed_smoothness(v, window=w, kappa=kappa)

    grad = gradient_magnitude(v)
    grad_sq = grad * grad
    del grad
    mean_w = local_mean(v, size=w)
    dev_sq = (v - mean_w) ** 2

    legacy_l = _match_dtype(v, 0.5 * a * grad_sq + 0.5 * b * dev_sq)
    legacy_h = _match_dtype(v, 0.5 * grad_sq / (a + eps) + 0.5 * b * dev_sq)
    lap = local_laplacian(v)
    el_norm = _match_dtype(v, np.abs(-a * lap + b * (v - mean_w)))
    del lap, mean_w, dev_sq, grad_sq

    return {
        "smoothness": smoothness,
        "legacy_lagrangian_density": legacy_l,
        "legacy_hamiltonian": legacy_h,
        "el_residual_norm": el_norm,
        "rigidity_like_legacy_L": rigidity_like_from_energy(legacy_l),
        "rigidity_like_legacy_H": rigidity_like_from_energy(legacy_h),
        "rigidity_like_el": rigidity_like_from_energy(el_norm),
        "lagrangian_density": legacy_l,
        "hamiltonian": legacy_h,
        "rigidity_like_H": rigidity_like_from_energy(legacy_h),
    }


def compute_mechanics_maps(
    rho: np.ndarray,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
    window: int = 5,
    kappa: float = 1.0,
) -> dict[str, np.ndarray]:
    """Full mechanics feature bundle for correlation analysis."""
    v = np.asarray(rho)
    a, b = float(alpha), float(beta)
    w = int(window)
    eps = np.float32(1e-6) if v.dtype == np.float32 else 1e-12

    grad = gradient_magnitude(v)
    grad_sq = grad * grad
    mean_w = local_mean(v, size=w)
    dev_sq = (v - mean_w) ** 2

    kinetic = _match_dtype(v, 0.5 * a * grad_sq)
    potential = _match_dtype(v, 0.5 * b * dev_sq)
    legacy_l = _match_dtype(v, kinetic + potential)

    ham_kinetic = _match_dtype(v, 0.5 * grad_sq / (a + eps))
    ham_potential = potential
    legacy_h = _match_dtype(v, ham_kinetic + ham_potential)

    lap = local_laplacian(v)
    el = _match_dtype(v, -a * lap + b * (v - mean_w))
    el_norm = _match_dtype(v, np.abs(el.astype(np.float64)))

    return {
        "smoothness": windowed_smoothness(v, window=w, kappa=kappa),
        "legacy_kinetic_energy": kinetic,
        "legacy_potential_energy": potential,
        "legacy_lagrangian_density": legacy_l,
        "legacy_hamiltonian_kinetic": ham_kinetic,
        "legacy_hamiltonian_potential": ham_potential,
        "legacy_hamiltonian": legacy_h,
        "el_residual": el,
        "el_residual_norm": el_norm,
        "rigidity_like_legacy_H": rigidity_like_from_energy(legacy_h),
        "rigidity_like_legacy_L": rigidity_like_from_energy(legacy_l),
        "rigidity_like_el": rigidity_like_from_energy(el_norm),
        "lagrangian_density": legacy_l,
        "hamiltonian": legacy_h,
        "kinetic_energy": kinetic,
        "potential_energy": potential,
        "hamiltonian_kinetic": ham_kinetic,
        "hamiltonian_potential": ham_potential,
        "rigidity_like_H": rigidity_like_from_energy(legacy_h),
        "rigidity_like_L": rigidity_like_from_energy(legacy_l),
    }
