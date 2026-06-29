"""Follow-up: sphere-averaged λ_min, wider V-failure stratum, partial ρ(Q, λ_min | V).

Extends ``run_qscore_hessian_ablation.py`` without recomputing the first-pass table.

Example::

    source .venv/bin/activate
    PYTHONUNBUFFERED=1 python scripts/run_qscore_hessian_followup.py
    python scripts/run_qscore_hessian_followup.py --resume --emd-id 49450
"""

from __future__ import annotations

import argparse
import csv
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage, stats

from cryoem_mrc.density_source import zscore_halfmap_average
from cryoem_mrc.hessian import density_hessian_scalar_maps
from cryoem_mrc.local_stats import gradient_magnitude
from cryoem_mrc.map_grid import load_full_and_half_maps, load_map_grid
from cryoem_mrc.repo_paths import COHORT_MANIFEST, OUTPUTS_ROOT, resolve_halfmap_reliability_dir
from cryoem_mrc.structure_validation import iter_ca_residues, load_cohort_manifest_row, sample_volume_at_ca

QSCORE_PANEL_EXCLUDE = frozenset({"33736"})
DEFAULT_WINDOW = 5
DEFAULT_CHUNK_Z = 32
DEFAULT_SPHERE_A = 2.0

FIELDNAMES = [
    "emdb_id",
    "n_used",
    "n_v_failure_tertile",
    "n_v_failure_half",
    "rho_q_neg_lam_min_voxel",
    "rho_q_neg_lam_min_sphere",
    "rho_q_ridge_index_sphere",
    "partial_rho_q_neg_lam_min_given_v",
    "partial_rho_q_ridge_index_given_v",
    "rho_q_neg_lam_min_v_failure_tertile",
    "rho_q_neg_lam_min_v_failure_half",
    "rho_q_v_v_failure_half",
    "partial_rho_q_neg_lam_min_given_v_v_failure_half",
    "delta_neg_lam_sphere_minus_v_v_failure_half",
]

OUT_CSV = OUTPUTS_ROOT / "cohort_summary" / "qscore_hessian_followup.csv"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, default=COHORT_MANIFEST)
    p.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    p.add_argument("--chunk-z", type=int, default=DEFAULT_CHUNK_Z)
    p.add_argument("--sphere-a", type=float, default=DEFAULT_SPHERE_A, help="Cα sphere radius (Å)")
    p.add_argument("--emd-id", type=str, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--force", action="store_true")
    return p.parse_args(argv)


def _constraint_v(grad_mag: np.ndarray, *, window: int) -> np.ndarray:
    v_raw = (0.5 * grad_mag * grad_mag).astype(np.float64)
    if int(window) > 1:
        v_raw = ndimage.uniform_filter(v_raw, size=int(window), mode="nearest")
    return v_raw.astype(np.float32, copy=False)


def _spearman(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> tuple[float, int]:
    m = mask & np.isfinite(x) & np.isfinite(y)
    n = int(m.sum())
    if n < 10:
        return float("nan"), n
    rho, _ = stats.spearmanr(x[m], y[m])
    return float(rho), n


def _partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray, mask: np.ndarray) -> float:
    m = mask & np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if int(m.sum()) < 10:
        return float("nan")
    xr, yr, zr = stats.rankdata(x[m]), stats.rankdata(y[m]), stats.rankdata(z[m])
    r_xy = np.corrcoef(xr, yr)[0, 1]
    r_xz = np.corrcoef(xr, zr)[0, 1]
    r_yz = np.corrcoef(yr, zr)[0, 1]
    d = (1.0 - r_xz * r_xz) * (1.0 - r_yz * r_yz)
    if d <= 0:
        return float("nan")
    return float((r_xy - r_xz * r_yz) / np.sqrt(d))


def _v_failure_mask(
    q: np.ndarray,
    v: np.ndarray,
    in_mask: np.ndarray,
    *,
    v_quantile: float,
    q_quantile: float,
) -> np.ndarray:
    """Residues with high V (>= v_quantile) and low Q (<= q_quantile) among in-mask."""
    m = in_mask & np.isfinite(q) & np.isfinite(v)
    out = np.zeros_like(in_mask, dtype=bool)
    if int(m.sum()) < 30:
        return out
    q_sub = q[m]
    v_sub = v[m]
    q_cut = float(np.percentile(q_sub, q_quantile))
    v_cut = float(np.percentile(v_sub, v_quantile))
    out[m] = (q_sub <= q_cut) & (v_sub >= v_cut)
    return out


def _load_q_table(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["q_score"] = pd.to_numeric(df["q_score"], errors="coerce")
    df["in_contour_mask"] = df["in_contour_mask"].astype(int).astype(bool)
    return df


def _emd_ids_from_qscore_cohort() -> list[str]:
    ids: list[str] = []
    csv_path = OUTPUTS_ROOT / "cohort_summary" / "qscore_correlations.csv"
    if csv_path.is_file():
        with csv_path.open(newline="") as f:
            for row in csv.DictReader(f):
                eid = str(row["emdb_id"]).strip()
                if eid in QSCORE_PANEL_EXCLUDE:
                    continue
                ids.append(eid)
    return ids


def _load_existing_rows(path: Path) -> dict[str, dict[str, float | int | str]]:
    if not path.is_file():
        return {}
    out: dict[str, dict[str, float | int | str]] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            eid = str(row.get("emdb_id", "")).strip()
            if eid:
                out[eid] = {k: row.get(k, "") for k in FIELDNAMES}
    return out


def _append_row(path: Path, row: dict[str, float | int | str], *, write_header: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDNAMES})


def run_one(
    emd_id: str,
    *,
    manifest: Path,
    window: int,
    chunk_z: int,
    sphere_a: float,
) -> dict[str, float | int | str]:
    row = load_cohort_manifest_row(manifest, emd_id)
    q_path = resolve_halfmap_reliability_dir(emd_id) / "qscore_validation.csv"
    pdb_path = Path(row["flexibility_path_or_pdb"])
    ref_path = Path(row["reference_mrc"])

    if not q_path.is_file():
        raise FileNotFoundError(f"EMD-{emd_id}: missing {q_path}")

    q_df = _load_q_table(q_path)
    residues = iter_ca_residues(pdb_path)
    if len(residues) != len(q_df):
        raise ValueError(f"EMD-{emd_id}: residue count mismatch PDB={len(residues)} csv={len(q_df)}")

    q = q_df["q_score"].to_numpy(dtype=np.float64)
    use_mask = q_df["in_contour_mask"].to_numpy(dtype=bool)

    grid = load_map_grid(ref_path, dtype=np.float32)
    bundle = load_full_and_half_maps(
        ref_path,
        Path(row["half1_path"]),
        Path(row["half2_path"]),
        dtype=np.float32,
        resample_if_needed=True,
    )
    rho_z = zscore_halfmap_average(bundle.half1.data, bundle.half2.data)
    del bundle, grid.data
    gc.collect()

    grad = gradient_magnitude(rho_z)
    v_map = _constraint_v(grad, window=window)
    v_ca = sample_volume_at_ca(v_map, grid, residues, window_radius=0)
    del grad, v_map
    gc.collect()

    print(f"  [hessian_followup] EMD-{emd_id}: Hessian (chunk_z={chunk_z})...", flush=True)
    hess = density_hessian_scalar_maps(rho_z, chunk_z=chunk_z)
    del rho_z
    gc.collect()

    lam_min = np.asarray(hess.pop("hessian_eig_min"), dtype=np.float32)
    lam_max = np.asarray(hess.pop("hessian_eig_max"), dtype=np.float32)
    hess.clear()
    del hess
    gc.collect()

    eps = 1e-8
    ridge_vol = (lam_min / (np.abs(lam_max) + eps)).astype(np.float32)

    neg_lam_voxel = sample_volume_at_ca(-lam_min, grid, residues, window_radius=0)
    neg_lam_sphere = sample_volume_at_ca(-lam_min, grid, residues, sphere_radius_a=sphere_a)
    ridge_sphere = sample_volume_at_ca(ridge_vol, grid, residues, sphere_radius_a=sphere_a)
    del lam_min, lam_max, ridge_vol
    gc.collect()

    v_fail_tertile = _v_failure_mask(q, v_ca, use_mask, v_quantile=200 / 3, q_quantile=100 / 3)
    v_fail_half = _v_failure_mask(q, v_ca, use_mask, v_quantile=50.0, q_quantile=50.0)

    out: dict[str, float | int | str] = {"emdb_id": emd_id}
    _, n_used = _spearman(q, v_ca, use_mask)
    out["n_used"] = n_used
    out["n_v_failure_tertile"] = int(v_fail_tertile.sum())
    out["n_v_failure_half"] = int(v_fail_half.sum())

    rho_vox, _ = _spearman(neg_lam_voxel, q, use_mask)
    rho_sph, _ = _spearman(neg_lam_sphere, q, use_mask)
    rho_ridge, _ = _spearman(ridge_sphere, q, use_mask)
    out["rho_q_neg_lam_min_voxel"] = rho_vox
    out["rho_q_neg_lam_min_sphere"] = rho_sph
    out["rho_q_ridge_index_sphere"] = rho_ridge

    out["partial_rho_q_neg_lam_min_given_v"] = _partial_spearman(
        neg_lam_sphere, q, v_ca, use_mask
    )
    out["partial_rho_q_ridge_index_given_v"] = _partial_spearman(
        ridge_sphere, q, v_ca, use_mask
    )

    rho_tert, _ = _spearman(neg_lam_sphere, q, v_fail_tertile)
    rho_half, _ = _spearman(neg_lam_sphere, q, v_fail_half)
    rho_v_half, _ = _spearman(v_ca, q, v_fail_half)
    out["rho_q_neg_lam_min_v_failure_tertile"] = rho_tert
    out["rho_q_neg_lam_min_v_failure_half"] = rho_half
    out["rho_q_v_v_failure_half"] = rho_v_half

    out["partial_rho_q_neg_lam_min_given_v_v_failure_half"] = _partial_spearman(
        neg_lam_sphere, q, v_ca, v_fail_half
    )

    if np.isfinite(rho_half) and np.isfinite(rho_v_half):
        out["delta_neg_lam_sphere_minus_v_v_failure_half"] = float(rho_half) - float(rho_v_half)
    else:
        out["delta_neg_lam_sphere_minus_v_v_failure_half"] = float("nan")

    return out


def _as_float(val: object) -> float:
    try:
        return float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


def _med(rows: list[dict], key: str) -> tuple[float, int]:
    vals = [_as_float(r.get(key)) for r in rows]
    vals = [x for x in vals if np.isfinite(x)]
    return (float(np.median(vals)) if vals else float("nan"), len(vals))


def _summarize(rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        print("[hessian_followup] no rows", file=sys.stderr)
        return

    print("\n=== Cohort medians ===")
    for key, label in (
        ("rho_q_neg_lam_min_voxel", "ρ(Q, −λ_min) voxel"),
        ("rho_q_neg_lam_min_sphere", "ρ(Q, −λ_min) sphere 2Å"),
        ("rho_q_ridge_index_sphere", "ρ(Q, λ_min/λ_max) sphere"),
        ("partial_rho_q_neg_lam_min_given_v", "partial ρ(Q, −λ_min | V)"),
        ("partial_rho_q_ridge_index_given_v", "partial ρ(Q, ridge | V)"),
        ("rho_q_neg_lam_min_v_failure_tertile", "ρ(Q, −λ_min) V-fail tertile"),
        ("rho_q_neg_lam_min_v_failure_half", "ρ(Q, −λ_min) V-fail half"),
        ("rho_q_v_v_failure_half", "ρ(Q, V) V-fail half"),
        ("partial_rho_q_neg_lam_min_given_v_v_failure_half", "partial ρ(Q, −λ_min | V) V-fail half"),
        ("delta_neg_lam_sphere_minus_v_v_failure_half", "Δρ(−λ_min − V) V-fail half"),
    ):
        med, n = _med(rows, key)
        print(f"  {label:42s}  median = {med:+.4f}  (n={n})")

    n_tert = [_as_float(r.get("n_v_failure_tertile")) for r in rows]
    n_half = [_as_float(r.get("n_v_failure_half")) for r in rows]
    print(
        f"\n  median n V-failure tertile = {float(np.median([x for x in n_tert if np.isfinite(x)])):.0f}, "
        f"half = {float(np.median([x for x in n_half if np.isfinite(x)])):.0f}"
    )

    deltas = [_as_float(r.get("delta_neg_lam_sphere_minus_v_v_failure_half")) for r in rows]
    deltas = [x for x in deltas if np.isfinite(x)]
    if deltas:
        print(
            f"  −λ_min sphere wins vs V in half-stratum: "
            f"{sum(d > 0 for d in deltas)}/{len(deltas)} maps"
        )

    sph = [_as_float(r.get("rho_q_neg_lam_min_sphere")) for r in rows]
    vox = [_as_float(r.get("rho_q_neg_lam_min_voxel")) for r in rows]
    pairs = [(a, b) for a, b in zip(sph, vox) if np.isfinite(a) and np.isfinite(b)]
    if pairs:
        d = [a - b for a, b in pairs]
        print(f"  sphere − voxel ρ(Q, −λ_min): median Δ = {float(np.median(d)):+.4f}")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    ids = [args.emd_id] if args.emd_id else _emd_ids_from_qscore_cohort()
    if not ids:
        print("[hessian_followup] no Q-score cohort maps found", file=sys.stderr)
        return 1

    existing = {} if args.force or args.emd_id else _load_existing_rows(OUT_CSV)
    if args.resume and existing and not args.force:
        print(f"[hessian_followup] resume: {len(existing)} maps in {OUT_CSV}", flush=True)
    elif not args.emd_id and OUT_CSV.is_file() and not args.resume:
        OUT_CSV.unlink()

    rows_by_id: dict[str, dict[str, float | int | str]] = dict(existing)
    write_header = not OUT_CSV.is_file() or OUT_CSV.stat().st_size == 0

    for i, emd_id in enumerate(ids, 1):
        if args.resume and not args.force and emd_id in rows_by_id:
            print(f"[hessian_followup] ({i}/{len(ids)}) EMD-{emd_id} skip", flush=True)
            continue
        print(f"[hessian_followup] ({i}/{len(ids)}) EMD-{emd_id}", flush=True)
        try:
            row = run_one(
                emd_id,
                manifest=args.manifest,
                window=args.window,
                chunk_z=args.chunk_z,
                sphere_a=args.sphere_a,
            )
        except Exception as exc:
            print(f"[hessian_followup] EMD-{emd_id} FAILED: {exc}", file=sys.stderr, flush=True)
            continue
        rows_by_id[emd_id] = row
        _append_row(OUT_CSV, row, write_header=write_header)
        write_header = False

    rows = [rows_by_id[eid] for eid in ids if eid in rows_by_id]
    print(f"\n[hessian_followup] {len(rows)}/{len(ids)} maps → {OUT_CSV}", flush=True)
    _summarize(rows)
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
