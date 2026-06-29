"""Hessian λ_min vs gradient / constraint V for Q-score placement validation.

Tests whether minimum Hessian eigenvalue (positional degeneracy) predicts per-residue
Q-score better than gradient-based V, especially where V is falsely confident
(high V, low Q).

Reuses ``qscore_validation.csv`` from the existing Q-score cohort. Computes maps
from the z-scored half-map average (same ρ as production V).

Example::

    source .venv/bin/activate
    PYTHONUNBUFFERED=1 python scripts/run_qscore_hessian_ablation.py --emd-id 49450
    python scripts/run_qscore_hessian_ablation.py --resume
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

PREDICTORS = (
    "v_windowed",
    "grad_mag",
    "lam_min",
    "lam_min_windowed",
    "neg_lam_min",
    "lam_max",
    "anisotropy",
    "trace",
    "frobenius",
)

FIELDNAMES = [
    "emdb_id",
    "n_used",
    "n_v_failure",
    *[f"rho_q_{p}" for p in PREDICTORS],
    *[f"rho_q_{p}_v_failure" for p in ("lam_min", "v_windowed", "grad_mag")],
    "delta_lam_min_minus_v_v_failure",
]

OUT_CSV = OUTPUTS_ROOT / "cohort_summary" / "qscore_hessian_ablation.csv"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, default=COHORT_MANIFEST)
    p.add_argument("--window", type=int, default=DEFAULT_WINDOW, help="Box window for V and λ_min smoothing")
    p.add_argument("--chunk-z", type=int, default=DEFAULT_CHUNK_Z, help="Z-chunk size for Hessian")
    p.add_argument("--emd-id", type=str, default=None, help="Single map (smoke test)")
    p.add_argument("--resume", action="store_true", help="Skip maps already in the output CSV")
    p.add_argument("--force", action="store_true", help="Recompute all maps")
    return p.parse_args(argv)


def _constraint_v(grad_mag: np.ndarray, *, window: int) -> np.ndarray:
    v_raw = (0.5 * grad_mag * grad_mag).astype(np.float64)
    if int(window) > 1:
        v_raw = ndimage.uniform_filter(v_raw, size=int(window), mode="nearest")
    return v_raw.astype(np.float32, copy=False)


def _maybe_window(vol: np.ndarray, window: int) -> np.ndarray:
    if int(window) <= 1:
        return vol.astype(np.float32, copy=False)
    return ndimage.uniform_filter(vol.astype(np.float64), size=int(window), mode="nearest").astype(
        np.float32, copy=False
    )


def _spearman(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> tuple[float, int]:
    m = mask & np.isfinite(x) & np.isfinite(y)
    n = int(m.sum())
    if n < 10:
        return float("nan"), n
    rho, _ = stats.spearmanr(x[m], y[m])
    return float(rho), n


def _v_failure_mask(q: np.ndarray, v: np.ndarray, in_mask: np.ndarray) -> np.ndarray:
    """High V (top tertile) and low Q (bottom tertile) among in-mask residues."""
    m = in_mask & np.isfinite(q) & np.isfinite(v)
    out = np.zeros_like(in_mask, dtype=bool)
    if int(m.sum()) < 30:
        return out
    q_sub = q[m]
    v_sub = v[m]
    q_hi = float(np.percentile(q_sub, 100 / 3))
    v_lo = float(np.percentile(v_sub, 200 / 3))
    out[m] = (q_sub <= q_hi) & (v_sub >= v_lo)
    return out


def _load_q_table(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["q_score"] = pd.to_numeric(df["q_score"], errors="coerce")
    df["in_contour_mask"] = df["in_contour_mask"].astype(int).astype(bool)
    return df


def _emd_ids_from_qscore_cohort(manifest: Path) -> list[str]:
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

    for bundle in sorted(OUTPUTS_ROOT.glob("emd_*/halfmap_reliability/qscore_validation.csv")):
        eid = bundle.parent.parent.name.removeprefix("emd_")
        if eid in QSCORE_PANEL_EXCLUDE:
            continue
        ids.append(eid)
    for bundle in sorted(OUTPUTS_ROOT.glob("emd_*/lh_map_reliability/qscore_validation.csv")):
        eid = bundle.parent.parent.name.removeprefix("emd_")
        if eid in QSCORE_PANEL_EXCLUDE or eid in ids:
            continue
        ids.append(eid)
    return ids


def _load_existing_rows(path: Path) -> dict[str, dict[str, float | int | str]]:
    if not path.is_file():
        return {}
    out: dict[str, dict[str, float | int | str]] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
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
    ca_maps: dict[str, np.ndarray] = {
        "v_windowed": sample_volume_at_ca(v_map, grid, residues, window_radius=0),
        "grad_mag": sample_volume_at_ca(grad, grid, residues, window_radius=0),
    }
    del grad, v_map
    gc.collect()

    print(f"  [hessian_ablation] EMD-{emd_id}: Hessian (chunk_z={chunk_z})...", flush=True)
    hess = density_hessian_scalar_maps(rho_z, chunk_z=chunk_z)
    del rho_z
    gc.collect()

    lam_min = np.asarray(hess.pop("hessian_eig_min"), dtype=np.float32)
    ca_maps["lam_min"] = sample_volume_at_ca(lam_min, grid, residues, window_radius=0)
    lam_min_w = _maybe_window(lam_min, window)
    ca_maps["lam_min_windowed"] = sample_volume_at_ca(lam_min_w, grid, residues, window_radius=0)
    del lam_min, lam_min_w
    gc.collect()

    hess_keys = (
        ("hessian_eig_max", "lam_max"),
        ("hessian_anisotropy", "anisotropy"),
        ("hessian_trace", "trace"),
        ("hessian_frobenius", "frobenius"),
    )
    for hkey, pname in hess_keys:
        vol = hess.pop(hkey)
        ca_maps[pname] = sample_volume_at_ca(vol, grid, residues, window_radius=0)
        del vol
    hess.clear()
    del hess
    gc.collect()

    ca_maps["neg_lam_min"] = -ca_maps["lam_min"]

    v_fail = _v_failure_mask(q, ca_maps["v_windowed"], use_mask)

    out: dict[str, float | int | str] = {"emdb_id": emd_id}
    _, n_used = _spearman(q, ca_maps["v_windowed"], use_mask)
    out["n_used"] = n_used
    out["n_v_failure"] = int(v_fail.sum())

    for name, scores in ca_maps.items():
        rho, _ = _spearman(scores, q, use_mask)
        out[f"rho_q_{name}"] = rho

    for name in ("lam_min", "v_windowed", "grad_mag"):
        rho, _ = _spearman(ca_maps[name], q, v_fail)
        out[f"rho_q_{name}_v_failure"] = rho

    lam_r = out.get("rho_q_lam_min_v_failure", float("nan"))
    v_r = out.get("rho_q_v_windowed_v_failure", float("nan"))
    if isinstance(lam_r, (int, float)) and isinstance(v_r, (int, float)) and np.isfinite(lam_r) and np.isfinite(v_r):
        out["delta_lam_min_minus_v_v_failure"] = float(lam_r) - float(v_r)
    else:
        out["delta_lam_min_minus_v_v_failure"] = float("nan")

    return out


def _as_float(val: object) -> float:
    try:
        return float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


def _summarize(rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        print("[hessian_ablation] no rows", file=sys.stderr)
        return

    print("\n=== Per-map Spearman ρ(Q, predictor) ===")
    print(",".join(FIELDNAMES))
    for r in rows:
        print(",".join(str(r.get(k, "")) for k in FIELDNAMES))

    print("\n=== Cohort median ρ(Q, ·) — all in-mask Cα ===")
    for p in PREDICTORS:
        key = f"rho_q_{p}"
        vals = [_as_float(r.get(key)) for r in rows]
        vals = [x for x in vals if np.isfinite(x)]
        med = float(np.median(vals)) if vals else float("nan")
        print(f"  {p:20s}  median ρ = {med:+.4f}  (n={len(vals)} maps)")

    print("\n=== V-failure stratum (high V, bottom-tercile Q) ===")
    n_fail = [_as_float(r.get("n_v_failure")) for r in rows]
    n_fail_f = [x for x in n_fail if np.isfinite(x)]
    if n_fail_f:
        print(f"  median n_v_failure = {float(np.median(n_fail_f)):.0f} residues/map")

    for p in ("lam_min", "v_windowed", "grad_mag"):
        key = f"rho_q_{p}_v_failure"
        vals = [_as_float(r.get(key)) for r in rows]
        vals = [x for x in vals if np.isfinite(x)]
        med = float(np.median(vals)) if vals else float("nan")
        print(f"  ρ(Q, {p:14s})  median = {med:+.4f}  (n={len(vals)} maps)")

    deltas = [_as_float(r.get("delta_lam_min_minus_v_v_failure")) for r in rows]
    deltas = [x for x in deltas if np.isfinite(x)]
    if deltas:
        print(
            f"\n  Δρ = ρ(Q,λ_min) − ρ(Q,V) in V-failure stratum: "
            f"median = {float(np.median(deltas)):+.4f}, "
            f"λ_min wins = {sum(d > 0 for d in deltas)}/{len(deltas)} maps"
        )

    g_lam = [_as_float(r.get("rho_q_lam_min")) for r in rows]
    g_v = [_as_float(r.get("rho_q_v_windowed")) for r in rows]
    pairs = [(a, b) for a, b in zip(g_lam, g_v) if np.isfinite(a) and np.isfinite(b)]
    if pairs:
        d_global = [a - b for a, b in pairs]
        print(
            f"  Global Δρ(Q,λ_min) − ρ(Q,V): median = {float(np.median(d_global)):+.4f}, "
            f"λ_min wins = {sum(d > 0 for d in d_global)}/{len(d_global)} maps"
        )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    ids = [args.emd_id] if args.emd_id else _emd_ids_from_qscore_cohort(args.manifest)
    if not ids:
        print("[hessian_ablation] no Q-score cohort maps found", file=sys.stderr)
        return 1

    existing = {} if args.force or args.emd_id else _load_existing_rows(OUT_CSV)
    if args.resume and existing and not args.force:
        print(f"[hessian_ablation] resume: {len(existing)} maps already in {OUT_CSV}", flush=True)
    elif not args.emd_id and OUT_CSV.is_file() and not args.resume:
        OUT_CSV.unlink()

    rows_by_id: dict[str, dict[str, float | int | str]] = dict(existing)
    write_header = not OUT_CSV.is_file() or OUT_CSV.stat().st_size == 0

    for i, emd_id in enumerate(ids, 1):
        if args.resume and not args.force and emd_id in rows_by_id:
            print(f"[hessian_ablation] ({i}/{len(ids)}) EMD-{emd_id} skip (done)", flush=True)
            continue
        print(f"[hessian_ablation] ({i}/{len(ids)}) EMD-{emd_id}", flush=True)
        try:
            row = run_one(emd_id, manifest=args.manifest, window=args.window, chunk_z=args.chunk_z)
        except Exception as exc:
            print(f"[hessian_ablation] EMD-{emd_id} FAILED: {exc}", file=sys.stderr, flush=True)
            continue
        rows_by_id[emd_id] = row
        _append_row(OUT_CSV, row, write_header=write_header)
        write_header = False

    rows = [rows_by_id[eid] for eid in ids if eid in rows_by_id]
    print(f"\n[hessian_ablation] {len(rows)}/{len(ids)} maps → {OUT_CSV}", flush=True)
    _summarize(rows)
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
