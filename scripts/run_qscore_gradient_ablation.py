"""Ablation: central-difference vs Sobel gradient for constraint V vs Q-score.

Compares ρ(Q, V) at in-mask Cα for four V variants on the existing Q-score cohort:
  - central_unwindowed: ½‖∇ρ‖² (np.gradient)
  - central_windowed:   box-filter(½‖∇ρ‖², 5³) — production default
  - sobel_unwindowed:     ½‖∇ρ‖² (scipy.ndimage.sobel)
  - sobel_windowed:       box-filter(½‖∇ρ‖²_sobel, 5³)

Reuses per-residue Q-scores from existing ``qscore_validation.csv`` files.

Example::

    source .venv/bin/activate
    PYTHONUNBUFFERED=1 python scripts/run_qscore_gradient_ablation.py
    python scripts/run_qscore_gradient_ablation.py --resume   # after interrupt/crash
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
from cryoem_mrc.local_stats import gradient_magnitude
from cryoem_mrc.map_grid import load_full_and_half_maps, load_map_grid
from cryoem_mrc.repo_paths import COHORT_MANIFEST, OUTPUTS_ROOT, resolve_halfmap_reliability_dir
from cryoem_mrc.structure_validation import iter_ca_residues, load_cohort_manifest_row, sample_volume_at_ca

QSCORE_PANEL_EXCLUDE = frozenset({"33736"})
DEFAULT_WINDOW = 5

VARIANTS = (
    "central_unwindowed",
    "central_windowed",
    "sobel_unwindowed",
    "sobel_windowed",
    "production_npz",
)

FIELDNAMES = ["emdb_id"] + [f"rho_{v}" for v in VARIANTS] + ["n_used"]
OUT_CSV = OUTPUTS_ROOT / "cohort_summary" / "qscore_gradient_ablation.csv"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, default=COHORT_MANIFEST)
    p.add_argument("--window", type=int, default=DEFAULT_WINDOW, help="Box window for *_windowed variants")
    p.add_argument("--emd-id", type=str, default=None, help="Single map (smoke test)")
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip maps already present in the output CSV (append new rows incrementally)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Recompute all maps even when --resume would skip them",
    )
    return p.parse_args(argv)


def _sobel_gradient_magnitude(volume: np.ndarray) -> np.ndarray:
    """Sobel-filtered gradient magnitude on (Z, Y, X) volume."""
    v = np.asarray(volume, dtype=np.float64)
    gz = ndimage.sobel(v, axis=0, mode="nearest")
    gy = ndimage.sobel(v, axis=1, mode="nearest")
    gx = ndimage.sobel(v, axis=2, mode="nearest")
    mag = np.sqrt(gz * gz + gy * gy + gx * gx)
    return mag.astype(np.float32, copy=False)


def _constraint_v(rho_z: np.ndarray, *, grad_mag: np.ndarray, window: int) -> np.ndarray:
    v_raw = (0.5 * grad_mag * grad_mag).astype(np.float64)
    if int(window) > 1:
        v_raw = ndimage.uniform_filter(v_raw, size=int(window), mode="nearest")
    return v_raw.astype(np.float32, copy=False)


def _compute_v_variants(rho_z: np.ndarray, *, window: int) -> dict[str, np.ndarray]:
    central = gradient_magnitude(rho_z)
    sobel = _sobel_gradient_magnitude(rho_z)
    return {
        "central_unwindowed": _constraint_v(rho_z, grad_mag=central, window=1),
        "central_windowed": _constraint_v(rho_z, grad_mag=central, window=window),
        "sobel_unwindowed": _constraint_v(rho_z, grad_mag=sobel, window=1),
        "sobel_windowed": _constraint_v(rho_z, grad_mag=sobel, window=window),
    }


def _spearman_q_v(q: np.ndarray, v: np.ndarray, mask: np.ndarray) -> tuple[float, int]:
    m = mask & np.isfinite(q) & np.isfinite(v)
    n = int(m.sum())
    if n < 10:
        return float("nan"), n
    rho, _ = stats.spearmanr(q[m], v[m])
    return float(rho), n


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


def run_one(emd_id: str, *, manifest: Path, window: int) -> dict[str, float | int | str]:
    row = load_cohort_manifest_row(manifest, emd_id)
    q_path = resolve_halfmap_reliability_dir(emd_id) / "qscore_validation.csv"
    npz_path = resolve_halfmap_reliability_dir(emd_id) / "reliability.npz"
    pdb_path = Path(row["flexibility_path_or_pdb"])
    ref_path = Path(row["reference_mrc"])

    if not q_path.is_file():
        raise FileNotFoundError(f"EMD-{emd_id}: missing {q_path}")

    q_df = _load_q_table(q_path)
    residues = iter_ca_residues(pdb_path)
    if len(residues) != len(q_df):
        raise ValueError(f"EMD-{emd_id}: residue count mismatch PDB={len(residues)} csv={len(q_df)}")

    q = q_df["q_score"].to_numpy(dtype=np.float64)
    in_mask = q_df["in_contour_mask"].to_numpy(dtype=bool)

    grid = load_map_grid(ref_path, dtype=np.float32)
    bundle = load_full_and_half_maps(
        ref_path,
        Path(row["half1_path"]),
        Path(row["half2_path"]),
        dtype=np.float32,
        resample_if_needed=True,
    )
    rho_z = zscore_halfmap_average(bundle.half1.data, bundle.half2.data)
    del bundle
    gc.collect()

    v_maps = _compute_v_variants(rho_z, window=window)
    del rho_z
    gc.collect()

    if npz_path.is_file():
        with np.load(npz_path, allow_pickle=False) as d:
            key = "reliability_smoothness" if "reliability_smoothness" in d else "reliability_constraint_V"
            v_maps["production_npz"] = np.asarray(d[key], dtype=np.float32)

    out: dict[str, float | int | str] = {"emdb_id": emd_id}
    for name, vol in v_maps.items():
        v_ca = sample_volume_at_ca(vol, grid, residues, window_radius=0)
        rho, n = _spearman_q_v(q, v_ca, in_mask)
        out[f"rho_{name}"] = rho
        out["n_used"] = n
        del v_ca, vol
    gc.collect()
    return out


def _as_float(val: object) -> float:
    try:
        return float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


def _summarize(rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        print("[gradient_ablation] no rows", file=sys.stderr)
        return

    print("\n=== Per-map Spearman ρ(Q, V) ===")
    print(",".join(FIELDNAMES))
    for r in rows:
        print(",".join(str(r.get(k, "")) for k in FIELDNAMES))

    print("\n=== Cohort summary (median ρ across maps) ===")
    for v in VARIANTS:
        key = f"rho_{v}"
        vals = [_as_float(r.get(key)) for r in rows]
        vals = [x for x in vals if np.isfinite(x)]
        med = float(np.median(vals)) if vals else float("nan")
        print(f"  {v:22s}  median ρ = {med:+.4f}  (n={len(vals)})")

    base_key = "rho_central_windowed"
    sob_key = "rho_sobel_windowed"
    deltas = [
        _as_float(r[sob_key]) - _as_float(r[base_key])
        for r in rows
        if np.isfinite(_as_float(r.get(sob_key))) and np.isfinite(_as_float(r.get(base_key)))
    ]
    if deltas:
        print(
            f"\n  sobel_windowed − central_windowed: "
            f"median Δρ = {float(np.median(deltas)):+.4f}, "
            f"mean Δρ = {float(np.mean(deltas)):+.4f}, "
            f"maps improved = {sum(d > 0 for d in deltas)}/{len(deltas)}"
        )

    prod = [_as_float(r["rho_production_npz"]) for r in rows if np.isfinite(_as_float(r.get("rho_production_npz")))]
    rec = [_as_float(r["rho_central_windowed"]) for r in rows if np.isfinite(_as_float(r.get("rho_central_windowed")))]
    if prod and rec:
        repro = [p - r for p, r in zip(prod, rec)]
        print(f"  production_npz vs recomputed central_windowed: max |Δρ| = {max(abs(x) for x in repro):.6f}")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    ids = [args.emd_id] if args.emd_id else _emd_ids_from_qscore_cohort(args.manifest)
    if not ids:
        print("[gradient_ablation] no Q-score cohort maps found", file=sys.stderr)
        return 1

    existing = {} if args.force or args.emd_id else _load_existing_rows(OUT_CSV)
    if args.resume and existing and not args.force:
        print(f"[gradient_ablation] resume: {len(existing)} maps already in {OUT_CSV}", flush=True)
    elif not args.emd_id and OUT_CSV.is_file() and not args.resume:
        OUT_CSV.unlink()

    rows_by_id: dict[str, dict[str, float | int | str]] = dict(existing)
    write_header = not OUT_CSV.is_file() or OUT_CSV.stat().st_size == 0

    for i, emd_id in enumerate(ids, 1):
        if args.resume and not args.force and emd_id in rows_by_id:
            print(f"[gradient_ablation] ({i}/{len(ids)}) EMD-{emd_id} skip (done)", flush=True)
            continue
        print(f"[gradient_ablation] ({i}/{len(ids)}) EMD-{emd_id}", flush=True)
        try:
            row = run_one(emd_id, manifest=args.manifest, window=args.window)
        except Exception as exc:
            print(f"[gradient_ablation] EMD-{emd_id} FAILED: {exc}", file=sys.stderr, flush=True)
            continue
        rows_by_id[emd_id] = row
        _append_row(OUT_CSV, row, write_header=write_header)
        write_header = False

    rows = [rows_by_id[eid] for eid in ids if eid in rows_by_id]
    print(f"\n[gradient_ablation] {len(rows)}/{len(ids)} maps in {OUT_CSV}", flush=True)
    _summarize(rows)
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
