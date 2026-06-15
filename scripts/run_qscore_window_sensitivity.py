"""ρ(Q, V) vs box-window width for constraint V (central-diff gradient).

Sweeps odd window sizes on the Q-score cohort; reuses qscore_validation.csv Q values.

Example::

    PYTHONUNBUFFERED=1 python scripts/run_qscore_window_sensitivity.py
    python scripts/run_qscore_window_sensitivity.py --resume
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
DEFAULT_WINDOWS = (1, 3, 5, 7, 9, 11, 15)
OUT_CSV = OUTPUTS_ROOT / "cohort_summary" / "qscore_window_sensitivity.csv"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, default=COHORT_MANIFEST)
    p.add_argument("--windows", type=int, nargs="+", default=list(DEFAULT_WINDOWS))
    p.add_argument("--emd-id", type=str, default=None)
    p.add_argument("--resume", action="store_true")
    return p.parse_args(argv)


def _emd_ids(manifest: Path) -> list[str]:
    path = OUTPUTS_ROOT / "cohort_summary" / "qscore_correlations.csv"
    ids: list[str] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            eid = str(row["emdb_id"]).strip()
            if eid not in QSCORE_PANEL_EXCLUDE:
                ids.append(eid)
    return ids


def _constraint_v_windowed(rho_z: np.ndarray, *, window: int) -> np.ndarray:
    grad_sq = gradient_magnitude(rho_z) ** 2
    v_raw = (0.5 * grad_sq).astype(np.float64)
    w = int(window)
    if w > 1:
        v_raw = ndimage.uniform_filter(v_raw, size=w, mode="nearest")
    return v_raw.astype(np.float32, copy=False)


def _spearman(q: np.ndarray, v: np.ndarray, mask: np.ndarray) -> float:
    m = mask & np.isfinite(q) & np.isfinite(v)
    if int(m.sum()) < 10:
        return float("nan")
    return float(stats.spearmanr(q[m], v[m]).statistic)


def run_one(emd_id: str, *, manifest: Path, windows: list[int]) -> dict[str, float | int | str]:
    row = load_cohort_manifest_row(manifest, emd_id)
    q_path = resolve_halfmap_reliability_dir(emd_id) / "qscore_validation.csv"
    pdb_path = Path(row["flexibility_path_or_pdb"])
    ref_path = Path(row["reference_mrc"])
    q_df = pd.read_csv(q_path)
    q_df["q_score"] = pd.to_numeric(q_df["q_score"], errors="coerce")
    q_df["in_contour_mask"] = q_df["in_contour_mask"].astype(int).astype(bool)

    residues = iter_ca_residues(pdb_path)
    q = q_df["q_score"].to_numpy(dtype=np.float64)
    in_mask = q_df["in_contour_mask"].to_numpy(dtype=bool)

    grid = load_map_grid(ref_path, dtype=np.float32)
    bundle = load_full_and_half_maps(
        ref_path, Path(row["half1_path"]), Path(row["half2_path"]),
        dtype=np.float32, resample_if_needed=True,
    )
    rho_z = zscore_halfmap_average(bundle.half1.data, bundle.half2.data)
    del bundle
    gc.collect()

    out: dict[str, float | int | str] = {"emdb_id": emd_id, "n_used": int((in_mask & np.isfinite(q)).sum())}
    for w in windows:
        vol = _constraint_v_windowed(rho_z, window=w)
        v_ca = sample_volume_at_ca(vol, grid, residues, window_radius=0)
        out[f"rho_w{w}"] = _spearman(q, v_ca, in_mask)
        del vol, v_ca
    del rho_z
    gc.collect()
    return out


def _load_done(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    with path.open(newline="") as f:
        return {str(r["emdb_id"]).strip() for r in csv.DictReader(f)}


def _summarize(rows: list[dict], windows: list[int]) -> None:
    print("\n=== Cohort median ρ(Q, V) by window size ===")
    for w in windows:
        key = f"rho_w{w}"
        vals = [float(r[key]) for r in rows if key in r and np.isfinite(float(r[key]))]
        med = float(np.median(vals)) if vals else float("nan")
        print(f"  window {w:2d}³ voxels  median ρ = {med:+.4f}  (n={len(vals)})")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    windows = sorted({int(w) for w in args.windows})
    ids = [args.emd_id] if args.emd_id else _emd_ids(args.manifest)
    fieldnames = ["emdb_id"] + [f"rho_w{w}" for w in windows] + ["n_used"]

    done = _load_done(OUT_CSV) if args.resume else set()
    if done and args.resume:
        print(f"[window_sens] resume: {len(done)} maps done", flush=True)
    elif OUT_CSV.is_file() and not args.resume and not args.emd_id:
        OUT_CSV.unlink()

    write_header = not OUT_CSV.is_file() or OUT_CSV.stat().st_size == 0
    rows: list[dict] = []
    if done:
        with OUT_CSV.open(newline="") as f:
            rows.extend(csv.DictReader(f))

    for i, emd_id in enumerate(ids, 1):
        if emd_id in done:
            print(f"[window_sens] ({i}/{len(ids)}) EMD-{emd_id} skip", flush=True)
            continue
        print(f"[window_sens] ({i}/{len(ids)}) EMD-{emd_id}", flush=True)
        try:
            row = run_one(emd_id, manifest=args.manifest, windows=windows)
        except Exception as exc:
            print(f"[window_sens] EMD-{emd_id} FAILED: {exc}", file=sys.stderr, flush=True)
            continue
        OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        with OUT_CSV.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                w.writeheader()
                write_header = False
            w.writerow({k: row.get(k, "") for k in fieldnames})
        rows.append(row)

    print(f"\n[window_sens] {len(rows)}/{len(ids)} maps → {OUT_CSV}", flush=True)
    _summarize(rows, windows)
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
