"""Ablation: half-map agreement weighting of constraint V vs Q-score.

Tests whether downweighting high |∇ρ|² in locally irreproducible regions improves
ρ(Q, V) relative to production windowed V.

Variants (all use central-difference gradient on z-scored half-map average):
  - v_baseline:           box-filter(½‖∇ρ‖², 5³) — production default
  - v_x_cc_pre_window:    box-filter(½‖∇ρ‖² × clip(CC,0,1))
  - v_x_cc_post_window:   box-filter(½‖∇ρ‖²) × clip(CC,0,1)
  - v_x_inv_var_diff:     box-filter(½‖∇ρ‖² × 1/(1 + var_diff/p50))
  - v_x_one_minus_vard:   box-filter(½‖∇ρ‖² × (1 − robust01(var_diff)))

Reuses per-residue Q-scores from existing ``qscore_validation.csv`` files.

Example::

    source .venv/bin/activate
    PYTHONUNBUFFERED=1 python scripts/run_qscore_v_cc_weight_ablation.py --emd-id 49450
    python scripts/run_qscore_v_cc_weight_ablation.py --resume
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
from cryoem_mrc.halfmap_metrics import (
    WINDOWED_HALFMAP_CORRELATION_KEY,
    half_map_local_metrics,
    load_windowed_halfmap_correlation,
)
from cryoem_mrc.local_stats import gradient_magnitude
from cryoem_mrc.map_grid import load_full_and_half_maps, load_map_grid
from cryoem_mrc.repo_paths import COHORT_MANIFEST, OUTPUTS_ROOT, glob_halfmap_reliability_files, halfmap_metrics_npz, resolve_halfmap_reliability_dir
from cryoem_mrc.structure_validation import iter_ca_residues, load_cohort_manifest_row, sample_volume_at_ca

QSCORE_PANEL_EXCLUDE = frozenset({"33736"})
DEFAULT_WINDOW = 5

VARIANTS = (
    "v_baseline",
    "v_x_cc_pre_window",
    "v_x_cc_post_window",
    "v_x_inv_var_diff",
    "v_x_one_minus_vard",
)

FIELDNAMES = ["emdb_id"] + [f"rho_{v}" for v in VARIANTS] + ["n_used"]
OUT_CSV = OUTPUTS_ROOT / "cohort_summary" / "qscore_v_cc_weight_ablation.csv"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, default=COHORT_MANIFEST)
    p.add_argument("--window", type=int, default=DEFAULT_WINDOW, help="Box window for V smoothing")
    p.add_argument("--cc-window", type=int, default=DEFAULT_WINDOW, help="Half-map local CC window")
    p.add_argument("--emd-id", type=str, default=None, help="Single map (smoke test)")
    p.add_argument("--resume", action="store_true", help="Skip maps already in the output CSV")
    p.add_argument("--force", action="store_true", help="Recompute all maps")
    return p.parse_args(argv)


def _window(vol: np.ndarray, window: int) -> np.ndarray:
    if int(window) <= 1:
        return np.asarray(vol, dtype=np.float64)
    return ndimage.uniform_filter(np.asarray(vol, dtype=np.float64), size=int(window), mode="nearest")


def _robust01(vol: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    finite = vol[np.isfinite(vol)]
    if finite.size == 0:
        return np.zeros_like(vol, dtype=np.float64)
    lo, hi = np.percentile(finite, [5.0, 95.0])
    return np.clip((vol - lo) / (hi - lo + eps), 0.0, 1.0)


def _load_halfmap_agreement(
    half1: np.ndarray,
    half2: np.ndarray,
    emd_id: str,
    *,
    cc_window: int,
) -> dict[str, np.ndarray]:
    hm_path = halfmap_metrics_npz(emd_id)
    if hm_path.is_file():
        with np.load(hm_path, allow_pickle=False) as d:
            cc = load_windowed_halfmap_correlation(d)
            out = {WINDOWED_HALFMAP_CORRELATION_KEY: cc.astype(np.float32, copy=False)}
            for key in ("local_variance_difference", "local_reproducibility_snr"):
                if key in d.files:
                    out[key] = np.asarray(d[key], dtype=np.float32)
            if "local_variance_difference" not in out:
                metrics = half_map_local_metrics(half1, half2, window=cc_window)
                out["local_variance_difference"] = metrics["local_variance_difference"]
            return out

    metrics = half_map_local_metrics(half1, half2, window=cc_window)
    return {k: v.astype(np.float32, copy=False) for k, v in metrics.items()}


def _compute_v_variant(
    *,
    v_raw: np.ndarray,
    cc: np.ndarray,
    inv_var_w: np.ndarray,
    one_minus_vard_w: np.ndarray,
    window: int,
    name: str,
) -> np.ndarray:
    if name == "v_baseline":
        out = _window(v_raw, window)
    elif name == "v_x_cc_pre_window":
        out = _window(v_raw * cc, window)
    elif name == "v_x_cc_post_window":
        out = _window(v_raw, window) * cc
    elif name == "v_x_inv_var_diff":
        out = _window(v_raw * inv_var_w, window)
    elif name == "v_x_one_minus_vard":
        out = _window(v_raw * one_minus_vard_w, window)
    else:
        raise ValueError(f"unknown variant {name!r}")
    return out.astype(np.float32, copy=False)


def _prepare_v_weights(
    rho_z: np.ndarray,
    agreement: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    grad_sq = gradient_magnitude(rho_z).astype(np.float64) ** 2
    v_raw = 0.5 * grad_sq
    del grad_sq

    cc = np.clip(np.asarray(agreement[WINDOWED_HALFMAP_CORRELATION_KEY], dtype=np.float64), 0.0, 1.0)
    var_diff = np.asarray(agreement["local_variance_difference"], dtype=np.float64)
    p50 = float(np.median(var_diff[np.isfinite(var_diff)]))
    inv_var_w = 1.0 / (1.0 + var_diff / (p50 + 1e-12))
    one_minus_vard_w = 1.0 - _robust01(var_diff)
    return v_raw, cc, inv_var_w, one_minus_vard_w


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

    for bundle in glob_halfmap_reliability_files(OUTPUTS_ROOT, "qscore_validation.csv"):
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
    cc_window: int,
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
    in_mask = q_df["in_contour_mask"].to_numpy(dtype=bool)

    grid = load_map_grid(ref_path, dtype=np.float32)
    bundle = load_full_and_half_maps(
        ref_path,
        Path(row["half1_path"]),
        Path(row["half2_path"]),
        dtype=np.float32,
        resample_if_needed=True,
    )
    agreement = _load_halfmap_agreement(
        bundle.half1.data,
        bundle.half2.data,
        emd_id,
        cc_window=cc_window,
    )
    rho_z = zscore_halfmap_average(bundle.half1.data, bundle.half2.data)
    del bundle
    gc.collect()

    grid = load_map_grid(ref_path, dtype=np.float32)
    bundle = load_full_and_half_maps(
        ref_path,
        Path(row["half1_path"]),
        Path(row["half2_path"]),
        dtype=np.float32,
        resample_if_needed=True,
    )
    agreement = _load_halfmap_agreement(
        bundle.half1.data,
        bundle.half2.data,
        emd_id,
        cc_window=cc_window,
    )
    rho_z = zscore_halfmap_average(bundle.half1.data, bundle.half2.data)
    del bundle
    gc.collect()

    v_raw, cc, inv_var_w, one_minus_vard_w = _prepare_v_weights(rho_z, agreement)
    del rho_z
    gc.collect()

    out: dict[str, float | int | str] = {"emdb_id": emd_id}
    for name in VARIANTS:
        vol = _compute_v_variant(
            v_raw=v_raw,
            cc=cc,
            inv_var_w=inv_var_w,
            one_minus_vard_w=one_minus_vard_w,
            window=window,
            name=name,
        )
        v_ca = sample_volume_at_ca(vol, grid, residues, window_radius=0)
        rho, n = _spearman_q_v(q, v_ca, in_mask)
        out[f"rho_{name}"] = rho
        out["n_used"] = n
        del vol, v_ca
        gc.collect()
    del v_raw, cc, inv_var_w, one_minus_vard_w, agreement, grid
    gc.collect()
    return out


def _as_float(val: object) -> float:
    try:
        return float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


def _summarize(rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        print("[v_cc_weight_ablation] no rows", file=sys.stderr)
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

    base_key = "rho_v_baseline"
    for v in VARIANTS[1:]:
        key = f"rho_{v}"
        deltas = [
            _as_float(r[key]) - _as_float(r[base_key])
            for r in rows
            if np.isfinite(_as_float(r.get(key))) and np.isfinite(_as_float(r.get(base_key)))
        ]
        if not deltas:
            continue
        print(
            f"\n  {v} − v_baseline: "
            f"median Δρ = {float(np.median(deltas)):+.4f}, "
            f"mean Δρ = {float(np.mean(deltas)):+.4f}, "
            f"maps improved = {sum(d > 0 for d in deltas)}/{len(deltas)}"
        )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    ids = [args.emd_id] if args.emd_id else _emd_ids_from_qscore_cohort(args.manifest)
    if not ids:
        print("[v_cc_weight_ablation] no Q-score cohort maps found", file=sys.stderr)
        return 1

    existing = {} if args.force or args.emd_id else _load_existing_rows(OUT_CSV)
    if args.resume and existing and not args.force:
        print(f"[v_cc_weight_ablation] resume: {len(existing)} maps already in {OUT_CSV}", flush=True)
    elif not args.emd_id and OUT_CSV.is_file() and not args.resume:
        OUT_CSV.unlink()

    rows_by_id: dict[str, dict[str, float | int | str]] = dict(existing)
    write_header = not OUT_CSV.is_file() or OUT_CSV.stat().st_size == 0

    for i, emd_id in enumerate(ids, 1):
        if args.resume and not args.force and emd_id in rows_by_id:
            print(f"[v_cc_weight_ablation] ({i}/{len(ids)}) EMD-{emd_id} skip (done)", flush=True)
            continue
        print(f"[v_cc_weight_ablation] ({i}/{len(ids)}) EMD-{emd_id}", flush=True)
        try:
            row = run_one(emd_id, manifest=args.manifest, window=args.window, cc_window=args.cc_window)
        except Exception as exc:
            print(f"[v_cc_weight_ablation] EMD-{emd_id} FAILED: {exc}", file=sys.stderr, flush=True)
            continue
        rows_by_id[emd_id] = row
        _append_row(OUT_CSV, row, write_header=write_header)
        write_header = False

    rows = [rows_by_id[eid] for eid in ids if eid in rows_by_id]
    print(f"\n[v_cc_weight_ablation] {len(rows)}/{len(ids)} maps → {OUT_CSV}", flush=True)
    _summarize(rows)
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
