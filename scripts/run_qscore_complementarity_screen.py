"""Screen gradient-family + Hessian λ_min predictors vs Q-score.

Focuses on constraint V, von Weizsäcker T, Hessian curvature V, and −λ_min.
Reports per-map ρ(Q, ·), cross-feature decorrelation, and LOMO rank-OLS beyond
{local variance, half-map CC, BlocRes}.

Example::

    source .venv/bin/activate
    PYTHONUNBUFFERED=1 python scripts/run_qscore_complementarity_screen.py --emd-id 49450
    python scripts/run_qscore_complementarity_screen.py --resume --force
"""

from __future__ import annotations

import argparse
import csv
import gc
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np
import pandas as pd

from thesis.complementarity import (
    FOCUS_PREDICTORS,
    GRADIENT_FAMILY_COLUMNS,
    LOMO_MODEL_FEATURES,
    build_map_frame_for_model,
    pairwise_spearman_median,
    run_lomo_model_screen,
    spearman_vs_q,
)
from cryoem_mrc.density_source import zscore_halfmap_average
from cryoem_mrc.hessian import density_hessian_scalar_maps
from cryoem_mrc.map_grid import load_full_and_half_maps, load_map_grid
from cryoem_mrc.repo_paths import COHORT_MANIFEST, OUTPUTS_ROOT, glob_halfmap_reliability_files, resolve_halfmap_reliability_dir
from cryoem_mrc.structure_validation import iter_ca_residues, load_cohort_manifest_row, sample_volume_at_ca
from thesis.tv_curvature import density_tv_curvature_maps
from thesis.incremental_prediction import load_qscore_target, load_metrics_dataframe
from thesis.metric_comparison import load_all_metrics

from thesis.qscore_cohort import filter_emdb_ids, qscore_exclude_ids
DEFAULT_SPHERE_A = 2.0

PER_MAP_FIELDS = ["emdb_id", "n_used", *[f"rho_q_{c}" for c in FOCUS_PREDICTORS]]

OUT_PER_MAP = OUTPUTS_ROOT / "cohort_summary" / "qscore_complementarity_per_map.csv"
OUT_PAIRWISE = OUTPUTS_ROOT / "cohort_summary" / "qscore_complementarity_pairwise.csv"
OUT_LOMO = OUTPUTS_ROOT / "cohort_summary" / "qscore_complementarity_lomo.csv"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, default=COHORT_MANIFEST)
    p.add_argument("--chunk-z", type=int, default=DEFAULT_CHUNK_Z)
    p.add_argument("--sphere-a", type=float, default=DEFAULT_SPHERE_A)
    p.add_argument("--emd-id", type=str, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument(
        "--core",
        action="store_true",
        help="Use core Q-score cohort exclusions (thesis headline panel)",
    )
    return p.parse_args(argv)


def _emd_ids_from_qscore_cohort(*, core: bool = False) -> list[str]:
    ids: list[str] = []
    csv_path = OUTPUTS_ROOT / "cohort_summary" / "qscore_correlations.csv"
    if csv_path.is_file():
        with csv_path.open(newline="") as f:
            for row in csv.DictReader(f):
                eid = str(row["emdb_id"]).strip()
                ids.append(eid)
    else:
        for bundle in glob_halfmap_reliability_files(OUTPUTS_ROOT, "qscore_validation.csv"):
            eid = bundle.parent.parent.name.removeprefix("emd_")
            if eid not in ids:
                ids.append(eid)
    return filter_emdb_ids(ids, core=core)


def run_one(
    emd_id: str,
    *,
    manifest: Path,
    chunk_z: int,
    sphere_a: float,
) -> pd.DataFrame:
    row = load_cohort_manifest_row(manifest, emd_id)
    q_path = resolve_halfmap_reliability_dir(emd_id) / "qscore_validation.csv"
    if not q_path.is_file():
        raise FileNotFoundError(f"EMD-{emd_id}: missing {q_path}")

    metrics = load_metrics_dataframe(emd_id, manifest=manifest, sphere_radius_a=sphere_a)
    if metrics is None:
        metrics = load_all_metrics(emd_id, manifest=manifest, sphere_radius_a=sphere_a)
    metrics = load_qscore_target(metrics, emd_id)
    if metrics is None:
        raise FileNotFoundError(f"EMD-{emd_id}: could not merge Q-scores")

    ref_path = Path(row["reference_mrc"])
    pdb_path = Path(row["flexibility_path_or_pdb"])
    grid = load_map_grid(ref_path, dtype=np.float32)
    residues = iter_ca_residues(pdb_path)

    metrics = metrics.copy()
    metrics["constraint_V"] = pd.to_numeric(metrics["v_metric"], errors="coerce")

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

    hess = density_hessian_scalar_maps(rho_z, chunk_z=chunk_z)
    lam_min = np.asarray(hess.pop("hessian_eig_min"), dtype=np.float32)
    metrics["neg_lam_min"] = sample_volume_at_ca(lam_min, grid, residues, sphere_radius_a=sphere_a)
    del hess, lam_min, rho_z
    gc.collect()

    tv = density_tv_curvature_maps(
        np.asarray(grid.data, dtype=np.float32),
        spacing_zyx=grid.voxel_size_zyx,
        chunk_z=chunk_z,
    )
    metrics["T_vonweizsacker"] = sample_volume_at_ca(
        tv["T_vonweizsacker"], grid, residues, sphere_radius_a=sphere_a
    )
    metrics["V_curvature"] = sample_volume_at_ca(
        tv["V_curvature"], grid, residues, sphere_radius_a=sphere_a
    )
    del tv
    gc.collect()

    metrics["emdb_id"] = emd_id
    return metrics


def _load_existing_per_map(path: Path) -> dict[str, dict[str, object]]:
    if not path.is_file():
        return {}
    out: dict[str, dict[str, object]] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            eid = str(row.get("emdb_id", "")).strip()
            if eid:
                out[eid] = row
    return out


def _append_csv(path: Path, row: dict[str, object], fieldnames: list[str], *, write_header: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fieldnames})


def _as_float(val: object) -> float:
    try:
        return float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


def _summarize_per_map(rows: list[dict[str, object]]) -> None:
    print("\n=== Median ρ(Q, predictor) across maps ===")
    for col in FOCUS_PREDICTORS:
        key = f"rho_q_{col}"
        vals = [_as_float(r.get(key)) for r in rows]
        vals = [x for x in vals if np.isfinite(x)]
        if not vals:
            continue
        print(f"  {col:24s}  median ρ = {float(np.median(vals)):+.4f}  (n={len(vals)})")

    grad_keys = [f"rho_q_{c}" for c in GRADIENT_FAMILY_COLUMNS]
    grad_medians = [
        float(np.median([_as_float(r.get(k)) for r in rows if np.isfinite(_as_float(r.get(k)))]))
        for k in grad_keys
    ]
    if all(np.isfinite(x) for x in grad_medians):
        spread = max(grad_medians) - min(grad_medians)
        print(f"\n  gradient-family spread (max−min median ρ) = {spread:.4f}")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    ids = [args.emd_id] if args.emd_id else _emd_ids_from_qscore_cohort(core=args.core)
    if not ids:
        print("[complementarity] no Q-score cohort maps found", file=sys.stderr)
        return 1

    existing = {} if args.force or args.emd_id else _load_existing_per_map(OUT_PER_MAP)
    if not args.emd_id and OUT_PER_MAP.is_file() and not args.resume and not args.force:
        OUT_PER_MAP.unlink()

    per_map_rows: list[dict[str, object]] = []
    tables: list[pd.DataFrame] = []
    write_header = not OUT_PER_MAP.is_file() or OUT_PER_MAP.stat().st_size == 0

    for i, emd_id in enumerate(ids, 1):
        if args.resume and not args.force and emd_id in existing:
            print(f"[complementarity] ({i}/{len(ids)}) EMD-{emd_id} skip (done)", flush=True)
            continue
        print(f"[complementarity] ({i}/{len(ids)}) EMD-{emd_id}", flush=True)
        try:
            df = run_one(
                emd_id,
                manifest=args.manifest,
                chunk_z=args.chunk_z,
                sphere_a=args.sphere_a,
            )
        except Exception as exc:
            print(f"[complementarity] EMD-{emd_id} FAILED: {exc}", file=sys.stderr, flush=True)
            continue

        rhos = spearman_vs_q(df, FOCUS_PREDICTORS)
        n_used = int(df.loc[df["in_contour_mask"].astype(bool), "q_score"].notna().sum())
        row: dict[str, object] = {"emdb_id": emd_id, "n_used": n_used}
        for name, rho in rhos.items():
            row[f"rho_q_{name}"] = rho
        _append_csv(OUT_PER_MAP, row, PER_MAP_FIELDS, write_header=write_header)
        write_header = False
        per_map_rows.append(row)
        tables.append(df)
        del df
        gc.collect()

    if existing and args.resume and not args.force:
        for eid, row in existing.items():
            if eid not in {r["emdb_id"] for r in per_map_rows}:
                per_map_rows.append(row)

    _summarize_per_map(per_map_rows)

    if len(tables) >= 2:
        pair_df = pairwise_spearman_median(tables, FOCUS_PREDICTORS)
        pair_df.to_csv(OUT_PAIRWISE, index=False)
        print(f"\n[complementarity] pairwise medians → {OUT_PAIRWISE}", flush=True)
        print("\n=== Cross-feature ρ (focus: grad cluster vs −λ_min) ===")
        for _, r in pair_df.sort_values("median_rho").iterrows():
            print(
                f"  {r['feature_i']:20s} vs {r['feature_j']:20s}  "
                f"median ρ = {r['median_rho']:+.3f}  (n={int(r['n_maps'])})"
            )

    frames_by_model: dict[str, list] = {k: [] for k in LOMO_MODEL_FEATURES}
    for df in tables:
        eid = str(df["emdb_id"].iloc[0])
        for model in LOMO_MODEL_FEATURES:
            frame = build_map_frame_for_model(df, emdb_id=eid, model=model)
            if frame is not None:
                frames_by_model[model].append(frame)

    summaries = run_lomo_model_screen(frames_by_model, target="q_score")
    if summaries:
        print("\n=== LOMO rank-OLS vs baseline {var, CC, locres} ===")
        base_med = next((s.median_r2 for s in summaries if s.model == "baseline"), float("nan"))
        print(f"  baseline median R² = {base_med:.4f}")
        lomo_rows: list[dict[str, object]] = []
        for s in summaries:
            if s.model == "baseline":
                continue
            print(
                f"  {s.model:18s}  median R²={s.median_r2:.4f}  "
                f"ΔR² vs baseline={s.median_delta_r2_vs_baseline:+.4f}  "
                f"maps improved={s.n_positive_delta_r2}/{s.n_maps}"
            )
            lomo_rows.append(
                {
                    "model": s.model,
                    "n_maps": s.n_maps,
                    "median_r2": s.median_r2,
                    "median_delta_r2_vs_baseline": s.median_delta_r2_vs_baseline,
                    "n_positive_delta_r2": s.n_positive_delta_r2,
                    "sign_test_p_value": s.sign_test_p_value,
                }
            )
        pd.DataFrame(lomo_rows).to_csv(OUT_LOMO, index=False)
        print(f"\n[complementarity] LOMO summary → {OUT_LOMO}", flush=True)
    else:
        print("\n[complementarity] LOMO skipped (need ≥3 maps with complete features)", flush=True)

    print(f"\n[complementarity] {len(per_map_rows)}/{len(ids)} maps → {OUT_PER_MAP}", flush=True)
    return 0 if per_map_rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
