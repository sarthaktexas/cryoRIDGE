"""Per-structure Spearman summary: V vs EMRinger and ResMap vs EMRinger.

For every cohort entry with a flat EMRinger CSV and cached ResMap metrics this writes
one row with in-mask Spearman ρ for both comparisons on the same deposited model.

EMRinger headline medians use maps with global resolution ≤5 Å (Barad et al. 2015
breakdown threshold). Maps coarser than 5 Å are flagged ``emringer_interpretable=False``.
The 2.5–4 Å model-building band is flagged separately. V vs EMRinger medians are
also reported on the full deposited cohort.

Output: ``outputs/cohort_summary/emringer_cross_metric_summary.csv``

Example::

    source .venv/bin/activate
    python scripts/run_emringer_cross_metric_summary.py --resume
    python scripts/run_emringer_cross_metric_summary.py --emd-id 49450
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from cryoem_mrc.emringer import (
    attach_emringer_scores,
    build_manifest_emringer_lookup,
    missing_emringer_csvs,
    pdb_code_from_flexibility_path,
)
from cryoem_mrc.emringer_cohort import (
    BUILDING_REGIME_MAX_RESOLUTION_A,
    BUILDING_REGIME_MIN_RESOLUTION_A,
    EMRINGER_BARAD_2015_CITATION,
    EMRINGER_INTERPRETABLE_MAX_RESOLUTION_A,
    building_regime_panel,
    emringer_interpretable,
    emringer_panel_reason,
    load_manifest_global_resolution_a,
)
from cryoem_mrc.repo_paths import COHORT_MANIFEST, EMRINGER_FLAT_DIR, OUTPUTS_ROOT, emd_output_dir

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

OUTPUT_CSV = OUTPUTS_ROOT / "cohort_summary" / "emringer_cross_metric_summary.csv"
MIN_FINITE = 30
MIN_PAIRS = 10

SUMMARY_COLUMNS = [
    "emdb_id",
    "display_name",
    "pdb_code",
    "global_resolution_a",
    "emringer_interpretable",
    "building_regime_panel",
    "emringer_panel_reason",
    "emringer_csv",
    "rho_v_emringer",
    "p_value_v",
    "n_pairs_v",
    "n_v_in_mask",
    "n_emringer_in_mask",
    "v_variance",
    "emringer_variance_v",
    "nan_reason_v",
    "rho_resmap_emringer",
    "p_value_resmap",
    "n_pairs_resmap",
    "n_resmap_in_mask",
    "resmap_variance",
    "emringer_variance_resmap",
    "nan_reason_resmap",
]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--manifest", type=Path, default=COHORT_MANIFEST)
    p.add_argument(
        "--emringer-dir",
        type=Path,
        default=EMRINGER_FLAT_DIR,
        help="Flat directory of {pdb_code}_emringer.csv files",
    )
    p.add_argument("--out", type=Path, default=OUTPUT_CSV)
    p.add_argument("--emd-id", type=str, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument(
        "--require-all",
        action="store_true",
        help="Exit with error if any deposited structure lacks a flat EMRinger CSV",
    )
    p.add_argument(
        "--max-resolution-a",
        type=float,
        default=EMRINGER_INTERPRETABLE_MAX_RESOLUTION_A,
        help="EMRinger interpretable when global resolution (Å) is at or below this cutoff "
        f"(default {EMRINGER_INTERPRETABLE_MAX_RESOLUTION_A:g}, Barad et al. 2015 breakdown)",
    )
    return p.parse_args(argv)


def _configure_logging(*, verbose: bool) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if not verbose:
        logging.getLogger("cryoem_mrc.local_resolution").setLevel(logging.ERROR)


def _display_names(manifest: Path) -> dict[str, str]:
    names: dict[str, str] = {}
    with manifest.open(newline="") as f:
        for row in csv.DictReader(f):
            eid = str(row.get("emdb_id", "")).strip()
            if eid:
                names[eid] = row.get("display_name", "").strip()
    return names


def _pdb_codes(manifest: Path) -> dict[str, str]:
    codes: dict[str, str] = {}
    with manifest.open(newline="") as f:
        for row in csv.DictReader(f):
            eid = str(row.get("emdb_id", "")).strip()
            pdb_raw = row.get("flexibility_path_or_pdb", "").strip()
            if eid and pdb_raw and Path(pdb_raw).is_file():
                codes[eid] = pdb_code_from_flexibility_path(pdb_raw)
    return codes


def _pdb_paths(manifest: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    with manifest.open(newline="") as f:
        for row in csv.DictReader(f):
            eid = str(row.get("emdb_id", "")).strip()
            pdb_raw = row.get("flexibility_path_or_pdb", "").strip()
            if eid and pdb_raw and Path(pdb_raw).is_file():
                paths[eid] = Path(pdb_raw)
    return paths


def _load_existing_rows(out: Path) -> dict[str, dict]:
    if not out.is_file():
        return {}
    df = pd.read_csv(out)
    return {str(row["emdb_id"]).strip(): row.to_dict() for _, row in df.iterrows()}


def _write_rows(out: Path, rows: list[dict]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows)[SUMMARY_COLUMNS].sort_values(
        "rho_v_emringer", na_position="last"
    ).to_csv(out, index=False)


def _load_metrics_with_resmap(emdb_id: str) -> pd.DataFrame:
    base_path = emd_output_dir(emdb_id) / "metric_comparison" / "residue_metrics.csv"
    resmap_path = (
        emd_output_dir(emdb_id) / "metric_comparison_resmap" / "residue_metrics.csv"
    )
    if not base_path.is_file():
        raise FileNotFoundError(f"missing {base_path}")
    if not resmap_path.is_file():
        raise FileNotFoundError(f"missing {resmap_path}")
    base = pd.read_csv(base_path)
    resmap = pd.read_csv(resmap_path, usecols=["chain", "seq_num", "local_resolution"])
    out = base.drop(columns=["local_resolution"], errors="ignore")
    return out.merge(resmap, on=["chain", "seq_num"], how="left")


def _spearman_summary(
    sub: pd.DataFrame,
    x_col: str,
    y_col: str,
    *,
    min_finite_x: int,
    min_finite_y: int,
    x_var_name: str,
    y_var_name: str,
) -> dict | None:
    n_x = int(sub[x_col].notna().sum())
    n_y = int(sub[y_col].notna().sum())
    if n_x < min_finite_x or n_y < min_finite_y:
        return None

    paired = sub[x_col].notna() & sub[y_col].notna()
    n_pairs = int(paired.sum())
    x = sub.loc[paired, x_col].to_numpy(dtype=float)
    y = sub.loc[paired, y_col].to_numpy(dtype=float)
    x_var = float(np.var(x)) if n_pairs else float("nan")
    y_var = float(np.var(y)) if n_pairs else float("nan")

    rho = float("nan")
    pval = float("nan")
    nan_reason = ""
    if n_pairs < MIN_PAIRS:
        nan_reason = f"n_pairs<{MIN_PAIRS}"
    elif x_var == 0.0 and y_var == 0.0:
        nan_reason = f"zero_{x_var_name}_and_{y_var_name}_variance"
    elif x_var == 0.0:
        nan_reason = f"zero_{x_var_name}_variance"
    elif y_var == 0.0:
        nan_reason = f"zero_{y_var_name}_variance"
    else:
        r, p = stats.spearmanr(x, y)
        rho, pval = float(r), float(p)
        if not np.isfinite(rho):
            nan_reason = "spearman_undefined"

    return {
        "rho": rho,
        "p_value": pval,
        "n_pairs": n_pairs,
        f"n_{x_var_name}_in_mask": n_x,
        f"n_{y_var_name}_in_mask": n_y,
        f"{x_var_name}_variance": x_var,
        f"{y_var_name}_variance": y_var,
        "nan_reason": nan_reason,
    }


def _summarize_entry(
    emdb_id: str,
    *,
    manifest: Path,
    emringer_csv: Path,
    pdb_path: Path,
) -> dict | None:
    try:
        df = attach_emringer_scores(
            _load_metrics_with_resmap(emdb_id),
            emringer_csv,
            pdb_path=pdb_path,
        )
    except (FileNotFoundError, ValueError, KeyError) as exc:
        logger.warning("skip EMD-%s: %s", emdb_id, exc)
        return None

    sub = df[df["in_contour_mask"].astype(bool)]
    v_sum = _spearman_summary(
        sub,
        "v_metric",
        "emringer_score",
        min_finite_x=MIN_FINITE,
        min_finite_y=MIN_FINITE,
        x_var_name="v",
        y_var_name="emringer",
    )
    res_sum = _spearman_summary(
        sub,
        "local_resolution",
        "emringer_score",
        min_finite_x=MIN_FINITE,
        min_finite_y=MIN_FINITE,
        x_var_name="resmap",
        y_var_name="emringer",
    )
    if v_sum is None and res_sum is None:
        return None

    row: dict = {"emdb_id": emdb_id}
    if v_sum is not None:
        row.update(
            {
                "rho_v_emringer": v_sum["rho"],
                "p_value_v": v_sum["p_value"],
                "n_pairs_v": v_sum["n_pairs"],
                "n_v_in_mask": v_sum["n_v_in_mask"],
                "n_emringer_in_mask": v_sum["n_emringer_in_mask"],
                "v_variance": v_sum["v_variance"],
                "emringer_variance_v": v_sum["emringer_variance"],
                "nan_reason_v": v_sum["nan_reason"],
            }
        )
    else:
        row.update(
            {
                "rho_v_emringer": float("nan"),
                "p_value_v": float("nan"),
                "n_pairs_v": 0,
                "n_v_in_mask": int(sub["v_metric"].notna().sum()),
                "n_emringer_in_mask": int(sub["emringer_score"].notna().sum()),
                "v_variance": float("nan"),
                "emringer_variance_v": float("nan"),
                "nan_reason_v": "insufficient_finite_v_or_emringer",
            }
        )

    if res_sum is not None:
        row.update(
            {
                "rho_resmap_emringer": res_sum["rho"],
                "p_value_resmap": res_sum["p_value"],
                "n_pairs_resmap": res_sum["n_pairs"],
                "n_resmap_in_mask": res_sum["n_resmap_in_mask"],
                "resmap_variance": res_sum["resmap_variance"],
                "emringer_variance_resmap": res_sum["emringer_variance"],
                "nan_reason_resmap": res_sum["nan_reason"],
            }
        )
    else:
        row.update(
            {
                "rho_resmap_emringer": float("nan"),
                "p_value_resmap": float("nan"),
                "n_pairs_resmap": 0,
                "n_resmap_in_mask": int(sub["local_resolution"].notna().sum()),
                "resmap_variance": float("nan"),
                "emringer_variance_resmap": float("nan"),
                "nan_reason_resmap": "insufficient_finite_resmap_or_emringer",
            }
        )
    return row


def _print_medians(out_df: pd.DataFrame, *, max_resolution_a: float) -> None:
    interpretable = out_df[out_df["emringer_interpretable"].astype(bool)]
    building = out_df[out_df["building_regime_panel"].astype(bool)]
    full_v = out_df["rho_v_emringer"].dropna()
    panel_v = interpretable["rho_v_emringer"].dropna()
    panel_r = interpretable["rho_resmap_emringer"].dropna()
    build_v = building["rho_v_emringer"].dropna()
    build_r = building["rho_resmap_emringer"].dropna()
    print(
        f"[emringer_xmetric] V vs EMRinger (full {len(full_v)} maps): "
        f"median ρ={float(full_v.median()) if len(full_v) else float('nan'):+.3f}",
        flush=True,
    )
    print(
        f"[emringer_xmetric] EMRinger panel (≤{max_resolution_a:g} Å, "
        f"{EMRINGER_BARAD_2015_CITATION}; n={len(interpretable)}): "
        f"V median ρ={float(panel_v.median()) if len(panel_v) else float('nan'):+.3f}, "
        f"ResMap median ρ={float(panel_r.median()) if len(panel_r) else float('nan'):+.3f}",
        flush=True,
    )
    print(
        f"[emringer_xmetric] building regime "
        f"({BUILDING_REGIME_MIN_RESOLUTION_A:g}–{BUILDING_REGIME_MAX_RESOLUTION_A:g} Å, "
        f"n={len(building)}): "
        f"V median ρ={float(build_v.median()) if len(build_v) else float('nan'):+.3f}, "
        f"ResMap median ρ={float(build_r.median()) if len(build_r) else float('nan'):+.3f}",
        flush=True,
    )
    excluded = out_df[~out_df["emringer_interpretable"].astype(bool)]
    if len(excluded):
        ids = ", ".join(str(x) for x in excluded["emdb_id"].astype(str).tolist())
        print(
            f"[emringer_xmetric] excluded (coarser than {max_resolution_a:g} Å, "
            f"n={len(excluded)}): {ids}",
            flush=True,
        )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _configure_logging(verbose=args.verbose)

    missing = missing_emringer_csvs(args.manifest, args.emringer_dir)
    if missing:
        ids = ", ".join(f"EMD-{r.emdb_id} ({r.pdb_code})" for r in missing)
        msg = f"[emringer_xmetric] missing flat EMRinger CSV for {len(missing)} deposits: {ids}"
        if args.require_all:
            print(msg, file=sys.stderr, flush=True)
            print(
                "[emringer_xmetric] run: python scripts/run_emringer_cohort_batch.py",
                file=sys.stderr,
                flush=True,
            )
            return 1
        print(msg, flush=True)

    names = _display_names(args.manifest)
    pdb_codes = _pdb_codes(args.manifest)
    pdb_paths = _pdb_paths(args.manifest)
    resolutions = load_manifest_global_resolution_a(args.manifest)
    lookup = build_manifest_emringer_lookup(
        args.manifest, args.emringer_dir, require_existing=True
    )
    print(
        f"[emringer_xmetric] lookup: {len(lookup)} structures with flat EMRinger CSV",
        flush=True,
    )

    existing = _load_existing_rows(args.out) if args.resume else {}
    rows: list[dict] = list(existing.values())

    for emdb_id, emringer_csv in sorted(lookup.items(), key=lambda kv: kv[0]):
        if args.emd_id and emdb_id != args.emd_id.strip():
            continue
        if args.resume and emdb_id in existing:
            print(f"[emringer_xmetric] EMD-{emdb_id}: resume skip", flush=True)
            continue
        summary = _summarize_entry(
            emdb_id,
            manifest=args.manifest,
            emringer_csv=emringer_csv,
            pdb_path=pdb_paths[emdb_id],
        )
        if summary is None:
            continue
        summary["display_name"] = names.get(emdb_id, "")
        summary["pdb_code"] = pdb_codes.get(emdb_id, "")
        summary["global_resolution_a"] = resolutions.get(emdb_id, float("nan"))
        summary["emringer_interpretable"] = emringer_interpretable(
            emdb_id,
            resolutions=resolutions,
            max_resolution_a=args.max_resolution_a,
        )
        summary["building_regime_panel"] = building_regime_panel(
            emdb_id, resolutions=resolutions
        )
        summary["emringer_panel_reason"] = emringer_panel_reason(
            emdb_id,
            resolutions=resolutions,
            max_resolution_a=args.max_resolution_a,
        )
        summary["emringer_csv"] = str(emringer_csv)
        rows = [r for r in rows if str(r["emdb_id"]).strip() != emdb_id]
        rows.append(summary)
        _write_rows(args.out, rows)
        panel = "panel" if summary["emringer_interpretable"] else "excluded"
        print(
            f"[emringer_xmetric] EMD-{emdb_id} ({summary['pdb_code']}, "
            f"{summary['global_resolution_a']:.2f} Å, {panel}): "
            f"V ρ={summary['rho_v_emringer']:+.3f} "
            f"ResMap ρ={summary['rho_resmap_emringer']:+.3f}",
            flush=True,
        )

    if not rows:
        print("[emringer_xmetric] no eligible entries", file=sys.stderr)
        return 1

    out_df = pd.DataFrame(rows)[SUMMARY_COLUMNS]
    _print_medians(out_df, max_resolution_a=args.max_resolution_a)
    print(f"[emringer_xmetric] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
