"""Per-structure Spearman summary: map-only proxies vs Q-score and EMRinger.

For every manifest entry with a flat EMRinger CSV this writes in-mask Spearman ρ for:

- ρ(V, EMRinger), ρ(ResMap, EMRinger)
- ρ(Q, EMRinger), ρ(Q, V), ρ(Q, ResMap) when ``qscore_validation.csv`` exists

EMRinger headline medians use maps with global resolution ≤5 Å (Barad et al. 2015).
Expansion cohort maps with ``resmap_expected_failure=document`` (megacomplexes,
membrane assemblies, partial composites) are flagged; V remains valid when ResMap
returns flat 100 Å sentinel output.

Output: ``outputs/cohort_summary/emringer_cross_metric_summary.csv`` (core)
        ``outputs/cohort_summary/emringer_cross_metric_summary_expansion.csv``

Example::

    source .venv/bin/activate
    python scripts/run_emringer_cross_metric_summary.py --resume
    python scripts/run_emringer_cross_metric_summary.py --emd-id 49450
    python scripts/run_emringer_cross_metric_summary.py \\
        --manifest cohort/expansion_manifest.csv \\
        --out outputs/cohort_summary/emringer_cross_metric_summary_expansion.csv
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
from thesis.incremental_prediction import load_qscore_target, normalize_metrics_columns
from cryoem_mrc.local_resolution import RESMAP_UNRESOLVED_SENTINEL_A
from cryoem_mrc.manifest_policy import (
    cohort_tag_for_manifest,
    load_manifest_policy_by_emdb,
    row_qscore_eligible,
    row_resmap_ca_headline_eligible,
    row_resmap_expected_failure,
)
from cryoem_mrc.repo_paths import (
    COHORT_MANIFEST,
    EMRINGER_FLAT_DIR,
    EXPANSION_COHORT_MANIFEST,
    OUTPUTS_ROOT,
    emd_output_dir,
)

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

OUTPUT_CSV = OUTPUTS_ROOT / "cohort_summary" / "emringer_cross_metric_summary.csv"
OUTPUT_CSV_EXPANSION = (
    OUTPUTS_ROOT / "cohort_summary" / "emringer_cross_metric_summary_expansion.csv"
)
MIN_FINITE = 30
MIN_PAIRS = 10

SUMMARY_COLUMNS = [
    "emdb_id",
    "display_name",
    "pdb_code",
    "cohort_tag",
    "global_resolution_a",
    "emringer_interpretable",
    "building_regime_panel",
    "emringer_panel_reason",
    "qscore_eligible",
    "resmap_expected_failure",
    "resmap_headline_eligible",
    "resmap_usable",
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
    "rho_q_emringer",
    "p_value_q_emringer",
    "n_pairs_q_emringer",
    "nan_reason_q_emringer",
    "rho_q_v",
    "p_value_q_v",
    "n_pairs_q_v",
    "nan_reason_q_v",
    "rho_q_resmap",
    "p_value_q_resmap",
    "n_pairs_q_resmap",
    "nan_reason_q_resmap",
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
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV (default: core or expansion path from --manifest)",
    )
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


def _default_out_path(manifest: Path) -> Path:
    if cohort_tag_for_manifest(manifest) == "expansion":
        return OUTPUT_CSV_EXPANSION
    return OUTPUT_CSV


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
    df = pd.DataFrame(rows)
    for col in SUMMARY_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df[SUMMARY_COLUMNS].sort_values("rho_v_emringer", na_position="last").to_csv(out, index=False)


def _load_base_metrics(emdb_id: str) -> pd.DataFrame:
    base_path = emd_output_dir(emdb_id) / "metric_comparison" / "residue_metrics.csv"
    if not base_path.is_file():
        raise FileNotFoundError(f"missing {base_path}")
    return normalize_metrics_columns(pd.read_csv(base_path))


def _attach_resmap_locres(df: pd.DataFrame, emdb_id: str) -> pd.DataFrame:
    resmap_path = (
        emd_output_dir(emdb_id) / "metric_comparison_resmap" / "residue_metrics.csv"
    )
    out = df.drop(columns=["local_resolution"], errors="ignore")
    if not resmap_path.is_file():
        out["local_resolution"] = np.nan
        return out
    resmap = pd.read_csv(resmap_path, usecols=["chain", "seq_num", "local_resolution"])
    return out.merge(resmap, on=["chain", "seq_num"], how="left")


def _resmap_usable(sub: pd.DataFrame) -> bool:
    loc = pd.to_numeric(sub["local_resolution"], errors="coerce")
    finite = loc[np.isfinite(loc)]
    if int(finite.notna().sum()) < MIN_FINITE:
        return False
    # ResMap failure mode: flat sentinel (100 Å) or near-zero variance.
    usable = finite[finite < RESMAP_UNRESOLVED_SENTINEL_A - 1.0]
    if int(usable.notna().sum()) < MIN_FINITE:
        return False
    return float(np.var(usable.to_numpy(dtype=float))) > 1e-6


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
    if x_col not in sub.columns or y_col not in sub.columns:
        return None
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


def _apply_spearman_block(
    row: dict,
    prefix: str,
    summary: dict | None,
    *,
    fallback_x_in_mask: int,
    fallback_y_in_mask: int,
    insufficient_reason: str,
) -> None:
    if summary is not None:
        row[f"rho_{prefix}"] = summary["rho"]
        row[f"p_value_{prefix}"] = summary["p_value"]
        row[f"n_pairs_{prefix}"] = summary["n_pairs"]
        row[f"nan_reason_{prefix}"] = summary["nan_reason"]
        return
    row[f"rho_{prefix}"] = float("nan")
    row[f"p_value_{prefix}"] = float("nan")
    row[f"n_pairs_{prefix}"] = 0
    row[f"nan_reason_{prefix}"] = insufficient_reason


def _summarize_entry(
    emdb_id: str,
    *,
    emringer_csv: Path,
    pdb_path: Path,
) -> dict | None:
    try:
        df = _attach_resmap_locres(_load_base_metrics(emdb_id), emdb_id)
        df = attach_emringer_scores(df, emringer_csv, pdb_path=pdb_path)
        q_df = load_qscore_target(df, emdb_id)
        if q_df is not None:
            df = q_df
    except (FileNotFoundError, ValueError, KeyError) as exc:
        logger.warning("skip EMD-%s: %s", emdb_id, exc)
        return None

    sub = df[df["in_contour_mask"].astype(bool)]
    has_q = "q_score" in sub.columns

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
    q_em_sum = (
        _spearman_summary(
            sub,
            "q_score",
            "emringer_score",
            min_finite_x=MIN_FINITE,
            min_finite_y=MIN_FINITE,
            x_var_name="q",
            y_var_name="emringer",
        )
        if has_q
        else None
    )
    q_v_sum = (
        _spearman_summary(
            sub,
            "q_score",
            "v_metric",
            min_finite_x=MIN_FINITE,
            min_finite_y=MIN_FINITE,
            x_var_name="q",
            y_var_name="v",
        )
        if has_q
        else None
    )
    q_res_sum = (
        _spearman_summary(
            sub,
            "q_score",
            "local_resolution",
            min_finite_x=MIN_FINITE,
            min_finite_y=MIN_FINITE,
            x_var_name="q",
            y_var_name="resmap",
        )
        if has_q
        else None
    )

    if v_sum is None and res_sum is None and q_em_sum is None:
        return None

    row: dict = {
        "emdb_id": emdb_id,
        "resmap_usable": _resmap_usable(sub),
    }

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

    _apply_spearman_block(
        row,
        "q_emringer",
        q_em_sum,
        fallback_x_in_mask=int(sub["q_score"].notna().sum()) if has_q else 0,
        fallback_y_in_mask=int(sub["emringer_score"].notna().sum()),
        insufficient_reason="missing_qscore_validation" if not has_q else "insufficient_finite_q_or_emringer",
    )
    _apply_spearman_block(
        row,
        "q_v",
        q_v_sum,
        fallback_x_in_mask=0,
        fallback_y_in_mask=0,
        insufficient_reason="missing_qscore_validation" if not has_q else "insufficient_finite_q_or_v",
    )
    _apply_spearman_block(
        row,
        "q_resmap",
        q_res_sum,
        fallback_x_in_mask=0,
        fallback_y_in_mask=0,
        insufficient_reason="missing_qscore_validation" if not has_q else "insufficient_finite_q_or_resmap",
    )
    return row


def _median_rho(series: pd.Series, *, sign_align: bool = False) -> float:
    v = pd.to_numeric(series, errors="coerce").dropna()
    if sign_align:
        v = -v
    return float(v.median()) if len(v) else float("nan")


def _print_medians(out_df: pd.DataFrame, *, max_resolution_a: float) -> None:
    interpretable = out_df[out_df["emringer_interpretable"].astype(bool)]
    building = out_df[out_df["building_regime_panel"].astype(bool)]
    with_q = out_df[pd.to_numeric(out_df["rho_q_emringer"], errors="coerce").notna()]

    print(
        f"[emringer_xmetric] V vs EMRinger (full n={len(out_df)}): "
        f"median ρ={_median_rho(out_df['rho_v_emringer']):+.3f}",
        flush=True,
    )
    print(
        f"[emringer_xmetric] EMRinger panel (≤{max_resolution_a:g} Å, "
        f"{EMRINGER_BARAD_2015_CITATION}; n={len(interpretable)}): "
        f"V ρ={_median_rho(interpretable['rho_v_emringer']):+.3f}, "
        f"ResMap ρ={_median_rho(interpretable['rho_resmap_emringer'], sign_align=True):+.3f} "
        f"(raw {_median_rho(interpretable['rho_resmap_emringer']):+.3f})",
        flush=True,
    )
    if len(with_q):
        print(
            f"[emringer_xmetric] Q-score triangle (n={len(with_q)} maps with qscore_validation): "
            f"ρ(Q, EMRinger)={_median_rho(with_q['rho_q_emringer']):+.3f}, "
            f"ρ(Q, V)={_median_rho(with_q['rho_q_v']):+.3f}, "
            f"ρ(Q, ResMap)={_median_rho(with_q['rho_q_resmap'], sign_align=True):+.3f}",
            flush=True,
        )

    resmap_fail = out_df[out_df["resmap_expected_failure"].astype(str).str.lower() == "document"]
    if len(resmap_fail):
        usable = resmap_fail[resmap_fail["resmap_usable"].astype(bool)]
        print(
            f"[emringer_xmetric] resmap_expected_failure=document (n={len(resmap_fail)}): "
            f"V vs EMRinger median ρ={_median_rho(resmap_fail['rho_v_emringer']):+.3f}; "
            f"ResMap usable on {len(usable)}/{len(resmap_fail)} maps",
            flush=True,
        )
        if len(usable):
            print(
                f"  ResMap-usable subset: "
                f"ρ(V, EMR)={_median_rho(usable['rho_v_emringer']):+.3f}, "
                f"|ρ(ResMap, EMR)|={_median_rho(usable['rho_resmap_emringer'], sign_align=True):+.3f}",
                flush=True,
            )

    print(
        f"[emringer_xmetric] building regime "
        f"({BUILDING_REGIME_MIN_RESOLUTION_A:g}–{BUILDING_REGIME_MAX_RESOLUTION_A:g} Å, "
        f"n={len(building)}): "
        f"V ρ={_median_rho(building['rho_v_emringer']):+.3f}, "
        f"ResMap ρ={_median_rho(building['rho_resmap_emringer'], sign_align=True):+.3f}",
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
    out_path = args.out or _default_out_path(args.manifest)
    cohort_tag = cohort_tag_for_manifest(args.manifest)
    policy_by_emdb = load_manifest_policy_by_emdb(args.manifest)

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
        f"[emringer_xmetric] manifest={args.manifest} cohort={cohort_tag} "
        f"lookup={len(lookup)} structures with flat EMRinger CSV",
        flush=True,
    )

    existing = _load_existing_rows(out_path) if args.resume else {}
    rows: list[dict] = list(existing.values())

    for emdb_id, emringer_csv in sorted(lookup.items(), key=lambda kv: kv[0]):
        if args.emd_id and emdb_id != args.emd_id.strip():
            continue
        if args.resume and emdb_id in existing and "rho_q_emringer" in existing[emdb_id]:
            print(f"[emringer_xmetric] EMD-{emdb_id}: resume skip", flush=True)
            continue
        summary = _summarize_entry(
            emdb_id,
            emringer_csv=emringer_csv,
            pdb_path=pdb_paths[emdb_id],
        )
        if summary is None:
            continue
        pol = policy_by_emdb.get(emdb_id, {})
        summary["display_name"] = names.get(emdb_id, "")
        summary["pdb_code"] = pdb_codes.get(emdb_id, "")
        summary["cohort_tag"] = cohort_tag
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
        summary["qscore_eligible"] = row_qscore_eligible(pol) if pol else False
        summary["resmap_expected_failure"] = str(
            pol.get("resmap_expected_failure", "") or ""
        ).strip()
        summary["resmap_headline_eligible"] = (
            row_resmap_ca_headline_eligible(pol) if pol else True
        )
        summary["emringer_csv"] = str(emringer_csv)
        rows = [r for r in rows if str(r["emdb_id"]).strip() != emdb_id]
        rows.append(summary)
        _write_rows(out_path, rows)
        panel = "panel" if summary["emringer_interpretable"] else "excluded"
        q_part = ""
        if pd.notna(summary.get("rho_q_emringer")):
            q_part = f" Q×EMR ρ={summary['rho_q_emringer']:+.3f}"
        print(
            f"[emringer_xmetric] EMD-{emdb_id} ({summary['pdb_code']}, "
            f"{summary['global_resolution_a']:.2f} Å, {panel}): "
            f"V ρ={summary['rho_v_emringer']:+.3f} "
            f"ResMap ρ={summary['rho_resmap_emringer']:+.3f}"
            f"{q_part}",
            flush=True,
        )

    if not rows:
        print("[emringer_xmetric] no eligible entries", file=sys.stderr)
        return 1

    out_df = pd.DataFrame(rows)
    for col in SUMMARY_COLUMNS:
        if col not in out_df.columns:
            out_df[col] = np.nan
    _print_medians(out_df[SUMMARY_COLUMNS], max_resolution_a=args.max_resolution_a)
    print(f"[emringer_xmetric] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
