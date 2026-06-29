"""Export the per-structure V-metric vs EMRinger summary.

For every cohort entry with a deposited PDB and a matching flat EMRinger CSV
(``outputs/emringer_flat/{pdb_code}_emringer.csv``, one-to-one via manifest
``flexibility_path_or_pdb``) this writes one row of in-mask Spearman statistics
between V at Cα and per-residue EMRinger scores.

Headline EMRinger medians use maps with global resolution ≤5 Å (Barad et al. 2015).
Maps coarser than 5 Å are flagged and excluded from those medians. V medians are
also reported on the full deposited cohort.

Output: ``outputs/cohort_summary/v_vs_emringer_summary.csv`` plus printed medians.

Example::

    source .venv/bin/activate
    python scripts/run_v_emringer_summary.py --resume
    python scripts/run_v_emringer_summary.py --emd-id 49450
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
from thesis.metric_comparison import load_all_metrics

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

OUTPUT_CSV = OUTPUTS_ROOT / "cohort_summary" / "v_vs_emringer_summary.csv"
MIN_FINITE = 30
MIN_PAIRS = 10


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
    p.add_argument(
        "--emd-id",
        type=str,
        default=None,
        help="Process one EMDB entry only (for spot checks)",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip EMDB IDs already present in the output CSV",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Show per-residue local-resolution aggregation warnings",
    )
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


SUMMARY_COLUMNS = [
    "emdb_id",
    "display_name",
    "pdb_code",
    "global_resolution_a",
    "emringer_interpretable",
    "building_regime_panel",
    "emringer_panel_reason",
    "emringer_csv",
    "rho",
    "p_value",
    "n_pairs",
    "n_v_in_mask",
    "n_emringer_in_mask",
    "v_variance",
    "emringer_variance",
    "nan_reason",
]


def _load_existing_rows(out: Path) -> dict[str, dict]:
    if not out.is_file():
        return {}
    df = pd.read_csv(out)
    return {str(row["emdb_id"]).strip(): row.to_dict() for _, row in df.iterrows()}


def _write_rows(out: Path, rows: list[dict]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows)[SUMMARY_COLUMNS].sort_values("rho", na_position="last").to_csv(
        out, index=False
    )


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


def _load_metrics(emdb_id: str, *, manifest: Path) -> pd.DataFrame:
    cached = emd_output_dir(emdb_id) / "metric_comparison" / "residue_metrics.csv"
    if cached.is_file():
        return pd.read_csv(cached)
    return load_all_metrics(emdb_id, manifest=manifest)


def _pdb_paths(manifest: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    with manifest.open(newline="") as f:
        for row in csv.DictReader(f):
            eid = str(row.get("emdb_id", "")).strip()
            pdb_raw = row.get("flexibility_path_or_pdb", "").strip()
            if eid and pdb_raw and Path(pdb_raw).is_file():
                paths[eid] = Path(pdb_raw)
    return paths


def _summarize_entry(
    emdb_id: str,
    *,
    manifest: Path,
    emringer_csv: Path,
    pdb_path: Path,
) -> dict | None:
    """One summary row, or ``None`` if the entry never enters the analysis panel."""
    try:
        df = attach_emringer_scores(
            _load_metrics(emdb_id, manifest=manifest),
            emringer_csv,
            pdb_path=pdb_path,
        )
    except (FileNotFoundError, ValueError, KeyError) as exc:
        logger.warning("skip EMD-%s: %s", emdb_id, exc)
        return None

    sub = df[df["in_contour_mask"].astype(bool)]
    n_v = int(sub["v_metric"].notna().sum())
    n_em = int(sub["emringer_score"].notna().sum())
    if n_v < MIN_FINITE or n_em < MIN_FINITE:
        return None

    paired = sub["v_metric"].notna() & sub["emringer_score"].notna()
    n_pairs = int(paired.sum())
    v = sub.loc[paired, "v_metric"].to_numpy(dtype=float)
    em = sub.loc[paired, "emringer_score"].to_numpy(dtype=float)

    v_var = float(np.var(v)) if n_pairs else float("nan")
    em_var = float(np.var(em)) if n_pairs else float("nan")

    rho = float("nan")
    pval = float("nan")
    nan_reason = ""
    if n_pairs < MIN_PAIRS:
        nan_reason = f"n_pairs<{MIN_PAIRS}"
    elif v_var == 0.0 and em_var == 0.0:
        nan_reason = "zero_v_and_emringer_variance"
    elif v_var == 0.0:
        nan_reason = "zero_v_variance"
    elif em_var == 0.0:
        nan_reason = "zero_emringer_variance"
    else:
        r, p = stats.spearmanr(v, em)
        rho, pval = float(r), float(p)
        if not np.isfinite(rho):
            nan_reason = "spearman_undefined"

    return {
        "emdb_id": emdb_id,
        "rho": rho,
        "p_value": pval,
        "n_pairs": n_pairs,
        "n_v_in_mask": n_v,
        "n_emringer_in_mask": n_em,
        "v_variance": v_var,
        "emringer_variance": em_var,
        "nan_reason": nan_reason,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _configure_logging(verbose=args.verbose)

    missing = missing_emringer_csvs(args.manifest, args.emringer_dir)
    if missing:
        ids = ", ".join(f"EMD-{r.emdb_id} ({r.pdb_code})" for r in missing)
        msg = f"[v_emringer] missing flat EMRinger CSV for {len(missing)} deposits: {ids}"
        if args.require_all:
            print(msg, file=sys.stderr, flush=True)
            print(
                "[v_emringer] run: python scripts/run_emringer_cohort_batch.py",
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
        f"[v_emringer] lookup: {len(lookup)} manifest rows with flat EMRinger CSV "
        f"under {args.emringer_dir}",
        flush=True,
    )

    existing = _load_existing_rows(args.out) if args.resume else {}
    rows: list[dict] = list(existing.values())

    for emdb_id, emringer_csv in sorted(lookup.items(), key=lambda kv: kv[0]):
        if args.emd_id and emdb_id != args.emd_id.strip():
            continue
        if args.resume and emdb_id in existing:
            print(f"[v_emringer] EMD-{emdb_id}: resume skip", flush=True)
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
            f"[v_emringer] EMD-{emdb_id} ({summary['pdb_code']}, "
            f"{summary['global_resolution_a']:.2f} Å, {panel}): rho={summary['rho']:+.3f} "
            f"n={summary['n_pairs']} {summary['nan_reason']}",
            flush=True,
        )

    if not rows:
        print("[v_emringer] no eligible entries", file=sys.stderr)
        return 1

    out_df = pd.DataFrame(rows)[SUMMARY_COLUMNS].sort_values("rho", na_position="last")
    full = out_df["rho"].dropna()
    panel = out_df[out_df["emringer_interpretable"].astype(bool)]["rho"].dropna()
    building = out_df[out_df["building_regime_panel"].astype(bool)]["rho"].dropna()
    print(
        f"[v_emringer] V vs EMRinger (full {len(full)} maps): median rho = "
        f"{float(full.median()) if len(full) else float('nan'):+.3f}",
        flush=True,
    )
    print(
        f"[v_emringer] EMRinger panel (≤{args.max_resolution_a:g} Å, "
        f"{EMRINGER_BARAD_2015_CITATION}; n={len(panel)}): "
        f"median rho = {float(panel.median()) if len(panel) else float('nan'):+.3f}",
        flush=True,
    )
    print(
        f"[v_emringer] building regime "
        f"({BUILDING_REGIME_MIN_RESOLUTION_A:g}–{BUILDING_REGIME_MAX_RESOLUTION_A:g} Å, "
        f"n={len(building)}): "
        f"median rho = {float(building.median()) if len(building) else float('nan'):+.3f}",
        flush=True,
    )
    excluded = out_df[~out_df["emringer_interpretable"].astype(bool)]
    if len(excluded):
        ids = ", ".join(str(x) for x in excluded["emdb_id"].astype(str).tolist())
        print(
            f"[v_emringer] excluded (coarser than {args.max_resolution_a:g} Å): {ids}",
            flush=True,
        )
    print(f"[v_emringer] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
