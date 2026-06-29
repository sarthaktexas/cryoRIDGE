"""Build filtered core Q-score cohort summaries from existing cohort CSVs.

Applies :data:`cryoem_mrc.qscore_cohort.QSCORE_CORE_EXCLUDE` plus resolution
cutoff (``QSCORE_CORE_MAX_RESOLUTION_A``) without re-running map pipelines.

Example::

    source .venv/bin/activate
    python scripts/run_qscore_core_cohort_summary.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np
import pandas as pd

from cryoem_mrc.cohort_labels import load_display_name_map
from cryoem_mrc.qscore_cohort import (
    QSCORE_CORE_EXCLUDE,
    QSCORE_CORE_MAX_RESOLUTION_A,
    QSCORE_PANEL_EXCLUDE,
    core_cohort_output_path,
    filter_emdb_ids,
    qscore_exclude_ids,
)
from cryoem_mrc.repo_paths import COHORT_MANIFEST, OUTPUTS_ROOT

COHORT_DIR = OUTPUTS_ROOT / "cohort_summary"

SOURCE_TABLES: tuple[str, ...] = (
    "qscore_correlations.csv",
    "placement_rank_recovery.csv",
    "q_vs_locres_summary_both.csv",
    "qscore_complementarity_per_map.csv",
    "qscore_complementarity_lomo.csv",
    "qscore_hessian_ablation.csv",
    "qscore_gradient_ablation.csv",
    "placement_decoupling_cohort.csv",
)

OUT_MD = COHORT_DIR / "qscore_core_cohort_summary.md"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, default=COHORT_MANIFEST)
    p.add_argument("--in-dir", type=Path, default=COHORT_DIR)
    return p.parse_args(argv)


def _load_table(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    df = pd.read_csv(path)
    if "emdb_id" not in df.columns:
        return None
    df["emdb_id"] = df["emdb_id"].astype(str).str.strip()
    return df


def _filter_df(df: pd.DataFrame, keep_ids: set[str]) -> pd.DataFrame:
    return df[df["emdb_id"].isin(keep_ids)].copy()


def _median_col(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns:
        return float("nan")
    v = pd.to_numeric(df[col], errors="coerce")
    v = v[np.isfinite(v)]
    return float(v.median()) if len(v) else float("nan")


def _write_summary_md(
    path: Path,
    *,
    full_n: int,
    core_n: int,
    keep_ids: list[str],
    names: dict[str, str],
    full_q: pd.DataFrame,
    core_q: pd.DataFrame,
    full_rr: pd.DataFrame | None,
    core_rr: pd.DataFrame | None,
    full_qloc: pd.DataFrame | None,
    core_qloc: pd.DataFrame | None,
) -> None:
    lines = [
        "# Q-score core cohort summary",
        "",
        "Pre-specified exclusions for thesis headline statistics. Full panel tables",
        "are retained as sensitivity analyses (`*_full` / unfiltered CSVs).",
        "",
        "## Inclusion rules",
        "",
        f"- Start from Q-score panel (exclude panel IDs: {sorted(QSCORE_PANEL_EXCLUDE)})",
        f"- Core exclude: {sorted(QSCORE_CORE_EXCLUDE)}",
        f"- Global resolution < {QSCORE_CORE_MAX_RESOLUTION_A:g} Å (manifest)",
        "",
        f"**Maps:** {core_n} core / {full_n} full panel",
        "",
        "## Median ρ(Q, V)",
        "",
        f"- Full panel: **{_median_col(full_q, 'spearman_q_vs_V'):+.3f}** (n={len(full_q)})",
        f"- Core cohort: **{_median_col(core_q, 'spearman_q_vs_V'):+.3f}** (n={len(core_q)})",
        "",
    ]
    if full_rr is not None and core_rr is not None and "spearman_q_vs_v" in core_rr.columns:
        lines.extend(
            [
                "## Median |ρ(Q, ·)| (placement rank recovery)",
                "",
                f"- |ρ(Q, V)| full: **{_median_col(full_rr, 'spearman_q_vs_v'):.3f}**",
                f"- |ρ(Q, V)| core: **{pd.to_numeric(core_rr['spearman_q_vs_v'], errors='coerce').abs().median():.3f}**",
                f"- |ρ(Q, CC)| core: **{pd.to_numeric(core_rr['spearman_q_vs_cc'], errors='coerce').abs().median():.3f}**",
                "",
            ]
        )
    if full_qloc is not None and core_qloc is not None:
        full_r = pd.to_numeric(full_qloc.get("rho_Q_resmap"), errors="coerce").abs()
        core_r = pd.to_numeric(core_qloc.get("rho_Q_resmap"), errors="coerce").abs()
        lines.extend(
            [
                "## Median |ρ(Q, ResMap)|",
                "",
                f"- Full: **{float(full_r.median()) if full_r.notna().any() else float('nan'):.3f}**",
                f"- Core: **{float(core_r.median()) if core_r.notna().any() else float('nan'):.3f}**",
                "",
            ]
        )
    lines.extend(["## Core cohort members", "", "| EMDB | Name |", "|------|------|"])
    for eid in sorted(keep_ids, key=lambda x: int(x)):
        lines.append(f"| {eid} | {names.get(eid, '')} |")
    lines.append("")
    path.write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    names = load_display_name_map(args.manifest)

    q_full = _load_table(args.in_dir / "qscore_correlations.csv")
    if q_full is None or q_full.empty:
        print("[core_cohort] missing qscore_correlations.csv", file=sys.stderr)
        return 1

    all_ids = q_full["emdb_id"].tolist()
    core_ids = filter_emdb_ids(all_ids, core=True)
    keep = set(core_ids)

    print(f"[core_cohort] full panel n={len(all_ids)} → core n={len(core_ids)}", flush=True)
    print(f"[core_cohort] excluded from core: {sorted(set(all_ids) - keep)}", flush=True)

    written: list[Path] = []
    for name in SOURCE_TABLES:
        src = args.in_dir / name
        df = _load_table(src)
        if df is None:
            continue
        out = core_cohort_output_path(name, out_dir=args.in_dir)
        filt = _filter_df(df, keep)
        filt.to_csv(out, index=False)
        written.append(out)
        print(f"[core_cohort] {name}: {len(df)} → {len(filt)} rows → {out.name}", flush=True)

    core_q = _filter_df(q_full, keep)
    full_rr = _load_table(args.in_dir / "placement_rank_recovery.csv")
    core_rr = _filter_df(full_rr, keep) if full_rr is not None else None
    full_qloc = _load_table(args.in_dir / "q_vs_locres_summary_both.csv")
    core_qloc = _filter_df(full_qloc, keep) if full_qloc is not None else None

    _write_summary_md(
        OUT_MD,
        full_n=len(all_ids),
        core_n=len(core_ids),
        keep_ids=core_ids,
        names=names,
        full_q=q_full,
        core_q=core_q,
        full_rr=full_rr,
        core_rr=core_rr,
        full_qloc=full_qloc,
        core_qloc=core_qloc,
    )
    print(f"[core_cohort] summary → {OUT_MD}", flush=True)

    if core_rr is not None and core_qloc is not None:
        v_med = pd.to_numeric(core_rr["spearman_q_vs_v"], errors="coerce").abs().median()
        r_med = pd.to_numeric(core_qloc["rho_Q_resmap"], errors="coerce").abs().median()
        print(f"\n=== Core cohort headline ===")
        print(f"  median |ρ(Q,V)|     = {v_med:.3f}")
        print(f"  median |ρ(Q,ResMap)| = {r_med:.3f}")
        print(f"  median ρ(Q,V) signed = {_median_col(core_q, 'spearman_q_vs_V'):+.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
