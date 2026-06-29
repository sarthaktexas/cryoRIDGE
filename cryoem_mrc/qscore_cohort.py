"""Q-score validation cohort membership and pre-specified exclusion rules."""

from __future__ import annotations

from pathlib import Path

from .repo_paths import COHORT_MANIFEST, OUTPUTS_ROOT
from .manifest_policy import QSCORE_HARD_EXCLUDE_EMDB_IDS

# Hard exclusions for any Q-score panel (core or expansion).
QSCORE_PANEL_EXCLUDE: frozenset[str] = QSCORE_HARD_EXCLUDE_EMDB_IDS

# Additional exclusions for the primary "core" Q-score cohort (thesis headline n≈24).
# Document each in methods; full panel retained for sensitivity tables.
QSCORE_CORE_EXCLUDE: frozenset[str] = frozenset(
    {
        "4940",  # ClpB WT-1, 6.2 Å — no Q–V coupling
        "9156",  # NMDA Extended-1, 6.8 Å — no Q–V coupling
        "11149",  # ATP synthase Fo — 60 in-mask Cα only
        "24120",  # 70S pre-transloc — reliability/CC decoupled, ρ≈0
        "28498",  # Eag1 down, 5.4 Å — weak all proxies
        "4941",  # ClpB WT-2A — decoupled (supplement / pair with 4940)
        "52525",  # Complex III — ~10% contour mask coverage
        "13308",  # GroEL-GroES tight — saturated CC, V-specific failure (pair: 16119)
    }
)

QSCORE_CORE_MAX_RESOLUTION_A: float = 5.0

QSCORE_CORE_MIN_IN_MASK: int = 100

CORE_COHORT_SUMMARY_STEM = "core"


def qscore_exclude_ids(*, core: bool = False) -> frozenset[str]:
    """Return EMDB IDs to omit from a Q-score cohort iteration."""
    if core:
        return QSCORE_PANEL_EXCLUDE | QSCORE_CORE_EXCLUDE
    return QSCORE_PANEL_EXCLUDE


def filter_emdb_ids(ids: list[str] | tuple[str, ...], *, core: bool = False) -> list[str]:
    """Drop excluded IDs; when ``core``, also drop global resolution ≥ cutoff from manifest."""
    exclude = qscore_exclude_ids(core=core)
    out = [str(eid).strip() for eid in ids if str(eid).strip() not in exclude]
    if not core:
        return out
    return _filter_by_manifest_resolution(out, max_resolution_a=QSCORE_CORE_MAX_RESOLUTION_A)


def _filter_by_manifest_resolution(
    ids: list[str],
    *,
    max_resolution_a: float,
    manifest: Path = COHORT_MANIFEST,
) -> list[str]:
    if not manifest.is_file():
        return ids
    import csv

    res_by_id: dict[str, float] = {}
    with manifest.open(newline="") as f:
        for row in csv.DictReader(f):
            eid = str(row.get("emdb_id", "")).strip()
            raw = str(row.get("global_resolution_a", "")).strip()
            try:
                res_by_id[eid] = float(raw)
            except ValueError:
                res_by_id[eid] = float("nan")
    out: list[str] = []
    for eid in ids:
        res = res_by_id.get(eid, float("nan"))
        if res >= max_resolution_a:
            continue
        out.append(eid)
    return out


def iter_qscore_emdb_ids(
    *,
    manifest: Path = COHORT_MANIFEST,
    core: bool = False,
    outputs_root: Path | None = None,
) -> list[str]:
    """List EMDB IDs with ``qscore_validation.csv``, applying exclusion rules."""
    from .incremental_prediction import iter_eligible_emdb_ids

    root = outputs_root or OUTPUTS_ROOT
    ids = iter_eligible_emdb_ids("q_score", manifest=manifest, outputs_root=root, qscore_exclude=frozenset())
    return filter_emdb_ids(ids, core=core)


def core_cohort_output_path(base_name: str, *, out_dir: Path | None = None) -> Path:
    """``qscore_correlations.csv`` → ``qscore_correlations_core.csv``."""
    root = out_dir or (OUTPUTS_ROOT / "cohort_summary")
    stem, suffix = base_name.rsplit(".", 1) if "." in base_name else (base_name, "csv")
    return root / f"{stem}_{CORE_COHORT_SUMMARY_STEM}.{suffix}"
