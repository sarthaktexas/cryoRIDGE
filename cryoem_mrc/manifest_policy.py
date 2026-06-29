"""Which cohort manifest rows support Cα / Q-score vs map-only (V, CC, locres) analysis."""

from __future__ import annotations

import csv
from pathlib import Path

# RNA-only model; NPC STA far below Q-score resolution band.
QSCORE_HARD_EXCLUDE_EMDB_IDS = frozenset(
    {
        "33736",  # tetrahymena ribozyme — zero protein Cα
        "52153",  # mouse NPC cytoplasmic ring STA ~29 Å — V robustness check only
    }
)

MAPS_ONLY_SOURCES = frozenset({"windowed_halfmap_correlation_only", "excluded"})

# ResMap sentinel-100 Å class (partial megacomplex / consensus maps). Still queue on Arc
# to document failure; omit from headline Cα ResMap panels unless explicitly included.
RESMAP_EXPECTED_FAILURE_VALUES = frozenset({"document", "yes", "true"})


def row_has_deposited_pdb(row: dict[str, str]) -> bool:
    raw = str(row.get("flexibility_path_or_pdb", "")).strip()
    return bool(raw) and Path(raw).is_file()


def row_uses_maps_only_metrics(row: dict[str, str]) -> bool:
    if str(row.get("model_metrics", "")).strip().lower() == "maps_only":
        return True
    return str(row.get("flexibility_source", "")).strip() in MAPS_ONLY_SOURCES


def row_resmap_expected_failure(row: dict[str, str]) -> bool:
    """True when ResMap 100 Å sentinel / flat output is expected (document, not headline)."""
    return str(row.get("resmap_expected_failure", "")).strip().lower() in RESMAP_EXPECTED_FAILURE_VALUES


def row_resmap_submit_recommended(row: dict[str, str]) -> bool:
    """Whether to queue ResMap on Arc (``skip`` = do not submit)."""
    return str(row.get("resmap_expected_failure", "")).strip().lower() != "skip"


def row_qscore_eligible(row: dict[str, str], *, include_borderline: bool = True) -> bool:
    """Residue-level Q-score / EMRinger cohort membership."""
    eid = str(row.get("emdb_id", "")).strip()
    if eid in QSCORE_HARD_EXCLUDE_EMDB_IDS:
        return False
    flag = str(row.get("qscore_eligible", "")).strip().lower()
    if flag == "no":
        return False
    if flag == "borderline" and not include_borderline:
        return False
    if row_uses_maps_only_metrics(row):
        return False
    return row_has_deposited_pdb(row)


def row_ca_metrics_eligible(row: dict[str, str]) -> bool:
    """Cα ResMap/BlocRes vs V, metric comparison, B-factor validation."""
    if row_uses_maps_only_metrics(row):
        return False
    return row_has_deposited_pdb(row)


def row_resmap_ca_headline_eligible(row: dict[str, str]) -> bool:
    """Cα ResMap comparisons for cohort medians (excludes expected-failure documentary maps)."""
    if row_resmap_expected_failure(row):
        return False
    return row_ca_metrics_eligible(row)


def load_manifest_policy_by_emdb(manifest: Path) -> dict[str, dict[str, str]]:
    """``emdb_id`` → manifest policy columns used for cohort stratification."""
    out: dict[str, dict[str, str]] = {}
    if not manifest.is_file():
        return out
    with manifest.open(newline="") as f:
        for row in csv.DictReader(f):
            eid = str(row.get("emdb_id", "")).strip()
            if not eid:
                continue
            out[eid] = dict(row)
    return out


def cohort_tag_for_manifest(manifest: Path) -> str:
    """``core`` vs ``expansion`` label from manifest path or filename."""
    resolved = manifest.resolve()
    name = resolved.name.lower()
    if "expansion" in name:
        return "expansion"
    return "core"
