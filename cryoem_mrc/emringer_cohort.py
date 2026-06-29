"""EMRinger validation cohort: resolution eligibility and manifest helpers.

Barad et al. (2015) *Nat Methods* 12, 943–946; doi:10.1038/nmeth.3541:

- Side-chain rotamer validation is interpretable in the roughly **3–5 Å** regime
  where χ1 scans resolve rotameric density.
- The metric **breaks down coarser than ~5 Å**; those maps are excluded from EMRinger
  comparison medians, not treated as validation failures.

Thesis convention: headline EMRinger panels use global resolution ≤5 Å. The cohort's
primary model-building band (2.5–4 Å global resolution) is flagged separately for
stratified summaries.
"""

from __future__ import annotations

import csv
from pathlib import Path

from .repo_paths import COHORT_MANIFEST

# Barad et al. 2015: breakdown coarser than ~5 Å.
EMRINGER_INTERPRETABLE_MAX_RESOLUTION_A: float = 5.0

# Cohort model-building regime (global resolution, Å).
BUILDING_REGIME_MIN_RESOLUTION_A: float = 2.5
BUILDING_REGIME_MAX_RESOLUTION_A: float = 4.0

EMRINGER_BARAD_2015_CITATION = (
    "Barad et al., Nat Methods 12, 943–946 (2015); doi:10.1038/nmeth.3541"
)


def load_manifest_global_resolution_a(
    manifest: Path = COHORT_MANIFEST,
) -> dict[str, float]:
    """``emdb_id`` → deposited global resolution (Å); missing/invalid → NaN."""
    out: dict[str, float] = {}
    with manifest.open(newline="") as f:
        for row in csv.DictReader(f):
            eid = str(row.get("emdb_id", "")).strip()
            if not eid:
                continue
            raw = str(row.get("global_resolution_a", "")).strip()
            try:
                out[eid] = float(raw)
            except ValueError:
                out[eid] = float("nan")
    return out


def emringer_interpretable(
    emdb_id: str,
    *,
    resolutions: dict[str, float] | None = None,
    max_resolution_a: float = EMRINGER_INTERPRETABLE_MAX_RESOLUTION_A,
) -> bool:
    """True when global resolution is finite and ≤ ``max_resolution_a`` (Å)."""
    import math

    res_map = resolutions if resolutions is not None else load_manifest_global_resolution_a()
    res = res_map.get(str(emdb_id).strip(), float("nan"))
    if not math.isfinite(res):
        return False
    return res <= max_resolution_a


def building_regime_panel(
    emdb_id: str,
    *,
    resolutions: dict[str, float] | None = None,
    min_resolution_a: float = BUILDING_REGIME_MIN_RESOLUTION_A,
    max_resolution_a: float = BUILDING_REGIME_MAX_RESOLUTION_A,
) -> bool:
    """True when global resolution falls in the model-building band [min, max] Å."""
    import math

    res_map = resolutions if resolutions is not None else load_manifest_global_resolution_a()
    res = res_map.get(str(emdb_id).strip(), float("nan"))
    if not math.isfinite(res):
        return False
    return min_resolution_a <= res <= max_resolution_a


def emringer_panel_reason(
    emdb_id: str,
    *,
    resolutions: dict[str, float] | None = None,
    max_resolution_a: float = EMRINGER_INTERPRETABLE_MAX_RESOLUTION_A,
) -> str:
    """Empty when interpretable; otherwise a documented exclusion reason."""
    if emringer_interpretable(
        emdb_id, resolutions=resolutions, max_resolution_a=max_resolution_a
    ):
        return ""
    import math

    res_map = resolutions if resolutions is not None else load_manifest_global_resolution_a()
    res = res_map.get(str(emdb_id).strip(), float("nan"))
    if not math.isfinite(res):
        return "missing_global_resolution"
    return f"resolution_coarser_than_{max_resolution_a:g}A_barad2015"
