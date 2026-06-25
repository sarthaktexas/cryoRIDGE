"""Conformation-pair helpers: coverage and visualization alignment."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from .structure_validation import ResidueValidationRow

COVERAGE_FLAG_THRESHOLD_PCT = 20.0
_COHORT_DIR = Path(__file__).resolve().parent.parent / "cohort"
CONFORMATION_PAIR_DOMAINS_PATH = _COHORT_DIR / "conformation_pair_domains.json"
TRPV1_EMDB_IDS = frozenset({"23129", "23130"})
MGTA_EMDB_IDS = frozenset({"49450", "48534", "48923"})
_DOMAINS_REGISTRY_CACHE: list[dict] | None = None


@dataclass
class ConformationPairCoverage:
    """How completely matched in-mask Cα cover each deposited model."""

    emdb_a: str
    emdb_b: str
    n_ca_total_a: int
    n_ca_total_b: int
    n_matched: int
    n_matched_in_mask_both: int
    n_matched_in_mask_a: int
    n_matched_in_mask_b: int
    frac_analysis_of_a: float
    frac_analysis_of_b: float
    missing_pct_a: float
    missing_pct_b: float
    coverage_flag: bool
    notes: str = ""


def compute_conformation_pair_coverage(
    pairs: Sequence[tuple[ResidueValidationRow, ResidueValidationRow]],
    *,
    emdb_a: str,
    emdb_b: str,
    n_ca_total_a: int,
    n_ca_total_b: int,
    threshold_pct: float = COVERAGE_FLAG_THRESHOLD_PCT,
) -> ConformationPairCoverage:
    """Compare analysis residue count to full deposited Cα count per model."""
    n_matched = len(pairs)
    n_mask_a = sum(1 for a, _ in pairs if a.in_contour_mask)
    n_mask_b = sum(1 for _, b in pairs if b.in_contour_mask)
    n_both = sum(1 for a, b in pairs if a.in_contour_mask and b.in_contour_mask)

    def _frac(n: int, total: int) -> float:
        return float(n / total) if total > 0 else float("nan")

    frac_a = _frac(n_both, n_ca_total_a)
    frac_b = _frac(n_both, n_ca_total_b)
    miss_a = 100.0 * (1.0 - frac_a) if np.isfinite(frac_a) else float("nan")
    miss_b = 100.0 * (1.0 - frac_b) if np.isfinite(frac_b) else float("nan")
    flagged = (np.isfinite(miss_a) and miss_a > threshold_pct) or (
        np.isfinite(miss_b) and miss_b > threshold_pct
    )

    notes = ""
    if flagged:
        notes = (
            f"> {threshold_pct:.0f}% of deposited Cα are outside the analysis set "
            f"(unmatched and/or below contour). Interpret coupling maps with this gap in mind."
        )

    return ConformationPairCoverage(
        emdb_a=emdb_a,
        emdb_b=emdb_b,
        n_ca_total_a=n_ca_total_a,
        n_ca_total_b=n_ca_total_b,
        n_matched=n_matched,
        n_matched_in_mask_both=n_both,
        n_matched_in_mask_a=n_mask_a,
        n_matched_in_mask_b=n_mask_b,
        frac_analysis_of_a=frac_a,
        frac_analysis_of_b=frac_b,
        missing_pct_a=miss_a,
        missing_pct_b=miss_b,
        coverage_flag=flagged,
        notes=notes,
    )


def interior_residue_indices(n: int, half_window: int) -> np.ndarray:
    """Residue indices with full ±half_window coupling windows (no edge NaNs)."""
    if n <= 2 * half_window:
        return np.arange(n, dtype=int)
    return np.arange(half_window, n - half_window, dtype=int)


def kabsch_align_coords(
    mobile: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rigid-body alignment of ``mobile`` onto ``target`` (N×3 Cα coordinates).

    Used for per-residue Cα RMSD (state B aligned onto state A) and ChimeraX overlays.
    """
    mobile = np.asarray(mobile, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if mobile.shape != target.shape or mobile.shape[0] < 3:
        return mobile.copy(), np.eye(3, dtype=np.float64)
    mob_ctr = mobile.mean(axis=0)
    tgt_ctr = target.mean(axis=0)
    rot, _ = Rotation.align_vectors(mobile - mob_ctr, target - tgt_ctr)
    aligned = rot.apply(mobile - mob_ctr) + tgt_ctr
    return aligned, rot.as_matrix()


def compute_per_residue_ca_rmsd(
    pairs: Sequence[tuple[ResidueValidationRow, ResidueValidationRow]],
    *,
    in_mask_both: bool = True,
) -> tuple[list[tuple[ResidueValidationRow, ResidueValidationRow]], np.ndarray]:
    """
    Per-residue Cα displacement (Å) after Kabsch alignment of state B onto state A.

    Matched residues are sorted by mmCIF chain order.
    """
    use = list(pairs)
    if in_mask_both:
        use = [
            (a, b)
            for a, b in pairs
            if a.in_contour_mask
            and b.in_contour_mask
            and np.isfinite(a.x)
            and np.isfinite(b.x)
        ]
    if len(use) < 3:
        return use, np.array([], dtype=np.float64)
    use.sort(key=lambda ab: (ab[0].chain, ab[0].seq_num, ab[0].seq_icode))
    coords_a = np.array([[a.x, a.y, a.z] for a, _ in use], dtype=np.float64)
    coords_b = np.array([[b.x, b.y, b.z] for _, b in use], dtype=np.float64)
    coords_b_aligned, _ = kabsch_align_coords(coords_b, coords_a)
    rmsd = np.linalg.norm(coords_b_aligned - coords_a, axis=1)
    return use, rmsd


@dataclass
class ConformationPairStats:
    """Per-residue Cα RMSD vs Δreliability on matched deposited models."""

    emdb_a: str
    emdb_b: str
    n_matched: int
    n_matched_in_mask_both: int
    spearman_rmsd_vs_delta_reliability: float
    median_ca_rmsd_a: float


def compute_conformation_pair_stats(
    pairs: Sequence[tuple[ResidueValidationRow, ResidueValidationRow]],
    *,
    emdb_a: str,
    emdb_b: str,
    in_mask_both: bool = True,
) -> ConformationPairStats:
    from scipy import stats

    use, rmsd = compute_per_residue_ca_rmsd(pairs, in_mask_both=in_mask_both)
    n_match = len(pairs)
    n_use = len(use)
    if n_use < 10:
        return ConformationPairStats(
            emdb_a=emdb_a,
            emdb_b=emdb_b,
            n_matched=n_match,
            n_matched_in_mask_both=n_use,
            spearman_rmsd_vs_delta_reliability=float("nan"),
            median_ca_rmsd_a=float("nan"),
        )
    drel = np.array([b.reliability_score - a.reliability_score for a, b in use], dtype=np.float64)
    r_rel, _ = stats.spearmanr(rmsd, drel)
    return ConformationPairStats(
        emdb_a=emdb_a,
        emdb_b=emdb_b,
        n_matched=n_match,
        n_matched_in_mask_both=n_use,
        spearman_rmsd_vs_delta_reliability=float(r_rel),
        median_ca_rmsd_a=float(np.median(rmsd)),
    )


def compute_rmsd_superposition_diagnostics(
    use: Sequence[tuple[ResidueValidationRow, ResidueValidationRow]],
    *,
    rmsd_global: np.ndarray,
    emdb_a: str | None = None,
    emdb_b: str | None = None,
) -> dict[str, object]:
    """Registration sanity checks for Table 3 median RMSD interpretation.

    Global Kabsch on all matched Cα can inflate medians when subunits move as
    independent rigid bodies (70S, GroEL) or when deposited models sit in
    different map origins. Per-chain medians and domain medians help distinguish
    biology from superposition artifacts.
    """
    if len(use) < 3:
        return {}

    coords_a = np.array([[a.x, a.y, a.z] for a, _ in use], dtype=np.float64)
    coords_b = np.array([[b.x, b.y, b.z] for _, b in use], dtype=np.float64)
    centroid_sep = float(np.linalg.norm(coords_a.mean(axis=0) - coords_b.mean(axis=0)))

    by_chain: dict[str, list[int]] = {}
    for i, (row, _) in enumerate(use):
        by_chain.setdefault(row.chain, []).append(i)

    per_chain_medians: dict[str, float] = {}
    for chain, idx in sorted(by_chain.items()):
        if len(idx) < 3:
            continue
        ca = coords_a[idx]
        cb = coords_b[idx]
        cb_aligned, _ = kabsch_align_coords(cb, ca)
        chain_rmsd = np.linalg.norm(cb_aligned - ca, axis=1)
        per_chain_medians[chain] = float(np.median(chain_rmsd))

    median_global = float(np.median(rmsd_global))
    median_per_chain = (
        float(np.median(list(per_chain_medians.values()))) if per_chain_medians else float("nan")
    )

    domain_medians: dict[str, float] = {}
    if emdb_a is not None and emdb_b is not None:
        regions = get_domain_regions_for_pair(emdb_a, emdb_b)
        if regions:
            assignments = get_domain_assignments(use, regions)
            for name, idx in assignments.items():
                if len(idx) >= 3:
                    domain_medians[name] = float(np.median(rmsd_global[idx]))

    note_parts: list[str] = []
    if centroid_sep > 50.0:
        note_parts.append("large pre-alignment centroid offset between deposited models")
    if (
        np.isfinite(median_per_chain)
        and median_global > 25.0
        and median_per_chain < 0.5 * median_global
    ):
        note_parts.append(
            "global superposition limited (per-chain median much lower than global)"
        )
    if len(by_chain) >= 2 and per_chain_medians:
        spread = max(per_chain_medians.values()) - min(per_chain_medians.values())
        if spread > 30.0:
            note_parts.append("heterogeneous per-chain RMSD (asymmetric oligomer or ring motion)")

    return {
        "centroid_separation_angstrom": centroid_sep,
        "median_ca_rmsd_global": median_global,
        "median_ca_rmsd_per_chain_median": median_per_chain,
        "median_ca_rmsd_by_chain": per_chain_medians,
        "median_ca_rmsd_by_domain": domain_medians,
        "rmsd_registration_note": "; ".join(note_parts) if note_parts else "",
    }


def write_conformation_pair_md(
    path: Path,
    pair_stats: ConformationPairStats,
    coverage: ConformationPairCoverage | None = None,
) -> None:
    cov_block = ""
    if coverage is not None:
        flag = " **YES — discuss in thesis**" if coverage.coverage_flag else " no"
        cov_block = f"""
## Coverage vs deposited model

| Metric | State A (EMD-{coverage.emdb_a}) | State B (EMD-{coverage.emdb_b}) |
|--------|--------------------------------:|--------------------------------:|
| Deposited Cα total | {coverage.n_ca_total_a:,} | {coverage.n_ca_total_b:,} |
| Matched (any mask) | {coverage.n_matched:,} | {coverage.n_matched:,} |
| Both in contour mask | {coverage.n_matched_in_mask_both:,} | {coverage.n_matched_in_mask_both:,} |
| Analysis / deposited Cα | {100 * coverage.frac_analysis_of_a:.1f}% | {100 * coverage.frac_analysis_of_b:.1f}% |
| Missing from analysis | {coverage.missing_pct_a:.1f}% | {coverage.missing_pct_b:.1f}% |

Flag (>20% missing):{flag}

{coverage.notes}
"""
    text = f"""# Conformation pair — EMD-{pair_stats.emdb_a} vs EMD-{pair_stats.emdb_b}

Matched Cα by mmCIF (label_asym_id, label_seq_id, insertion). State B is Kabsch-aligned onto
state A for per-residue Cα RMSD. Δreliability = reliability(B) − reliability(A).

| Metric | Value |
|--------|------:|
| Matched residues | {pair_stats.n_matched:,} |
| Both in contour mask | {pair_stats.n_matched_in_mask_both:,} |
| Median Cα RMSD (Å) | {pair_stats.median_ca_rmsd_a:.2f} |
| Spearman ρ(RMSD, Δreliability) | {pair_stats.spearman_rmsd_vs_delta_reliability:+.3f} |
{cov_block}
See `docs/CONFORMATION_PAIR_ANALYSIS.md` for coupling maps and figure outputs.
"""
    path.write_text(text)


@dataclass(frozen=True)
class DomainRegion:
    """Sequence-based fold band for heatmap annotation (auth seq_id)."""

    name: str
    seq_start: int
    seq_end: int
    color: str
    chains: frozenset[str] | None = None
    chain_prefixes: tuple[str, ...] | None = None


def is_trpv1_conformation_pair(emdb_a: str, emdb_b: str) -> bool:
    return {str(emdb_a).strip(), str(emdb_b).strip()} <= TRPV1_EMDB_IDS


def is_mgta_conformation_pair(emdb_a: str, emdb_b: str) -> bool:
    return {str(emdb_a).strip(), str(emdb_b).strip()} <= MGTA_EMDB_IDS


def _parse_domain_region(raw: dict) -> DomainRegion:
    chains = raw.get("chains")
    chain_prefixes = raw.get("chain_prefixes")
    return DomainRegion(
        name=str(raw["name"]),
        seq_start=int(raw["seq_start"]),
        seq_end=int(raw["seq_end"]),
        color=str(raw["color"]),
        chains=frozenset(str(c) for c in chains) if chains else None,
        chain_prefixes=tuple(str(p) for p in chain_prefixes) if chain_prefixes else None,
    )


def load_domain_regions_from_json(path: Path) -> list[DomainRegion]:
    raw = json.loads(path.read_text())
    return [_parse_domain_region(r) for r in raw["regions"]]


def _load_domains_registry() -> list[dict]:
    global _DOMAINS_REGISTRY_CACHE
    if _DOMAINS_REGISTRY_CACHE is None:
        if not CONFORMATION_PAIR_DOMAINS_PATH.is_file():
            _DOMAINS_REGISTRY_CACHE = []
        else:
            data = json.loads(CONFORMATION_PAIR_DOMAINS_PATH.read_text())
            _DOMAINS_REGISTRY_CACHE = list(data.get("entries", []))
    return _DOMAINS_REGISTRY_CACHE


def region_matches_residue(reg: DomainRegion, row: ResidueValidationRow) -> bool:
    """True when auth seq_id (and optional auth chain filter) falls in ``reg``."""
    chain = str(row.auth_chain or row.chain).strip()
    if reg.chains is not None and chain not in reg.chains:
        return False
    if reg.chain_prefixes is not None and not any(chain.startswith(p) for p in reg.chain_prefixes):
        return False
    seq_num = int(row.auth_seq_num or row.seq_num)
    return reg.seq_start <= seq_num <= reg.seq_end


def _domain_regions_for_registry_id(entry_id: str) -> list[DomainRegion]:
    for entry in _load_domains_registry():
        if str(entry.get("id", "")).strip() == entry_id:
            return [_parse_domain_region(r) for r in entry.get("regions", [])]
    return []


def load_trpv1_domain_regions() -> list[DomainRegion]:
    """Rat TRPV1 domain bands for EMD-23129/23130 (PDB 7L2I/7L2J auth numbering)."""
    return _domain_regions_for_registry_id("trpv1")


def load_mgta_domain_regions() -> list[DomainRegion]:
    """L. lactis MgtA domain bands for EMD-49450/48923/48534 (PDB 9NHZ/9N5J/9MQM)."""
    return _domain_regions_for_registry_id("mgta")


def domain_colors_from_regions(regions: Sequence[DomainRegion]) -> dict[str, str]:
    """Domain name → hex color (single source for figure panels)."""
    return {reg.name: reg.color for reg in regions}


def _merged_domain_colors() -> dict[str, str]:
    colors: dict[str, str] = {}
    for entry in _load_domains_registry():
        for raw in entry.get("regions", []):
            colors[str(raw["name"])] = str(raw["color"])
    return colors


DOMAIN_COLORS: dict[str, str] = _merged_domain_colors()
UNASSIGNED_DOMAIN_COLOR = "#aaaaaa"


def reload_domain_colors() -> dict[str, str]:
    """Refresh DOMAIN_COLORS after JSON edits (tests / long-running sessions)."""
    global DOMAIN_COLORS, _DOMAINS_REGISTRY_CACHE
    _DOMAINS_REGISTRY_CACHE = None
    DOMAIN_COLORS = _merged_domain_colors()
    return DOMAIN_COLORS


def get_domain_regions_for_emdb(emdb_id: str) -> list[DomainRegion]:
    """Return domain region definitions when ``emdb_id`` is in the registry, else []."""
    emdb_id = str(emdb_id).strip()
    for entry in _load_domains_registry():
        entry_ids = {str(x).strip() for x in entry.get("emdb_ids", [])}
        if emdb_id in entry_ids:
            return [_parse_domain_region(r) for r in entry.get("regions", [])]
    return []


def get_domain_regions_for_pair(emdb_a: str, emdb_b: str) -> list[DomainRegion]:
    """Return domain region definitions when annotated for this pair, else []."""
    pair_ids = {str(emdb_a).strip(), str(emdb_b).strip()}
    for entry in _load_domains_registry():
        entry_ids = {str(x).strip() for x in entry.get("emdb_ids", [])}
        if pair_ids <= entry_ids:
            return [_parse_domain_region(r) for r in entry.get("regions", [])]
    if is_trpv1_conformation_pair(emdb_a, emdb_b):
        return load_trpv1_domain_regions()
    if is_mgta_conformation_pair(emdb_a, emdb_b):
        return load_mgta_domain_regions()
    return []


def get_domain_assignments(
    chain_residue_list: Sequence[tuple[ResidueValidationRow, ResidueValidationRow]],
    regions: Sequence[DomainRegion] | None = None,
) -> dict[str, list[int]]:
    """Map domain name → chain-order residue indices (auth seq_id bands)."""
    if not regions:
        return {}
    assignments: dict[str, list[int]] = {reg.name: [] for reg in regions}
    for i, (row, _) in enumerate(chain_residue_list):
        for reg in regions:
            if region_matches_residue(reg, row):
                assignments[reg.name].append(i)
                break
    return assignments


def compute_domain_mean_coupling(
    corr: np.ndarray,
    assignments: dict[str, list[int]],
    *,
    domain_order: Sequence[str] | None = None,
    metric: str = "mean_abs",
    abs_threshold: float = 0.5,
) -> tuple[np.ndarray, list[str]]:
    """Summarize residue×residue coupling within each domain block.

    ``metric``:
        ``signed_mean`` — mean Pearson *r* (mixed signs often cancel → ~0 blocks).
        ``mean_abs`` — mean |*r*| (coupling magnitude regardless of sign).
        ``frac_strong`` — fraction of pairs with |*r*| > ``abs_threshold``.
    """
    if domain_order is None:
        names = [name for name, idx in assignments.items() if idx]
    else:
        names = [name for name in domain_order if assignments.get(name)]
    n_dom = len(names)
    mat = np.full((n_dom, n_dom), np.nan, dtype=np.float64)
    for i, di in enumerate(names):
        rows = np.asarray(assignments[di], dtype=int)
        for j, dj in enumerate(names):
            cols = np.asarray(assignments[dj], dtype=int)
            if rows.size and cols.size:
                block = corr[np.ix_(rows, cols)]
                if metric == "signed_mean":
                    mat[i, j] = float(np.nanmean(block))
                elif metric == "mean_abs":
                    mat[i, j] = float(np.nanmean(np.abs(block)))
                elif metric == "frac_strong":
                    mat[i, j] = float(np.nanmean(np.abs(block) > abs_threshold))
                else:
                    raise ValueError(
                        f"metric must be signed_mean, mean_abs, or frac_strong; got {metric!r}"
                    )
    return mat, names


def domain_residue_color(
    seq_num: int,
    regions: Sequence[DomainRegion],
    *,
    chain: str | None = None,
) -> str | None:
    """Return domain color for auth seq_id, or None when outside annotated bands."""
    for reg in regions:
        if chain is not None and (reg.chains is not None or reg.chain_prefixes is not None):
            ch = str(chain).strip()
            if reg.chains is not None and ch not in reg.chains:
                continue
            if reg.chain_prefixes is not None and not any(ch.startswith(p) for p in reg.chain_prefixes):
                continue
        if reg.seq_start <= int(seq_num) <= reg.seq_end:
            return DOMAIN_COLORS.get(reg.name, reg.color)
    return None


def domain_index_spans(
    use: Sequence[tuple[ResidueValidationRow, ResidueValidationRow]],
    regions: Sequence[DomainRegion],
) -> list[tuple[float, float, str, str]]:
    """Map auth seq_id regions to contiguous chain-order index intervals.

    Each chain copy yields its own span (homotetramer → up to four ANK/TM/C-term blocks).
    """
    spans: list[tuple[float, float, str, str]] = []
    n = len(use)
    for reg in regions:
        i = 0
        while i < n:
            if region_matches_residue(reg, use[i][0]):
                j = i + 1
                while j < n and region_matches_residue(reg, use[j][0]):
                    j += 1
                spans.append(
                    (float(i) - 0.5, float(j) - 0.5, reg.name, DOMAIN_COLORS.get(reg.name, reg.color))
                )
                i = j
            else:
                i += 1
    return spans


_SUMMARY_CSV_FIELDS: tuple[str, ...] = (
    "pair_id",
    "emdb_a",
    "emdb_b",
    "display_name_a",
    "display_name_b",
    "n_matched",
    "n_matched_in_mask_both",
    "median_ca_rmsd_a",
    "spearman_rmsd_vs_delta_reliability",
    "mean_ca_count",
    "mean_global_resolution_a",
    "diagonal_coupling_score",
    "domain_coupling_score",
    "coupling_layout_score",
    "hierarchical_cluster_score",
    "recommended_layout",
    "coupling_layout_threshold",
    "coverage_flag",
    "missing_pct_a",
    "missing_pct_b",
)


def _manifest_index(manifest: Path) -> dict[str, dict[str, str]]:
    import csv

    index: dict[str, dict[str, str]] = {}
    if not manifest.is_file():
        return index
    with manifest.open(newline="") as f:
        for row in csv.DictReader(f):
            eid = str(row.get("emdb_id", "")).strip()
            if eid:
                index[eid] = row
    return index


def _mean_pair_stat(stats: dict[str, object], key_a: str, key_b: str) -> float:
    vals: list[float] = []
    for key in (key_a, key_b):
        raw = stats.get(key, "")
        try:
            val = float(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if np.isfinite(val):
            vals.append(val)
    return float(np.mean(vals)) if vals else float("nan")


def _manifest_global_resolution_a(manifest_index: dict[str, dict[str, str]], emdb_id: str) -> float:
    row = manifest_index.get(str(emdb_id).strip(), {})
    try:
        return float(row.get("global_resolution_a", ""))
    except (TypeError, ValueError):
        return float("nan")


def collect_conformation_pair_rows(
    root: Path,
    *,
    manifest: Path | None = None,
) -> list[dict[str, object]]:
    """Load per-pair ``conformation_pair_stats.json`` files under ``root``."""
    from .cohort_labels import cohort_figure_label
    from .repo_paths import COHORT_MANIFEST

    manifest = manifest or COHORT_MANIFEST
    manifest_index = _manifest_index(manifest)
    rows: list[dict[str, object]] = []
    for stats_path in sorted(root.glob("emd_*_vs_*/conformation_pair_stats.json")):
        try:
            stats = json.loads(stats_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        emdb_a = str(stats.get("emdb_a", "")).strip()
        emdb_b = str(stats.get("emdb_b", "")).strip()
        if not emdb_a or not emdb_b:
            continue
        pair_id = stats_path.parent.name
        mean_ca = _mean_pair_stat(stats, "n_ca_total_a", "n_ca_total_b")
        res_a = _manifest_global_resolution_a(manifest_index, emdb_a)
        res_b = _manifest_global_resolution_a(manifest_index, emdb_b)
        mean_res = _mean_pair_stat(
            {"a": res_a, "b": res_b},
            "a",
            "b",
        )
        rows.append(
            {
                "pair_id": pair_id,
                "emdb_a": emdb_a,
                "emdb_b": emdb_b,
                "display_name_a": cohort_figure_label(emdb_a, manifest=manifest, short=True),
                "display_name_b": cohort_figure_label(emdb_b, manifest=manifest, short=True),
                "n_matched": stats.get("n_matched", ""),
                "n_matched_in_mask_both": stats.get("n_matched_in_mask_both", ""),
                "median_ca_rmsd_a": stats.get("median_ca_rmsd_a", ""),
                "spearman_rmsd_vs_delta_reliability": stats.get(
                    "spearman_rmsd_vs_delta_reliability", ""
                ),
                "mean_ca_count": mean_ca,
                "mean_global_resolution_a": mean_res,
                "diagonal_coupling_score": stats.get("diagonal_coupling_score", ""),
                "domain_coupling_score": stats.get("domain_coupling_score", ""),
                "coupling_layout_score": stats.get(
                    "coupling_layout_score", stats.get("cluster_separation_score", "")
                ),
                "hierarchical_cluster_score": stats.get("hierarchical_cluster_score", ""),
                "recommended_layout": stats.get("recommended_layout", ""),
                "coupling_layout_threshold": stats.get(
                    "coupling_layout_threshold", stats.get("cluster_separation_threshold", "")
                ),
                "coverage_flag": stats.get("coverage_flag", ""),
                "missing_pct_a": stats.get("missing_pct_a", ""),
                "missing_pct_b": stats.get("missing_pct_b", ""),
            }
        )
    return rows


def write_conformation_pairs_summary(
    root: Path,
    *,
    manifest: Path | None = None,
) -> tuple[Path | None, Path | None]:
    """
    Write cohort summary table for conformation pairs.

    Creates ``conformation_pairs_summary.csv`` and ``CONFORMATION_PAIRS.md`` in ``root``.
    """
    import csv
    from datetime import datetime, timezone

    rows = collect_conformation_pair_rows(root, manifest=manifest)
    if not rows:
        return None, None

    csv_path = root / "conformation_pairs_summary.csv"
    md_path = root / "CONFORMATION_PAIRS.md"
    root.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(_SUMMARY_CSV_FIELDS))
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in _SUMMARY_CSV_FIELDS})

    def _fmt_num(val: object, *, signed: bool = True) -> str:
        if val == "" or val is None:
            return "—"
        try:
            x = float(val)
        except (TypeError, ValueError):
            return str(val)
        if not np.isfinite(x):
            return "—"
        return f"{x:+.3f}" if signed else f"{x:.3f}"

    n_flag = sum(1 for r in rows if bool(r.get("coverage_flag")))
    table_lines = [
        "| Pair | ρ(RMSD,Δrel) | median RMSD (Å) | layout | recommended | n (mask) | coverage |",
        "|------|-------------:|----------------:|-------:|------------|--------:|----------|",
    ]
    for row in rows:
        flag = "flag" if bool(row.get("coverage_flag")) else "ok"
        table_lines.append(
            f"| EMD-{row['emdb_a']} vs {row['emdb_b']} "
            f"| {_fmt_num(row['spearman_rmsd_vs_delta_reliability'])} "
            f"| {_fmt_num(row['median_ca_rmsd_a'], signed=False)} "
            f"| {_fmt_num(row['coupling_layout_score'])} "
            f"| {row.get('recommended_layout', '—')} "
            f"| {row.get('n_matched_in_mask_both', '—')} "
            f"| {flag} |"
        )

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    md_path.write_text(
        f"""# Conformation pairs — cohort summary

Per-residue Cα RMSD (state B aligned onto A) vs Δreliability on matched in-mask residues.
Individual outputs live in ``emd_<A>_vs_<B>/``.

**Generated:** {generated}  
**Pairs:** {len(rows)} ({n_flag} with coverage flag >20% missing Cα)

---

## Summary table

{chr(10).join(table_lines)}

---

## Files

| File | Description |
|------|-------------|
| `conformation_pairs_summary.csv` | Machine-readable cohort table |
| `conformation_pairs_spearman_size_resolution_3d.png` | 3D cohort scatter (ρ vs size vs resolution) |
| `emd_<A>_vs_<B>/conformation_pair_stats.json` | Per-pair stats |
| `emd_<A>_vs_<B>/conformation_pair_summary.png` | Main figure |
"""
    )
    from .thesis_figures import plot_conformation_pairs_spearman_size_resolution_3d

    plot_conformation_pairs_spearman_size_resolution_3d(
        rows,
        save_path=root / "conformation_pairs_spearman_size_resolution_3d.png",
    )
    return csv_path, md_path


__all__ = [
    "COVERAGE_FLAG_THRESHOLD_PCT",
    "ConformationPairCoverage",
    "ConformationPairStats",
    "DomainRegion",
    "DOMAIN_COLORS",
    "UNASSIGNED_DOMAIN_COLOR",
    "compute_conformation_pair_coverage",
    "compute_conformation_pair_stats",
    "compute_domain_mean_coupling",
    "compute_per_residue_ca_rmsd",
    "compute_rmsd_superposition_diagnostics",
    "domain_index_spans",
    "domain_residue_color",
    "get_domain_assignments",
    "get_domain_regions_for_emdb",
    "get_domain_regions_for_pair",
    "region_matches_residue",
    "interior_residue_indices",
    "is_mgta_conformation_pair",
    "is_trpv1_conformation_pair",
    "kabsch_align_coords",
    "load_domain_regions_from_json",
    "load_mgta_domain_regions",
    "reload_domain_colors",
    "write_conformation_pair_md",
    "collect_conformation_pair_rows",
    "write_conformation_pairs_summary",
]
