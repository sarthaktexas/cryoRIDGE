"""Per-residue Q-score validation: cryo-EM-native map quality vs LH constraint V."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy import stats

from .analysis import build_contour_mask
from .map_grid import MapGrid, load_map_grid
from .repo_paths import (
    COHORT_MANIFEST,
    emd_output_dir,
    halfmap_reliability_dir,
    resolve_halfmap_reliability_dir,
)
from .structure_validation import (
    CaResidue,
    iter_ca_residues,
    load_cohort_manifest_row,
    physical_xyz_to_voxel_indices,
    sample_volume_at_ca,
)


@dataclass
class QscoreResidueRow:
    """Per-residue Q-score and LH V samples for external validation."""

    chain: str
    seq_num: int
    seq_icode: str
    res_name: str
    x: float
    y: float
    z: float
    b_iso: float
    q_score: float
    reliability_constraint_V: float
    reliability_constraint_V_rank: float
    in_contour_mask: bool
    auth_chain: str = ""
    auth_seq_num: int = 0

    @property
    def residue_key(self) -> tuple[str, int, str]:
        return (self.chain, self.seq_num, self.seq_icode)


@dataclass
class QscoreValidationStats:
    """Spearman correlations for Q-score vs LH V (in-mask Cα)."""

    emdb_id: str
    pdb_id: str
    n_residues: int
    n_in_mask: int
    n_with_q_score: int
    spearman_q_vs_V: float
    spearman_q_vs_V_rank: float
    spearman_q_vs_b_iso: float = float("nan")
    median_q_by_b_tercile: dict[str, float] = field(default_factory=dict)
    notes: str = ""


def _percentile_rank(values: np.ndarray) -> np.ndarray:
    """Map finite values to (0, 1] by rank (higher value = higher rank)."""
    out = np.full(values.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(values)
    if finite.sum() < 1:
        return out
    vals = values[finite]
    order = np.argsort(vals, kind="mergesort")
    ranks = np.empty(vals.size, dtype=np.float64)
    ranks[order] = np.arange(1, vals.size + 1, dtype=np.float64)
    out[finite] = ranks / vals.size
    return out


def compute_per_residue_q_scores(
    structure_path: str | Path,
    map_path: str | Path,
    residues: Sequence[CaResidue],
    *,
    num_points: int = 8,
) -> np.ndarray:
    """
    Run 3dem/qscore and align per-residue Q-scores to ``residues`` (gemmi Cα list).

    Matching uses deposited auth chain + auth seq id (BioPython convention in qscore).
    """
    from qscore.mrc_utils import load_mrc
    from qscore.pdb_utils import get_protein_from_file_path
    from qscore.q_score import calculate_q_score

    prot = get_protein_from_file_path(str(structure_path))
    mrc = load_mrc(str(map_path), False)
    atoms = prot.atom_positions[prot.atom_mask.astype(bool)]
    q_atoms = calculate_q_score(atoms, mrc, num_points=num_points)

    q_per_res = np.full(len(prot.aatype), np.nan, dtype=np.float64)
    atom_idx = 0
    for resid in range(len(prot.aatype)):
        n_atoms = int(prot.atom_mask[resid].sum())
        if n_atoms == 0:
            continue
        q_per_res[resid] = float(np.mean(q_atoms[atom_idx : atom_idx + n_atoms]))
        atom_idx += n_atoms

    q_by_auth: dict[tuple[str, int], float] = {}
    for resid in range(len(prot.aatype)):
        if not np.isfinite(q_per_res[resid]):
            continue
        chain = str(prot.chain_id[prot.chain_index[resid]])
        seq = int(prot.residue_index[resid])
        q_by_auth[(chain, seq)] = float(q_per_res[resid])

    out = np.full(len(residues), np.nan, dtype=np.float64)
    for i, res in enumerate(residues):
        key = (res.auth_chain or res.chain, res.auth_seq_num or res.seq_num)
        if key in q_by_auth:
            out[i] = q_by_auth[key]
    return out


def build_qscore_validation_table(
    residues: Sequence[CaResidue],
    *,
    grid: MapGrid,
    reference_density: np.ndarray,
    contour: float,
    reliability_constraint_V: np.ndarray,
    q_scores: np.ndarray,
    window_radius: int = 0,
) -> list[QscoreResidueRow]:
    """Join Cα coordinates with Q-scores and V samples inside the contour mask."""
    if reference_density.shape != reliability_constraint_V.shape:
        raise ValueError("reference_density and reliability_constraint_V must share shape")
    mask = build_contour_mask(reference_density, contour)

    v_s = sample_volume_at_ca(
        reliability_constraint_V, grid, residues, window_radius=window_radius
    )
    in_mask_s = sample_volume_at_ca(
        mask.astype(np.float64), grid, residues, window_radius=window_radius
    )

    v_in_mask = v_s[in_mask_s >= 0.5]
    v_rank_all = np.full(len(residues), np.nan, dtype=np.float64)
    if v_in_mask.size:
        rank_in_mask = _percentile_rank(v_in_mask)
        j = 0
        for i, inside in enumerate(in_mask_s >= 0.5):
            if inside:
                v_rank_all[i] = rank_in_mask[j]
                j += 1

    rows: list[QscoreResidueRow] = []
    for i, res in enumerate(residues):
        rows.append(
            QscoreResidueRow(
                chain=res.chain,
                seq_num=res.seq_num,
                seq_icode=res.seq_icode,
                res_name=res.res_name,
                x=res.x,
                y=res.y,
                z=res.z,
                b_iso=res.b_iso,
                q_score=float(q_scores[i]),
                reliability_constraint_V=float(v_s[i]),
                reliability_constraint_V_rank=float(v_rank_all[i]),
                in_contour_mask=bool(in_mask_s[i] >= 0.5),
                auth_chain=res.auth_chain or res.chain,
                auth_seq_num=res.auth_seq_num or res.seq_num,
            )
        )
    return rows


def compute_qscore_validation_stats(
    rows: Sequence[QscoreResidueRow],
    *,
    emdb_id: str,
    pdb_id: str,
    in_mask_only: bool = True,
) -> QscoreValidationStats:
    """Spearman ρ(Q-score, V) on residues with finite Q and V."""
    use = [
        r
        for r in rows
        if (r.in_contour_mask if in_mask_only else True)
        and np.isfinite(r.q_score)
        and np.isfinite(r.reliability_constraint_V)
    ]
    n_all = len(rows)
    n_use = len(use)
    n_in_mask = sum(1 for r in rows if r.in_contour_mask)
    if n_use < 10:
        return QscoreValidationStats(
            emdb_id=emdb_id,
            pdb_id=pdb_id,
            n_residues=n_all,
            n_in_mask=n_in_mask,
            n_with_q_score=sum(1 for r in rows if np.isfinite(r.q_score)),
            spearman_q_vs_V=float("nan"),
            spearman_q_vs_V_rank=float("nan"),
            notes="too few residues for correlation",
        )

    q = np.array([r.q_score for r in use], dtype=np.float64)
    v = np.array([r.reliability_constraint_V for r in use], dtype=np.float64)
    v_rank = np.array([r.reliability_constraint_V_rank for r in use], dtype=np.float64)
    b = np.array([r.b_iso for r in use], dtype=np.float64)

    rho_v, _ = stats.spearmanr(q, v)
    rho_vr, _ = stats.spearmanr(q, v_rank)
    rho_b, _ = stats.spearmanr(q, b)

    med_by_tercile: dict[str, float] = {}
    if b.size >= 3:
        terciles = np.quantile(b, [1 / 3, 2 / 3])
        labels = np.where(b <= terciles[0], "low_B", np.where(b <= terciles[1], "mid_B", "high_B"))
        for label in ("low_B", "mid_B", "high_B"):
            qb = q[labels == label]
            if qb.size:
                med_by_tercile[label] = float(np.median(qb))

    return QscoreValidationStats(
        emdb_id=emdb_id,
        pdb_id=pdb_id,
        n_residues=n_all,
        n_in_mask=n_in_mask,
        n_with_q_score=sum(1 for r in rows if np.isfinite(r.q_score)),
        spearman_q_vs_V=float(rho_v),
        spearman_q_vs_V_rank=float(rho_vr),
        spearman_q_vs_b_iso=float(rho_b),
        median_q_by_b_tercile=med_by_tercile,
    )


def read_qscore_lookup(path: str | Path) -> dict[tuple[str, int, str], float]:
    """Load ``q_score`` keyed by mmCIF (label_asym_id, label_seq_id, insertion)."""
    q_lookup, _ = read_qscore_validation_lookups(path)
    return q_lookup


def read_qscore_validation_lookups(
    path: str | Path,
) -> tuple[dict[tuple[str, int, str], float], dict[tuple[str, int, str], float]]:
    """Load Q-score and LH constraint V keyed by mmCIF residue key."""
    path = Path(path)
    q_lookup: dict[tuple[str, int, str], float] = {}
    v_lookup: dict[tuple[str, int, str], float] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            key = (row["chain"], int(row["seq_num"]), row["seq_icode"])
            q_raw = row.get("q_score", "")
            v_raw = row.get("reliability_constraint_V", "")
            q_lookup[key] = float(q_raw) if q_raw else float("nan")
            v_lookup[key] = float(v_raw) if v_raw else float("nan")
    return q_lookup, v_lookup


def build_qscore_only_table(
    residues: Sequence[CaResidue],
    *,
    grid: MapGrid,
    reference_density: np.ndarray,
    contour: float,
    q_scores: np.ndarray,
    window_radius: int = 0,
) -> list[QscoreResidueRow]:
    """Per-residue Q-scores and mask flags only (V filled later via merge)."""
    mask = build_contour_mask(reference_density, contour)
    in_mask_s = sample_volume_at_ca(
        mask.astype(np.float64), grid, residues, window_radius=window_radius
    )
    rows: list[QscoreResidueRow] = []
    for i, res in enumerate(residues):
        rows.append(
            QscoreResidueRow(
                chain=res.chain,
                seq_num=res.seq_num,
                seq_icode=res.seq_icode,
                res_name=res.res_name,
                x=res.x,
                y=res.y,
                z=res.z,
                b_iso=res.b_iso,
                q_score=float(q_scores[i]),
                reliability_constraint_V=float("nan"),
                reliability_constraint_V_rank=float("nan"),
                in_contour_mask=bool(in_mask_s[i] >= 0.5),
                auth_chain=res.auth_chain or res.chain,
                auth_seq_num=res.auth_seq_num or res.seq_num,
            )
        )
    return rows


def write_qscore_per_residue_csv(path: str | Path, rows: Sequence[QscoreResidueRow]) -> Path:
    """Write Q-only table (no V columns) for deferred merge with ``reliability.npz``."""
    path = Path(path)
    fieldnames = [
        "chain",
        "seq_num",
        "auth_chain",
        "auth_seq_num",
        "seq_icode",
        "res_name",
        "x",
        "y",
        "z",
        "b_iso",
        "q_score",
        "in_contour_mask",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "chain": r.chain,
                    "seq_num": r.seq_num,
                    "auth_chain": r.auth_chain or r.chain,
                    "auth_seq_num": r.auth_seq_num or r.seq_num,
                    "seq_icode": r.seq_icode,
                    "res_name": r.res_name,
                    "x": f"{r.x:.3f}",
                    "y": f"{r.y:.3f}",
                    "z": f"{r.z:.3f}",
                    "b_iso": f"{r.b_iso:.2f}",
                    "q_score": f"{r.q_score:.6f}" if np.isfinite(r.q_score) else "",
                    "in_contour_mask": int(r.in_contour_mask),
                }
            )
    return path


def read_qscore_per_residue_csv(path: str | Path) -> list[QscoreResidueRow]:
    """Load Q-only rows written by :func:`write_qscore_per_residue_csv`."""
    rows: list[QscoreResidueRow] = []
    with Path(path).open(newline="") as f:
        for row in csv.DictReader(f):
            q_raw = row.get("q_score", "")
            rows.append(
                QscoreResidueRow(
                    chain=row["chain"],
                    seq_num=int(row["seq_num"]),
                    seq_icode=row.get("seq_icode", ""),
                    res_name=row.get("res_name", ""),
                    x=float(row["x"]),
                    y=float(row["y"]),
                    z=float(row["z"]),
                    b_iso=float(row.get("b_iso", "nan")),
                    q_score=float(q_raw) if q_raw else float("nan"),
                    reliability_constraint_V=float("nan"),
                    reliability_constraint_V_rank=float("nan"),
                    in_contour_mask=bool(int(row.get("in_contour_mask", "0"))),
                    auth_chain=row.get("auth_chain") or row["chain"],
                    auth_seq_num=int(row.get("auth_seq_num") or row["seq_num"]),
                )
            )
    return rows


def production_v_metric_path(emdb_id: str) -> Path:
    """Authoritative per-residue V from ``metric_comparison/residue_metrics.csv``."""
    return emd_output_dir(emdb_id) / "metric_comparison" / "residue_metrics.csv"


def load_production_v_metric_lookup(emdb_id: str) -> dict[tuple[str, int], float]:
    path = production_v_metric_path(emdb_id)
    if not path.is_file():
        raise FileNotFoundError(f"EMD-{emdb_id} missing production v_metric table: {path}")
    lookup: dict[tuple[str, int], float] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            lookup[(row["chain"], int(row["seq_num"]))] = float(row["v_metric"])
    return lookup


def _recompute_in_mask_v_ranks(rows: Sequence[QscoreResidueRow]) -> list[QscoreResidueRow]:
    """Percentile rank of production V among in-mask Cα (higher V → higher rank)."""
    v_in_mask = np.array(
        [
            r.reliability_constraint_V
            for r in rows
            if r.in_contour_mask and np.isfinite(r.reliability_constraint_V)
        ],
        dtype=np.float64,
    )
    rank_in_mask = _percentile_rank(v_in_mask) if v_in_mask.size else np.array([], dtype=np.float64)
    out: list[QscoreResidueRow] = []
    j = 0
    for r in rows:
        v_rank = float("nan")
        if r.in_contour_mask and np.isfinite(r.reliability_constraint_V):
            v_rank = float(rank_in_mask[j])
            j += 1
        out.append(
            QscoreResidueRow(
                chain=r.chain,
                seq_num=r.seq_num,
                seq_icode=r.seq_icode,
                res_name=r.res_name,
                x=r.x,
                y=r.y,
                z=r.z,
                b_iso=r.b_iso,
                q_score=r.q_score,
                reliability_constraint_V=r.reliability_constraint_V,
                reliability_constraint_V_rank=v_rank,
                in_contour_mask=r.in_contour_mask,
                auth_chain=r.auth_chain,
                auth_seq_num=r.auth_seq_num,
            )
        )
    return out


def attach_production_v_metric(
    rows: Sequence[QscoreResidueRow],
    emdb_id: str,
) -> list[QscoreResidueRow]:
    """Attach production ``v_metric`` (not legacy ``reliability.npz`` samples)."""
    lookup = load_production_v_metric_lookup(emdb_id)
    merged: list[QscoreResidueRow] = []
    for r in rows:
        merged.append(
            QscoreResidueRow(
                chain=r.chain,
                seq_num=r.seq_num,
                seq_icode=r.seq_icode,
                res_name=r.res_name,
                x=r.x,
                y=r.y,
                z=r.z,
                b_iso=r.b_iso,
                q_score=r.q_score,
                reliability_constraint_V=lookup.get((r.chain, r.seq_num), float("nan")),
                reliability_constraint_V_rank=float("nan"),
                in_contour_mask=r.in_contour_mask,
                auth_chain=r.auth_chain,
                auth_seq_num=r.auth_seq_num,
            )
        )
    return _recompute_in_mask_v_ranks(merged)


def resolve_qscore_bundle_dir(emdb_id: str, *, for_write: bool = False) -> Path:
    """Canonical write path; reads fall back to legacy bundle during migration."""
    if for_write:
        return halfmap_reliability_dir(emdb_id)
    return resolve_halfmap_reliability_dir(emdb_id)


def load_reliability_constraint_v(npz_path: Path) -> np.ndarray:
    """Deprecated: prefer :func:`attach_production_v_metric`."""
    with np.load(npz_path, allow_pickle=False) as d:
        v_key = (
            "reliability_smoothness"
            if "reliability_smoothness" in d
            else "reliability_constraint_V"
        )
        return np.asarray(d[v_key], dtype=np.float32)


def write_qscore_validation_csv(path: str | Path, rows: Sequence[QscoreResidueRow]) -> Path:
    path = Path(path)
    fieldnames = [
        "chain",
        "seq_num",
        "auth_chain",
        "auth_seq_num",
        "seq_icode",
        "res_name",
        "x",
        "y",
        "z",
        "b_iso",
        "q_score",
        "reliability_constraint_V",
        "reliability_constraint_V_rank",
        "in_contour_mask",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "chain": r.chain,
                    "seq_num": r.seq_num,
                    "auth_chain": r.auth_chain or r.chain,
                    "auth_seq_num": r.auth_seq_num or r.seq_num,
                    "seq_icode": r.seq_icode,
                    "res_name": r.res_name,
                    "x": f"{r.x:.3f}",
                    "y": f"{r.y:.3f}",
                    "z": f"{r.z:.3f}",
                    "b_iso": f"{r.b_iso:.2f}",
                    "q_score": f"{r.q_score:.6f}" if np.isfinite(r.q_score) else "",
                    "reliability_constraint_V": f"{r.reliability_constraint_V:.6f}",
                    "reliability_constraint_V_rank": (
                        f"{r.reliability_constraint_V_rank:.6f}"
                        if np.isfinite(r.reliability_constraint_V_rank)
                        else ""
                    ),
                    "in_contour_mask": int(r.in_contour_mask),
                }
            )
    return path


def write_qscore_validation_md(
    path: Path,
    stats: QscoreValidationStats,
    *,
    pdb_path: Path,
    map_path: Path,
    contour: float,
) -> None:
    tercile_lines = "\n".join(
        f"| {label} | {val:.3f} |"
        for label, val in stats.median_q_by_b_tercile.items()
    )
    caveat = f"\n\n**Caveat:** {stats.notes}" if stats.notes else ""
    text = f"""# Q-score external validation — EMD-{stats.emdb_id}

Cryo-EM-native comparison of **per-residue Q-scores** (deposited model + map) vs
**production constraint V** (`v_metric` from ``metric_comparison/residue_metrics.csv``).

**Model:** `{pdb_path}`  
**Map:** `{map_path}`  
**Mask:** deposited reference ρ ≥ {contour} (Cα: nearest voxel)  
**Residues:** {stats.n_residues:,} Cα total; **{stats.n_in_mask:,}** inside contour mask;
**{stats.n_with_q_score:,}** with finite Q-score

---

## Spearman correlations (in-mask Cα with finite Q)

| Comparison | ρ |
|------------|--:|
| Q-score vs constraint V | {stats.spearman_q_vs_V:+.3f} |
| Q-score vs in-mask V rank | {stats.spearman_q_vs_V_rank:+.3f} |
| Q-score vs deposited B_iso | {stats.spearman_q_vs_b_iso:+.3f} |

**Framing:** Q-scores require a fitted model; V is computed from half-maps alone.
A **positive** ρ(Q, V) supports model-free recovery of per-residue map quality.{caveat}

---

## Median Q-score by B-factor tercile (in-mask)

| B-factor tercile | Median Q |
|------------------|---------:|
{tercile_lines or "| — | — |"}

---

## Files

| File | Description |
|------|-------------|
| `qscore_validation.csv` | Per-residue Q vs V table |
| `figures/qscore_vs_V_scatter.png` | Lead validation scatter |
"""
    path.write_text(text)


def run_emdb_qscore_only(
    emd_id: str,
    *,
    manifest: Path = COHORT_MANIFEST,
    reference: Path | None = None,
    pdb: Path | None = None,
    contour: float | None = None,
    window_radius: int = 0,
    num_points: int = 8,
) -> tuple[int, list[QscoreResidueRow], Path]:
    """
    Compute per-residue Q-scores only (no ``reliability.npz`` required).

    Writes ``qscore_per_residue.csv`` under the half-map reliability bundle dir.
    """
    row = load_cohort_manifest_row(manifest, emd_id)
    ref_path = reference or Path(row["reference_mrc"])
    pdb_path = pdb or Path(row["flexibility_path_or_pdb"])
    contour_val = contour if contour is not None else float(row["contour"])
    out_dir = resolve_qscore_bundle_dir(emd_id, for_write=True)

    for label, p in (("reference", ref_path), ("pdb", pdb_path)):
        if not p.exists():
            raise FileNotFoundError(f"EMD-{emd_id} missing {label}: {p}")

    residues = iter_ca_residues(pdb_path)
    grid = load_map_grid(ref_path, dtype=np.float32)
    reference_density = np.asarray(grid.data, dtype=np.float32)
    q_scores = compute_per_residue_q_scores(
        pdb_path, ref_path, residues, num_points=num_points
    )
    rows = build_qscore_only_table(
        residues,
        grid=grid,
        reference_density=reference_density,
        contour=contour_val,
        q_scores=q_scores,
        window_radius=window_radius,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    write_qscore_per_residue_csv(out_dir / "qscore_per_residue.csv", rows)
    return 0, rows, out_dir


def run_emdb_qscore_merge_reliability(
    emd_id: str,
    *,
    manifest: Path = COHORT_MANIFEST,
    reliability_npz: Path | None = None,
    reference: Path | None = None,
    pdb: Path | None = None,
    contour: float | None = None,
    window_radius: int = 0,
) -> tuple[int, list[QscoreResidueRow], QscoreValidationStats, Path]:
    """
    Merge precomputed Q-scores with production ``v_metric`` and write validation outputs.

    ``reliability_npz`` is ignored (kept for CLI compatibility with older invocations).
    """
    del reliability_npz, window_radius
    row = load_cohort_manifest_row(manifest, emd_id)
    ref_path = reference or Path(row["reference_mrc"])
    pdb_path = pdb or Path(row["flexibility_path_or_pdb"])
    contour_val = contour if contour is not None else float(row["contour"])
    read_dir = resolve_qscore_bundle_dir(emd_id)
    out_dir = resolve_qscore_bundle_dir(emd_id, for_write=True)
    q_only_path = read_dir / "qscore_per_residue.csv"

    for label, p in (
        ("reference", ref_path),
        ("pdb", pdb_path),
        ("qscore_per_residue.csv", q_only_path),
        ("production v_metric", production_v_metric_path(emd_id)),
    ):
        if not p.exists():
            raise FileNotFoundError(f"EMD-{emd_id} missing {label}: {p}")

    q_only = read_qscore_per_residue_csv(q_only_path)
    residues = iter_ca_residues(pdb_path)
    if len(residues) != len(q_only):
        raise ValueError(
            f"EMD-{emd_id}: Cα count mismatch ({len(residues)} vs {len(q_only)} Q rows)"
        )

    rows = attach_production_v_metric(q_only, emd_id)

    pdb_id = pdb_path.stem
    stats = compute_qscore_validation_stats(rows, emdb_id=emd_id, pdb_id=pdb_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_qscore_validation_csv(out_dir / "qscore_validation.csv", rows)
    write_qscore_validation_md(
        out_dir / "QSCORE_VALIDATION.md",
        stats,
        pdb_path=pdb_path,
        map_path=ref_path,
        contour=contour_val,
    )
    return 0, rows, stats, out_dir


def run_emdb_qscore_validation(
    emd_id: str,
    *,
    manifest: Path = COHORT_MANIFEST,
    reliability_npz: Path | None = None,
    reference: Path | None = None,
    pdb: Path | None = None,
    contour: float | None = None,
    window_radius: int = 0,
    num_points: int = 8,
) -> tuple[int, list[QscoreResidueRow], QscoreValidationStats | None, Path]:
    """
    Run Q-score vs production V validation for one EMDB entry.

    Returns ``(exit_code, rows, stats, out_dir)``. ``stats`` is None when skipped.
    """
    del reliability_npz
    row = load_cohort_manifest_row(manifest, emd_id)
    ref_path = reference or Path(row["reference_mrc"])
    pdb_path = pdb or Path(row["flexibility_path_or_pdb"])
    contour_val = contour if contour is not None else float(row["contour"])
    out_dir = resolve_qscore_bundle_dir(emd_id, for_write=True)

    for label, p in (
        ("reference", ref_path),
        ("pdb", pdb_path),
        ("production v_metric", production_v_metric_path(emd_id)),
    ):
        if not p.exists():
            raise FileNotFoundError(f"EMD-{emd_id} missing {label}: {p}")

    residues = iter_ca_residues(pdb_path)
    grid = load_map_grid(ref_path, dtype=np.float32)
    reference_density = np.asarray(grid.data, dtype=np.float32)

    q_scores = compute_per_residue_q_scores(
        pdb_path, ref_path, residues, num_points=num_points
    )
    rows = build_qscore_only_table(
        residues,
        grid=grid,
        reference_density=reference_density,
        contour=contour_val,
        q_scores=q_scores,
        window_radius=window_radius,
    )
    rows = attach_production_v_metric(rows, emd_id)

    pdb_id = pdb_path.stem
    stats = compute_qscore_validation_stats(rows, emdb_id=emd_id, pdb_id=pdb_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_qscore_per_residue_csv(out_dir / "qscore_per_residue.csv", rows)
    write_qscore_validation_csv(out_dir / "qscore_validation.csv", rows)
    write_qscore_validation_md(
        out_dir / "QSCORE_VALIDATION.md",
        stats,
        pdb_path=pdb_path,
        map_path=ref_path,
        contour=contour_val,
    )
    return 0, rows, stats, out_dir
