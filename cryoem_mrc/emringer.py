"""Manifest → flat EMRinger CSV lookup (one PDB code per structure)."""

from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .repo_paths import COHORT_MANIFEST, EMRINGER_FLAT_DIR
from .structure_validation import iter_ca_residues

logger = logging.getLogger(__name__)

EMRINGER_SCORE_COLUMNS = (
    "emringer_score",
    "emringer",
    "score",
    "z_score",
    "zscore",
)

_CHAIN_SEQ_RE = re.compile(r"^([A-Za-z0-9]+?)(\d+)$")
_FUSED_LABEL_RE = re.compile(r"^([A-Z]{3})([A-Za-z0-9]+?)(\d+)$")


def pdb_code_from_flexibility_path(path: str | Path) -> str:
    """``pdb/9nhz.cif`` → ``9nhz`` (lowercase stem)."""
    return Path(str(path).strip()).stem.lower()


def emringer_csv_path(pdb_code: str, flat_dir: Path = EMRINGER_FLAT_DIR) -> Path:
    """Expected flat filename: ``{pdb_code}_emringer.csv``."""
    return flat_dir / f"{pdb_code.lower()}_emringer.csv"


@dataclass(frozen=True)
class EmringerDepositRow:
    emdb_id: str
    pdb_code: str
    pdb_path: Path
    reference_mrc: Path
    csv_path: Path

    @property
    def has_csv(self) -> bool:
        return self.csv_path.is_file()


def iter_emringer_deposits(
    manifest: Path = COHORT_MANIFEST,
    flat_dir: Path = EMRINGER_FLAT_DIR,
) -> list[EmringerDepositRow]:
    """Every manifest row with a local deposited structure and expected flat CSV path."""
    rows: list[EmringerDepositRow] = []
    pdb_to_emdb: dict[str, str] = {}

    with manifest.open(newline="") as f:
        for row in csv.DictReader(f):
            emdb_id = str(row.get("emdb_id", "")).strip()
            pdb_raw = str(row.get("flexibility_path_or_pdb", "")).strip()
            ref_raw = str(row.get("reference_mrc", "")).strip()
            if not emdb_id or not pdb_raw:
                continue
            pdb_path = Path(pdb_raw)
            if not pdb_path.is_file():
                continue

            pdb_code = pdb_code_from_flexibility_path(pdb_raw)
            prev = pdb_to_emdb.get(pdb_code)
            if prev is not None and prev != emdb_id:
                raise ValueError(
                    f"PDB code {pdb_code!r} maps to both EMD-{prev} and EMD-{emdb_id}; "
                    "EMRinger lookup must be one-to-one"
                )
            pdb_to_emdb[pdb_code] = emdb_id
            rows.append(
                EmringerDepositRow(
                    emdb_id=emdb_id,
                    pdb_code=pdb_code,
                    pdb_path=pdb_path,
                    reference_mrc=Path(ref_raw),
                    csv_path=emringer_csv_path(pdb_code, flat_dir),
                )
            )
    return rows


def missing_emringer_csvs(
    manifest: Path = COHORT_MANIFEST,
    flat_dir: Path = EMRINGER_FLAT_DIR,
) -> list[EmringerDepositRow]:
    return [row for row in iter_emringer_deposits(manifest, flat_dir) if not row.has_csv]


def build_manifest_emringer_lookup(
    manifest: Path = COHORT_MANIFEST,
    flat_dir: Path = EMRINGER_FLAT_DIR,
    *,
    require_existing: bool = True,
) -> dict[str, Path]:
    """Map each manifest ``emdb_id`` to its flat EMRinger CSV (one-to-one via PDB code).

    Rows without a local deposited structure are skipped. Duplicate PDB codes across
    manifest rows raise ``ValueError`` because the lookup must stay one-to-one.
    """
    lookup: dict[str, Path] = {}
    for row in iter_emringer_deposits(manifest, flat_dir):
        if require_existing and not row.has_csv:
            logger.debug(
                "EMD-%s (%s): no EMRinger CSV at %s",
                row.emdb_id,
                row.pdb_code,
                row.csv_path,
            )
            continue
        lookup[row.emdb_id] = row.csv_path
    return lookup


def _is_phenix_emringer_csv(csv_path: Path) -> bool:
    with csv_path.open() as f:
        first = f.readline().strip()
    return ",2mFo-DFc,chi" in first


def _parse_phenix_auth_label(label: str) -> tuple[str, int]:
    """Parse Phenix auth residue ids from ``GLU A22``, ``ARG A22 A``, or ``LYSAc4``."""
    label = label.strip()
    head = label.split()[0]
    fused = _FUSED_LABEL_RE.match(head)
    if fused is not None:
        return fused.group(2), int(fused.group(3))

    parts = label.split()
    if len(parts) >= 2:
        chain_seq = parts[1]
        match = _CHAIN_SEQ_RE.match(chain_seq)
        if match is None:
            raise ValueError(f"cannot parse Phenix EMRinger residue label {label!r}")
        return match.group(1), int(match.group(2))

    raise ValueError(f"cannot parse Phenix EMRinger residue label {label!r}")


def auth_to_label_map(pdb_path: Path) -> pd.DataFrame:
    """Map deposited auth chain/seq ids to label ``chain`` / ``seq_num`` used in metrics."""
    rows: list[dict[str, object]] = []
    for residue in iter_ca_residues(pdb_path):
        rows.append(
            {
                "auth_chain": str(residue.auth_chain or residue.chain).strip(),
                "auth_seq_num": int(residue.auth_seq_num or residue.seq_num),
                "chain": str(residue.chain).strip(),
                "seq_num": int(residue.seq_num),
            }
        )
    if not rows:
        raise ValueError(f"no Cα residues found in {pdb_path}")
    return pd.DataFrame(rows).drop_duplicates(["auth_chain", "auth_seq_num"])


def _density_at_modeled_chi(model_chi_deg: float, densities: list[float]) -> float:
    if not densities:
        return float("nan")
    idx = int(round(float(model_chi_deg) / 5.0)) % len(densities)
    return float(densities[idx])


def _load_phenix_emringer_scores(csv_path: Path) -> pd.DataFrame:
    """Parse Phenix ``phenix.emringer`` chi-scan CSV (no header row).

    Uses chi1 rows only and records map density at the modeled chi1 angle, which
    is the side-chain agreement signal EMRinger evaluates. Residue ids are auth
    chain/sequence numbers as exported by Phenix.
    """
    rows: list[tuple[str, int, float]] = []
    with csv_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 5:
                continue
            label, _map_type, chi_name, model_chi_raw = parts[:4]
            if chi_name != "chi1":
                continue
            auth_chain, auth_seq_num = _parse_phenix_auth_label(label)
            densities = [float(x) for x in parts[4:]]
            score = _density_at_modeled_chi(float(model_chi_raw), densities)
            rows.append((auth_chain, auth_seq_num, score))

    if not rows:
        raise ValueError(f"no chi1 rows parsed from Phenix EMRinger CSV {csv_path}")

    out = pd.DataFrame(rows, columns=["auth_chain", "auth_seq_num", "emringer_score"])
    return out.drop_duplicates(["auth_chain", "auth_seq_num"], keep="first")


def _resolve_score_column(df: pd.DataFrame) -> str:
    lower = {c.lower(): c for c in df.columns}
    for candidate in EMRINGER_SCORE_COLUMNS:
        if candidate in lower:
            return lower[candidate]
    raise KeyError(
        f"EMRinger CSV missing score column (expected one of {EMRINGER_SCORE_COLUMNS}); "
        f"got {list(df.columns)}"
    )


def _resolve_chain_column(df: pd.DataFrame) -> str:
    lower = {c.lower(): c for c in df.columns}
    for candidate in ("chain", "chain_id", "label_asym_id"):
        if candidate in lower:
            return lower[candidate]
    raise KeyError(f"EMRinger CSV missing chain column; got {list(df.columns)}")


def _resolve_seq_num_column(df: pd.DataFrame) -> str:
    lower = {c.lower(): c for c in df.columns}
    for candidate in ("seq_num", "residue_number", "residue_num", "resid", "res_id"):
        if candidate in lower:
            return lower[candidate]
    raise KeyError(f"EMRinger CSV missing residue index column; got {list(df.columns)}")


def _load_tabular_emringer_scores(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    chain_col = _resolve_chain_column(df)
    seq_col = _resolve_seq_num_column(df)
    score_col = _resolve_score_column(df)

    out = pd.DataFrame(
        {
            "chain": df[chain_col].astype(str).str.strip(),
            "seq_num": pd.to_numeric(df[seq_col], errors="coerce").astype("Int64"),
            "emringer_score": pd.to_numeric(df[score_col], errors="coerce"),
        }
    )
    return out.dropna(subset=["chain", "seq_num"])


def load_emringer_scores(
    csv_path: Path,
    *,
    pdb_path: Path | None = None,
) -> pd.DataFrame:
    """Load per-residue EMRinger scores keyed by metrics ``chain`` / ``seq_num``."""
    if _is_phenix_emringer_csv(csv_path):
        if pdb_path is None or not pdb_path.is_file():
            raise ValueError(
                f"Phenix EMRinger CSV {csv_path} requires pdb_path for auth→label mapping"
            )
        auth_df = _load_phenix_emringer_scores(csv_path)
        label_map = auth_to_label_map(pdb_path)
        merged = auth_df.merge(label_map, on=["auth_chain", "auth_seq_num"], how="inner")
        return merged[["chain", "seq_num", "emringer_score"]].drop_duplicates(
            ["chain", "seq_num"], keep="first"
        )
    return _load_tabular_emringer_scores(csv_path)


def attach_emringer_scores(
    metrics_df: pd.DataFrame,
    csv_path: Path,
    *,
    pdb_path: Path | None = None,
) -> pd.DataFrame:
    """Left-merge ``emringer_score`` onto a per-residue metrics table."""
    em_df = load_emringer_scores(csv_path, pdb_path=pdb_path)
    out = metrics_df.copy()
    if "emringer_score" in out.columns:
        out = out.drop(columns=["emringer_score"])
    return out.merge(em_df, on=["chain", "seq_num"], how="left")
