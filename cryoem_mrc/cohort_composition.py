"""Deposited-model composition helpers for cohort stratification plots."""

from __future__ import annotations

from pathlib import Path

import gemmi


def compute_deposited_na_residue_fraction(structure_path: str | Path) -> float:
    """
    Fraction of polymer residues that are RNA/DNA in a deposited model.

    Counts one residue per mmCIF residue (after dropping waters and altlocs).
    Protein residues are those with a Cα atom; nucleic acid residues have P or
    C1' without Cα. This matches ribosome/nucleosome/ribozyme depositions in the
    cohort PDB set.
    """
    path = Path(structure_path)
    st = gemmi.read_structure(str(path))
    st.remove_alternative_conformations()
    st.remove_hydrogens()

    n_protein = 0
    n_na = 0
    for model in st:
        for chain in model:
            for residue in chain:
                if residue.entity_type == gemmi.EntityType.Water:
                    continue
                has_ca = residue.find_atom("CA", "\0") is not None
                has_p = residue.find_atom("P", "\0") is not None
                has_c1p = residue.find_atom("C1'", "\0") is not None
                if has_ca:
                    n_protein += 1
                elif has_p or has_c1p:
                    n_na += 1
        break

    total = n_protein + n_na
    if total == 0:
        return float("nan")
    return float(n_na) / float(total)


def infer_na_residue_fraction_from_name(display_name: str) -> float:
    """
    Fallback RNA/DNA fraction when no deposited model is listed in the manifest.

    Uses coarse architecture keywords; returns NaN when unknown.
    """
    name = display_name.lower()
    if any(k in name for k in ("ribozyme",)):
        return 1.0
    if any(k in name for k in ("nucleosome",)):
        return 0.25
    if any(k in name for k in ("ribosome", "70s", "spliceosome")):
        return 0.40
    if any(k in name for k in ("spike", "proteasome", "apoferritin", "gB", "hsv")):
        return 0.0
    return float("nan")


def resolve_cohort_na_residue_fraction(
    *,
    structure_path: str | None = None,
    display_name: str = "",
) -> float:
    """PDB-based fraction when available; otherwise keyword fallback."""
    pdb = str(structure_path or "").strip()
    if pdb:
        path = Path(pdb)
        if path.is_file():
            return compute_deposited_na_residue_fraction(path)
    return infer_na_residue_fraction_from_name(display_name)
