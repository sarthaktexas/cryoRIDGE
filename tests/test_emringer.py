from __future__ import annotations

import csv
from pathlib import Path

from unittest.mock import patch

import pytest

from cryoem_mrc.emringer import (
    attach_emringer_scores,
    build_manifest_emringer_lookup,
    emringer_csv_path,
    load_emringer_scores,
    pdb_code_from_flexibility_path,
)


def test_pdb_code_from_flexibility_path() -> None:
    assert pdb_code_from_flexibility_path("pdb/9nhz.cif") == "9nhz"
    assert pdb_code_from_flexibility_path("PDB/7A4M.CIF") == "7a4m"


def test_emringer_csv_path() -> None:
    flat = Path("/tmp/emringer_flat")
    assert emringer_csv_path("9nhz", flat) == flat / "9nhz_emringer.csv"


def test_build_manifest_emringer_lookup_one_to_one(tmp_path: Path) -> None:
    pdb_dir = tmp_path / "pdb"
    pdb_dir.mkdir()
    flat = tmp_path / "flat"
    flat.mkdir()
    for code in ("9nhz", "7a4m"):
        (pdb_dir / f"{code}.cif").write_text("data\n")
        (flat / f"{code}_emringer.csv").write_text(
            "chain,seq_num,emringer_score\nA,1,1.5\n", encoding="utf-8"
        )

    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["emdb_id", "flexibility_path_or_pdb"],
        )
        writer.writeheader()
        writer.writerow({"emdb_id": "49450", "flexibility_path_or_pdb": str(pdb_dir / "9nhz.cif")})
        writer.writerow({"emdb_id": "11638", "flexibility_path_or_pdb": str(pdb_dir / "7a4m.cif")})

    lookup = build_manifest_emringer_lookup(
        manifest,
        flat,
        require_existing=True,
    )
    assert lookup == {
        "49450": flat / "9nhz_emringer.csv",
        "11638": flat / "7a4m_emringer.csv",
    }


def test_build_manifest_emringer_lookup_rejects_duplicate_pdb(tmp_path: Path) -> None:
    pdb_dir = tmp_path / "pdb"
    pdb_dir.mkdir()
    flat = tmp_path / "flat"
    flat.mkdir()
    (pdb_dir / "9nhz.cif").write_text("data\n")
    (flat / "9nhz_emringer.csv").write_text(
        "chain,seq_num,emringer_score\nA,1,1.0\n", encoding="utf-8"
    )

    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["emdb_id", "flexibility_path_or_pdb"],
        )
        writer.writeheader()
        writer.writerow({"emdb_id": "49450", "flexibility_path_or_pdb": str(pdb_dir / "9nhz.cif")})
        writer.writerow({"emdb_id": "48534", "flexibility_path_or_pdb": str(pdb_dir / "9nhz.cif")})

    with pytest.raises(ValueError, match="one-to-one"):
        build_manifest_emringer_lookup(manifest, flat)


def test_load_and_attach_emringer_scores(tmp_path: Path) -> None:
    csv_path = tmp_path / "9nhz_emringer.csv"
    csv_path.write_text(
        "chain,seq_num,emringer_score\nA,10,2.0\nA,11,1.0\n",
        encoding="utf-8",
    )
    em_df = load_emringer_scores(csv_path)
    assert list(em_df.columns) == ["chain", "seq_num", "emringer_score"]
    assert len(em_df) == 2

    metrics = __import__("pandas").DataFrame(
        {
            "chain": ["A", "A", "B"],
            "seq_num": [10, 11, 1],
            "v_metric": [3.0, 2.5, 1.0],
        }
    )
    merged = attach_emringer_scores(metrics, csv_path)
    assert merged.loc[0, "emringer_score"] == pytest.approx(2.0)
    assert merged.loc[1, "emringer_score"] == pytest.approx(1.0)
    assert merged.loc[2, "emringer_score"] != merged.loc[2, "emringer_score"]  # NaN


def test_load_phenix_emringer_scores(tmp_path: Path) -> None:
    csv_path = tmp_path / "9nhz_emringer.csv"
    pdb_path = tmp_path / "9nhz.cif"
    pdb_path.write_text("data\n", encoding="utf-8")

    csv_path.write_text(
        "GLU A22,2mFo-DFc,chi1,10.0,0.100,0.200,0.300,0.400,0.500\n"
        "GLU A22,2mFo-DFc,chi2,55.0,0.900,0.800,0.700,0.600,0.500\n"
        "ILE A23,2mFo-DFc,chi1,5.0,0.111,0.222,0.333,0.444,0.555\n",
        encoding="utf-8",
    )

    from cryoem_mrc.structure_validation import CaResidue

    fake_residues = [
        CaResidue(
            chain="L1",
            seq_num=10,
            seq_icode="",
            res_name="GLU",
            x=0.0,
            y=0.0,
            z=0.0,
            b_iso=0.0,
            auth_chain="A",
            auth_seq_num=22,
        ),
        CaResidue(
            chain="L1",
            seq_num=11,
            seq_icode="",
            res_name="ILE",
            x=0.0,
            y=0.0,
            z=0.0,
            b_iso=0.0,
            auth_chain="A",
            auth_seq_num=23,
        ),
    ]
    with patch("cryoem_mrc.emringer.iter_ca_residues", return_value=fake_residues):
        em_df = load_emringer_scores(csv_path, pdb_path=pdb_path)
    assert len(em_df) == 2
    row22 = em_df[(em_df["chain"] == "L1") & (em_df["seq_num"] == 10)].iloc[0]
    assert row22["emringer_score"] == pytest.approx(0.300)
    row23 = em_df[(em_df["chain"] == "L1") & (em_df["seq_num"] == 11)].iloc[0]
    assert row23["emringer_score"] == pytest.approx(0.222)
