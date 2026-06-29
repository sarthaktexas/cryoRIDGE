from __future__ import annotations

import csv
import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from cryoem_mrc.emringer import (
    attach_emringer_scores,
    build_manifest_emringer_lookup,
    emringer_csv_path,
    load_emringer_scores,
    pdb_code_from_flexibility_path,
)


class TestEmringer(unittest.TestCase):
    def test_pdb_code_from_flexibility_path(self) -> None:
        self.assertEqual(pdb_code_from_flexibility_path("pdb/9nhz.cif"), "9nhz")
        self.assertEqual(pdb_code_from_flexibility_path("PDB/7A4M.CIF"), "7a4m")

    def test_emringer_csv_path(self) -> None:
        flat = Path("/tmp/emringer_flat")
        self.assertEqual(emringer_csv_path("9nhz", flat), flat / "9nhz_emringer.csv")

    def test_build_manifest_emringer_lookup_one_to_one(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdb_dir = root / "pdb"
            pdb_dir.mkdir()
            flat = root / "flat"
            flat.mkdir()
            for code in ("9nhz", "7a4m"):
                (pdb_dir / f"{code}.cif").write_text("data\n")
                (flat / f"{code}_emringer.csv").write_text(
                    "chain,seq_num,emringer_score\nA,1,1.5\n", encoding="utf-8"
                )

            manifest = root / "manifest.csv"
            with manifest.open("w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["emdb_id", "flexibility_path_or_pdb"],
                )
                writer.writeheader()
                writer.writerow(
                    {"emdb_id": "49450", "flexibility_path_or_pdb": str(pdb_dir / "9nhz.cif")}
                )
                writer.writerow(
                    {"emdb_id": "11638", "flexibility_path_or_pdb": str(pdb_dir / "7a4m.cif")}
                )

            lookup = build_manifest_emringer_lookup(
                manifest,
                flat,
                require_existing=True,
            )
            self.assertEqual(
                lookup,
                {
                    "49450": flat / "9nhz_emringer.csv",
                    "11638": flat / "7a4m_emringer.csv",
                },
            )

    def test_build_manifest_emringer_lookup_rejects_duplicate_pdb(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdb_dir = root / "pdb"
            pdb_dir.mkdir()
            flat = root / "flat"
            flat.mkdir()
            (pdb_dir / "9nhz.cif").write_text("data\n")
            (flat / "9nhz_emringer.csv").write_text(
                "chain,seq_num,emringer_score\nA,1,1.0\n", encoding="utf-8"
            )

            manifest = root / "manifest.csv"
            with manifest.open("w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["emdb_id", "flexibility_path_or_pdb"],
                )
                writer.writeheader()
                writer.writerow(
                    {"emdb_id": "49450", "flexibility_path_or_pdb": str(pdb_dir / "9nhz.cif")}
                )
                writer.writerow(
                    {"emdb_id": "48534", "flexibility_path_or_pdb": str(pdb_dir / "9nhz.cif")}
                )

            with self.assertRaisesRegex(ValueError, "one-to-one"):
                build_manifest_emringer_lookup(manifest, flat)

    def test_load_and_attach_emringer_scores(self) -> None:
        with TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "9nhz_emringer.csv"
            csv_path.write_text(
                "chain,seq_num,emringer_score\nA,10,2.0\nA,11,1.0\n",
                encoding="utf-8",
            )
            em_df = load_emringer_scores(csv_path)
            self.assertEqual(list(em_df.columns), ["chain", "seq_num", "emringer_score"])
            self.assertEqual(len(em_df), 2)

            metrics = __import__("pandas").DataFrame(
                {
                    "chain": ["A", "A", "B"],
                    "seq_num": [10, 11, 1],
                    "v_metric": [3.0, 2.5, 1.0],
                }
            )
            merged = attach_emringer_scores(metrics, csv_path)
            self.assertAlmostEqual(merged.loc[0, "emringer_score"], 2.0)
            self.assertAlmostEqual(merged.loc[1, "emringer_score"], 1.0)
            self.assertTrue(math.isnan(merged.loc[2, "emringer_score"]))

    def test_load_phenix_emringer_scores(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "9nhz_emringer.csv"
            pdb_path = root / "9nhz.cif"
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
            self.assertEqual(len(em_df), 2)
            row22 = em_df[(em_df["chain"] == "L1") & (em_df["seq_num"] == 10)].iloc[0]
            self.assertAlmostEqual(row22["emringer_score"], 0.300)
            row23 = em_df[(em_df["chain"] == "L1") & (em_df["seq_num"] == 11)].iloc[0]
            self.assertAlmostEqual(row23["emringer_score"], 0.222)


if __name__ == "__main__":
    unittest.main()
