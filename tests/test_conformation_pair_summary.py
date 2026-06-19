"""Cohort summary for conformation pairs."""

from __future__ import annotations

import json
from pathlib import Path

from cryoem_mrc.conformation_pair import (
    collect_conformation_pair_rows,
    write_conformation_pairs_summary,
)


def test_write_conformation_pairs_summary(tmp_path: Path) -> None:
    pair_dir = tmp_path / "emd_11111_vs_22222"
    pair_dir.mkdir()
    (pair_dir / "conformation_pair_stats.json").write_text(
        json.dumps(
            {
                "emdb_a": "11111",
                "emdb_b": "22222",
                "n_matched": 100,
                "n_matched_in_mask_both": 90,
                "n_ca_total_a": 800,
                "n_ca_total_b": 820,
                "median_ca_rmsd_a": 1.5,
                "spearman_rmsd_vs_delta_reliability": 0.5,
                "coupling_layout_score": 0.2,
                "recommended_layout": "block",
                "coverage_flag": False,
            }
        )
    )

    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "emdb_id,display_name,global_resolution_a\n"
        "11111,Test A,3.0\n"
        "22222,Test B,4.0\n"
    )

    rows = collect_conformation_pair_rows(tmp_path, manifest=manifest)
    assert len(rows) == 1
    assert rows[0]["emdb_a"] == "11111"
    assert rows[0]["mean_ca_count"] == 810.0
    assert rows[0]["mean_global_resolution_a"] == 3.5

    csv_path, md_path = write_conformation_pairs_summary(tmp_path, manifest=manifest)
    assert csv_path is not None and md_path is not None
    assert "spearman_rmsd_vs_delta_reliability" in csv_path.read_text()
    assert "mean_ca_count" in csv_path.read_text()
    assert "ρ(RMSD,Δrel)" in md_path.read_text()
    assert (tmp_path / "conformation_pairs_spearman_size_resolution_3d.png").is_file()
