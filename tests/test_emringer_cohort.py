from __future__ import annotations

import csv
from pathlib import Path

from cryoem_mrc.emringer_cohort import (
    BUILDING_REGIME_MAX_RESOLUTION_A,
    BUILDING_REGIME_MIN_RESOLUTION_A,
    EMRINGER_INTERPRETABLE_MAX_RESOLUTION_A,
    building_regime_panel,
    emringer_interpretable,
    emringer_panel_reason,
    load_manifest_global_resolution_a,
)


def test_emringer_interpretable_barad_breakdown(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["emdb_id", "global_resolution_a"]
        )
        writer.writeheader()
        writer.writerow({"emdb_id": "11638", "global_resolution_a": "1.22"})
        writer.writerow({"emdb_id": "4940", "global_resolution_a": "6.20"})
        writer.writerow({"emdb_id": "28498", "global_resolution_a": "5.40"})
        writer.writerow({"emdb_id": "48311", "global_resolution_a": "3.70"})

    res = load_manifest_global_resolution_a(manifest)
    assert EMRINGER_INTERPRETABLE_MAX_RESOLUTION_A == 5.0
    assert emringer_interpretable("11638", resolutions=res)
    assert emringer_interpretable("48311", resolutions=res)
    assert not emringer_interpretable("4940", resolutions=res)
    assert not emringer_interpretable("28498", resolutions=res)


def test_building_regime_panel(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["emdb_id", "global_resolution_a"]
        )
        writer.writeheader()
        writer.writerow({"emdb_id": "49450", "global_resolution_a": "2.73"})
        writer.writerow({"emdb_id": "48311", "global_resolution_a": "3.70"})
        writer.writerow({"emdb_id": "4941", "global_resolution_a": "4.00"})
        writer.writerow({"emdb_id": "11638", "global_resolution_a": "1.22"})
        writer.writerow({"emdb_id": "28498", "global_resolution_a": "5.40"})

    res = load_manifest_global_resolution_a(manifest)
    assert BUILDING_REGIME_MIN_RESOLUTION_A == 2.5
    assert BUILDING_REGIME_MAX_RESOLUTION_A == 4.0
    assert building_regime_panel("49450", resolutions=res)
    assert building_regime_panel("48311", resolutions=res)
    assert building_regime_panel("4941", resolutions=res)
    assert not building_regime_panel("11638", resolutions=res)
    assert not building_regime_panel("28498", resolutions=res)


def test_emringer_panel_reason(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["emdb_id", "global_resolution_a"]
        )
        writer.writeheader()
        writer.writerow({"emdb_id": "49450", "global_resolution_a": "2.73"})
        writer.writerow({"emdb_id": "9156", "global_resolution_a": "6.84"})

    res = load_manifest_global_resolution_a(manifest)
    assert emringer_panel_reason("49450", resolutions=res) == ""
    assert emringer_panel_reason("9156", resolutions=res) == (
        f"resolution_coarser_than_{EMRINGER_INTERPRETABLE_MAX_RESOLUTION_A:g}A_barad2015"
    )
