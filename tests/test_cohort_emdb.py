"""Tests for EMDB cohort metadata helpers."""

from cryoem_mrc.cohort_emdb import (
    parse_emdb_global_resolution_a,
    parse_emdb_reported_sharpening_b,
)


def test_parse_emdb_global_resolution_a() -> None:
    entry = {
        "structure_determination_list": {
            "structure_determination": [
                {
                    "image_processing": [
                        {
                            "final_reconstruction": {
                                "resolution": {"valueOf_": "2.73", "units": "Å"},
                            }
                        }
                    ]
                }
            ]
        }
    }
    assert parse_emdb_global_resolution_a(entry) == 2.73
    assert parse_emdb_global_resolution_a({}) is None


def test_parse_emdb_reported_sharpening_b() -> None:
    entry = {
        "structure_determination_list": {
            "structure_determination": [
                {
                    "image_processing": [
                        {
                            "details": "Map sharpening B-factor: -67 Å^2 applied in Relion postprocess."
                        }
                    ]
                }
            ]
        }
    }
    assert parse_emdb_reported_sharpening_b(entry) == -67.0
    assert parse_emdb_reported_sharpening_b({}) is None
