"""Thesis wrapper for half-map reliability export (EMD-49450 defaults).

Prefer explicit paths via ``cryoridge reliability`` for generic use.
"""

from __future__ import annotations

import sys

from cryoem_mrc.reliability_driver import main
from cryoem_mrc.repo_paths import (
    ANCHOR_EMDB_ID,
    DATA_ROOT,
    halfmap_reliability_dir,
)


def _default_argv() -> list[str]:
    data_dir = DATA_ROOT / "emd_49450-mgtA_e2p+e1"
    emd_id = ANCHOR_EMDB_ID
    contour = "0.116"
    return [
        "--reference",
        str(data_dir / f"emd_{emd_id}.map"),
        "--half1",
        str(data_dir / f"emd_{emd_id}_half_map_1.map"),
        "--half2",
        str(data_dir / f"emd_{emd_id}_half_map_2.map"),
        "--features",
        str(data_dir / f"emd_{emd_id}_avg_features_t0116.npz"),
        "--contour",
        contour,
        "--out-dir",
        str(halfmap_reliability_dir(emd_id)),
        "--label",
        f"emd_{emd_id}",
    ]


if __name__ == "__main__":
    argv = sys.argv[1:] or _default_argv()
    raise SystemExit(main(argv))
