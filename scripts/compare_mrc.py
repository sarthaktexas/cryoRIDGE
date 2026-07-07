"""Check whether two MRC/MAP files are the same.

Examples::

    python scripts/compare_mrc.py map_a.mrc map_b.mrc
    python scripts/compare_mrc.py --hash map_a.mrc map_b.mrc
    python scripts/compare_mrc.py --rtol 1e-5 --atol 1e-6 map_a.mrc map_b.mrc
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cryoem_mrc.mrc_compare import compare_mrc_files


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("path_a", type=Path, help="First MRC/MAP file")
    parser.add_argument("path_b", type=Path, help="Second MRC/MAP file")
    parser.add_argument(
        "--hash",
        action="store_true",
        help="Print SHA-256 digests when files are not byte-identical",
    )
    parser.add_argument(
        "--bytes-only",
        action="store_true",
        help="Only compare raw file bytes",
    )
    parser.add_argument(
        "--skip-grid",
        action="store_true",
        help="Skip grid/header alignment checks",
    )
    parser.add_argument(
        "--skip-data",
        action="store_true",
        help="Skip voxel-value comparison",
    )
    parser.add_argument("--rtol", type=float, default=0.0, help="Relative tolerance for voxel data")
    parser.add_argument("--atol", type=float, default=0.0, help="Absolute tolerance for voxel data")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report = compare_mrc_files(
        args.path_a,
        args.path_b,
        check_bytes=True,
        hash_bytes=args.hash,
        check_grid=not args.skip_grid and not args.bytes_only,
        check_data=not args.skip_data and not args.bytes_only,
        rtol=args.rtol,
        atol=args.atol,
    )
    for line in report.summary_lines():
        print(line)
    return 0 if report.same else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
