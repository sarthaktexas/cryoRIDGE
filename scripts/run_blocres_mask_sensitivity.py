"""Compare ρ(V, BlocRes) with vs without depositor-contour ``-Mask``.

Reads masked ``locres_blocres.mrc`` and nomask ``locres_blocres_nomask.mrc`` for
each entry. Correlations are still evaluated inside the depositor ``in_contour_mask``
(same gate as ``run_v_locres_summary.py``).

Run nomask BlocRes first (does not overwrite masked maps)::

    python scripts/run_blocres_local_resolution.py --emd-id 49450 --no-mask --force
    python scripts/run_blocres_mask_sensitivity.py --emd-id 49450 11638 52525

Output: ``outputs/cohort_summary/blocres_mask_sensitivity.csv``
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np
import pandas as pd

from cryoem_mrc.local_resolution import locres_blocres_nomask_path, locres_blocres_path
from cryoem_mrc.repo_paths import COHORT_MANIFEST, OUTPUTS_ROOT
from scripts.run_v_locres_summary import _summarize_entry

OUTPUT_CSV = OUTPUTS_ROOT / "cohort_summary" / "blocres_mask_sensitivity.csv"
DEFAULT_SPOT_CHECK = ("49450", "11638", "52525")

COLUMNS = [
    "emdb_id",
    "display_name",
    "rho_masked",
    "rho_nomask",
    "delta_rho",
    "n_pairs",
    "locres_range_masked",
    "locres_range_nomask",
    "has_masked",
    "has_nomask",
    "nan_reason_masked",
    "nan_reason_nomask",
]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--manifest", type=Path, default=COHORT_MANIFEST)
    p.add_argument(
        "--emd-id",
        nargs="+",
        default=None,
        help=f"EMDB IDs to compare (default spot-check: {', '.join(DEFAULT_SPOT_CHECK)})",
    )
    p.add_argument("--out", type=Path, default=OUTPUT_CSV)
    return p.parse_args(argv)


def _display_names(manifest: Path) -> dict[str, str]:
    names: dict[str, str] = {}
    with manifest.open(newline="") as f:
        for row in csv.DictReader(f):
            eid = str(row.get("emdb_id", "")).strip()
            if eid:
                names[eid] = row.get("display_name", "").strip()
    return names


def _eligible_emd_ids(manifest: Path) -> list[str]:
    ids: list[str] = []
    with manifest.open(newline="") as f:
        for row in csv.DictReader(f):
            emdb_id = str(row.get("emdb_id", "")).strip()
            pdb = row.get("flexibility_path_or_pdb", "").strip()
            if emdb_id and pdb and Path(pdb).is_file():
                ids.append(emdb_id)
    return ids


def _compare_one(emdb_id: str, *, manifest: Path) -> dict | None:
    masked_path = locres_blocres_path(emdb_id)
    nomask_path = locres_blocres_nomask_path(emdb_id)
    has_masked = masked_path.is_file()
    has_nomask = nomask_path.is_file()
    if not has_masked and not has_nomask:
        return None

    masked = (
        _summarize_entry(
            emdb_id,
            manifest=manifest,
            locres_source="blocres",
            locres_path_override=masked_path if has_masked else None,
        )
        if has_masked
        else None
    )
    nomask = (
        _summarize_entry(
            emdb_id,
            manifest=manifest,
            locres_source="blocres",
            locres_path_override=nomask_path,
        )
        if has_nomask
        else None
    )
    if masked is None and nomask is None:
        return None

    rho_m = float(masked["rho"]) if masked is not None else float("nan")
    rho_n = float(nomask["rho"]) if nomask is not None else float("nan")
    delta = rho_n - rho_m if np.isfinite(rho_m) and np.isfinite(rho_n) else float("nan")
    n_pairs = int(masked["n_pairs"]) if masked is not None else int(nomask["n_pairs"])

    return {
        "emdb_id": emdb_id,
        "rho_masked": rho_m,
        "rho_nomask": rho_n,
        "delta_rho": delta,
        "n_pairs": n_pairs,
        "locres_range_masked": masked["locres_range"] if masked is not None else float("nan"),
        "locres_range_nomask": nomask["locres_range"] if nomask is not None else float("nan"),
        "has_masked": has_masked,
        "has_nomask": has_nomask,
        "nan_reason_masked": masked["nan_reason"] if masked is not None else "missing_map",
        "nan_reason_nomask": nomask["nan_reason"] if nomask is not None else "missing_map",
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    names = _display_names(args.manifest)
    emd_ids = args.emd_id if args.emd_id else list(DEFAULT_SPOT_CHECK)

    rows: list[dict] = []
    for emdb_id in emd_ids:
        emdb_id = emdb_id.strip()
        row = _compare_one(emdb_id, manifest=args.manifest)
        if row is None:
            print(f"[mask_sensitivity] EMD-{emdb_id}: skip (no maps or ineligible)", flush=True)
            continue
        row["display_name"] = names.get(emdb_id, "")
        rows.append(row)
        print(
            f"[mask_sensitivity] EMD-{emdb_id}: "
            f"ρ_masked={row['rho_masked']:+.3f} ρ_nomask={row['rho_nomask']:+.3f} "
            f"Δ={row['delta_rho']:+.3f} "
            f"range {row['locres_range_masked']:.2f}→{row['locres_range_nomask']:.2f} Å",
            flush=True,
        )

    if not rows:
        print("[mask_sensitivity] no comparable entries", file=sys.stderr)
        return 1

    out_df = pd.DataFrame(rows)[COLUMNS]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)

    finite = out_df["delta_rho"].dropna()
    if len(finite):
        print(
            f"[mask_sensitivity] n={len(finite)} median Δρ={float(finite.median()):+.3f} "
            f"(nomask − masked)",
            flush=True,
        )
    print(f"[mask_sensitivity] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
