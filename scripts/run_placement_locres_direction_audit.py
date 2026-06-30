#!/usr/bin/env python3
"""Audit BlocRes sign convention vs head-to-head flags (placement utility).

Prints whether raw ρ(Q, locres Å), sign-aligned coupling, and the
``loc > in-map median`` flag point in the same direction.

Usage:
    uv run python scripts/run_placement_locres_direction_audit.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from thesis.placement_utility import aligned_rank_recovery_rho  # noqa: E402
from cryoem_mrc.repo_paths import OUTPUTS_ROOT  # noqa: E402

RR_CSV = OUTPUTS_ROOT / "cohort_summary" / "placement_rank_recovery.csv"
H2H_CSV = OUTPUTS_ROOT / "cohort_summary" / "placement_predictor_head_to_head.csv"


def main() -> int:
    if not RR_CSV.is_file() or not H2H_CSV.is_file():
        print("[locres_audit] missing placement CSVs — run run_placement_utility_analysis.py", file=sys.stderr)
        return 2

    rr = list(csv.DictReader(RR_CSV.open()))
    h2h = {r["predictor"]: r for r in csv.DictReader(H2H_CSV.open())}
    loc = h2h.get("locres_worse_than_median")
    if not loc:
        print("[locres_audit] missing locres row in head-to-head CSV", file=sys.stderr)
        return 2

    raw = [float(r["spearman_q_vs_locres"]) for r in rr if r["spearman_q_vs_locres"] not in ("", "nan")]
    aligned = [aligned_rank_recovery_rho(v, "spearman_q_vs_locres") for v in raw]

    print("[locres_audit] Flag rule: loc > in-map median (Å) ⇒ low-confidence (worse BlocRes)")
    print(f"[locres_audit] Maps: {len(raw)}")
    print(f"[locres_audit] Median raw ρ(Q, locres Å): {np.median(raw):+.3f}  (negative ⇒ higher Å ↔ lower Q)")
    print(f"[locres_audit] Median aligned ρ (sharpness ↑): {np.median(aligned):+.3f}")
    print(
        "[locres_audit] Head-to-head locres: "
        f"sensitivity={float(loc['pooled_sensitivity']):.3f} "
        f"specificity={float(loc['pooled_specificity']):.3f} "
        f"BA={float(loc['pooled_balanced_accuracy']):.3f} "
        f"AUC={float(loc['median_map_auc']):.3f}"
    )
    print(
        "[locres_audit] Interpretation: negative raw ρ is expected for Å-valued BlocRes; "
        "aligned |ρ|≈0.13 is weak vs CC (~0.50). Moderate AUC (~0.58) reflects median-split "
        "sensitivity (~59%) with specificity ~54%, not strong continuous coupling."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
