# Thesis Placement Utility — Figma plugin

Recreates placement Q-score validation figures from cohort summary CSVs:

| Mode | Figure |
|------|--------|
| **head-to-head** | Three-panel predictor comparison — `placement_predictor_head_to_head.png` |
| **rank recovery** | Median ρ(Q, proxy) bars — `placement_rank_recovery.png` |
| **low_q_roc** | Representative-map ROC curves (median AUC headline) — `placement_low_q_roc.png` |
| **all** | Head-to-head + rank recovery + ROC |

## Data export

```bash
uv run python scripts/run_placement_figma_export.py
```

Prerequisite: run `scripts/run_placement_utility_analysis.py` first.

## Install

1. Figma Desktop → Plugins → Development → Import plugin from manifest
2. Select `figma-plugins/thesis-placement/manifest.json`
3. Re-run export after CSV updates; reload plugin after `code.js` changes
