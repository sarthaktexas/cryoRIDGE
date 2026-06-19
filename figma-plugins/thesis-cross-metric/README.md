# Thesis Cross Metric — Figma plugin

Recreates cohort cross-metric figures from `cross_metric_correlations.csv`:

| Mode | Figure |
|------|--------|
| **median heatmap** | Median Spearman ρ matrix (metric × metric) — `cohort_cross_metric_median.png` |
| **locres pairs** | Per-map ρ vs BlocRes for V, B, windowed CC, variance — `cohort_cross_metric_locres_pairs.png` |
| **both** | Side-by-side layout |

## Data export

```bash
uv run python scripts/run_cross_metric_figma_export.py
```

Prerequisite: per-map `outputs/emd_<ID>/metric_comparison/cross_metric_correlations.csv`
(from `scripts/run_metric_comparison_export.py`).

## Install

1. Figma Desktop → Plugins → Development → Import plugin from manifest
2. Select `figma-plugins/thesis-cross-metric/manifest.json`
3. Re-run export and reload plugin after data changes (`code.js` changes only need reload)

## Styling

Uses thesis palette: diverging blue–white–red heatmap; categorical bars for locres pairs.
