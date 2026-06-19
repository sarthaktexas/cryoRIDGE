# Thesis Q vs V Cohort (Figma plugin)

Recreates the **cohort summary** figure (`qscore_vs_V_cohort.png`) from
`scripts/run_qscore_validation.py`.

## Install

1. Figma Desktop → **Plugins → Development → Import plugin from manifest…**
2. Choose `figma-plugins/thesis-q-vs-v/manifest.json`.

## Refresh data

```bash
uv run python scripts/run_qscore_figma_export.py
```

Requires `outputs/cohort_summary/qscore_correlations.csv` from:

```bash
uv run python scripts/run_qscore_validation.py --cohort-summary
```

Reload the plugin after export.

## Panels

| Panel | Content |
|-------|---------|
| **a** | Horizontal bar chart: Spearman ρ(Q, V) per structure, colored by global resolution |
| **b** (cohort) | Scatter: ρ vs global resolution + linear trend |
| **resolution sweep** | Line plot: median ρ in 0.5 Å bins (2–6 Å), 4 Å cutoff marker — **panel B** of `qscore_resolution_sensitivity.png` |
| **standard bins** | Vertical bars: median ρ in ≤2.5 / 2.5–4 / 4–6 / >6 Å bins — **panel A** |
| **cutoff** | Dual line plot: median ρ for maps with res ≤ cutoff vs res > cutoff — **panel C** |
| **A + B + C** | Full three-panel resolution sensitivity figure |

Default generate mode is **Resolution sensitivity — panels A + B + C**.

## Related

- `cryoem_mrc/qscore_figma_export.py`
- `outputs/cohort_summary/qscore_correlations.csv`
