# Thesis Q vs V by Class — Figma plugin

Recreates the cohort box-and-strip figure from `cohort_q_vs_v_by_class.png`:

- **y-axis:** Spearman ρ(Q-score, constraint V) per deposited structure
- **x-axis:** coarse protein class (ordered by class median ρ)
- **overlay:** jittered per-structure points + cohort median reference line

## Data export

```bash
uv run python scripts/run_reliability_by_class_figma_export.py
```

Prerequisite: `outputs/cohort_summary/qscore_correlations.csv`
(from `scripts/run_qscore_validation.py --cohort-summary`).

Regenerate the matplotlib figure with:

```bash
uv run python scripts/run_cohort_summary_figures.py
```

## Install

1. Figma Desktop → Plugins → Development → Import plugin from manifest
2. Select `figma-plugins/thesis-reliability-by-class/manifest.json`
3. Re-run export and reload plugin after data changes (`code.js` changes only need reload)

## Styling

Uses thesis categorical palette for protein classes; matches `plot_cohort_q_vs_v_by_class` in `thesis_figures.py`.
