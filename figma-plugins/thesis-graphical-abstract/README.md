# Thesis Graphical Abstract (Figma plugin)

Graphical abstract from a **single high-ρ exemplar map** in the **2.5–4 Å atomic-building regime**.

## Install

1. Figma Desktop → **Plugins → Development → Import plugin from manifest…**
2. Choose `figma-plugins/thesis-graphical-abstract/manifest.json`.

## Refresh data

```bash
# Default: highest ρ(Q, reliability) in 2.5–4 Å (currently EMD-62841, ρ≈+0.88)
uv run python scripts/run_graphical_abstract_export.py

# Thesis anchor instead
uv run python scripts/run_graphical_abstract_export.py --anchor

# Explicit map
uv run python scripts/run_graphical_abstract_export.py --emd-id 49450
```

Embeds JSON in `ui.html` (required for Figma). Reload the plugin after export.

## Default exemplar (auto)

| Field | Value |
|-------|-------|
| Map | **EMD-62841** (highest ρ in 2.5–4 Å, n≥500) |
| Global res | **2.80 Å** |
| ρ(Q, reliability) | **≈ +0.88** |
| ρ(Q, V) | **≈ +0.89** |
| ρ(Q, locRes) | **≈ −0.54** (weaker axis, even on strong maps) |

Thesis anchor **EMD-49450** (2.73 Å, ρ≈+0.82) is available via `--anchor`.

## Panels

| Panel | Content |
|-------|---------|
| **a** | BlocRes local resolution vs Q-score (exemplar Cα) |
| **b** | Reliability vs Q-score |
| **c** | Mean Q by reliability decile |
| **mini** | Per-map low-Q AUC bars |

## Related

- `cryoem_mrc/graphical_abstract_export.py`
- `outputs/cohort_summary/placement_rank_recovery.csv`
