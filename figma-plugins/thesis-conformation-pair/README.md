# Thesis Conformation Pair (Figma plugin)

Recreates **panel B** of the conformation-pair summary figure: Cα RMSD vs
Δreliability scatter (domain-colored residues), without ChimeraX structure panels.

Default export: **MsbA outward-facing vs inward-facing** (EMD-41596 vs EMD-41598).

## Install

1. Figma Desktop → **Plugins → Development → Import plugin from manifest…**
2. Choose `figma-plugins/thesis-conformation-pair/manifest.json`.

## Refresh data

```bash
uv run python scripts/run_conformation_pair_figma_export.py
```

Other pairs:

```bash
uv run python scripts/run_conformation_pair_figma_export.py --emd-a 23129 --emd-b 23130
```

Requires `residue_validation.csv` for both maps (from b-factor / reliability validation).

Reload the plugin after export.

## Panel

| Panel | Content |
|-------|---------|
| **b** | Scatter: per-residue Cα RMSD (B aligned onto A) vs Δreliability, colored by domain |

## Related

- `cryoem_mrc/conformation_pair_figma_export.py`
- `scripts/run_residue_bfactor_conformation_pair.py`
- `cryoem_mrc/thesis_figures.py` → `_draw_conformation_summary_scatter_panel`
