# Cohort gap audit

**Generated:** 2026-06-11T03:18:27Z UTC
**Manifest:** `cohort/manifest.csv`
**Outputs root:** `/Users/sarthakmohanty/Developer/thesis/outputs` (**not reachable — mount thesis-data volume**)

## Summary

| Metric | Count |
|--------|------:|
| Active manifest rows | 37 |
| Excluded / optional | 4 |
| With local reference + halves | 0 |
| **`reliability.npz` present** | **0** / 37 |
| Pending pipeline (`--pending`) | 0 |
| Pending Q-score (`--all`) | 0 |
| Pending BlocRes (`--all`) | 0 |
| Missing local map data | 37 |
| Narrative figures in `docs/figures/` | 26 / 26 |

> **Blocker:** local `data/` symlinks are broken or maps are not downloaded.
> Mount `/Volumes/Undergrad Thesis/thesis-data` (or repoint `data/` / `outputs/`) before running the pipeline.

## Last verified baseline (2026-06-08)

When the thesis-data volume was last mounted, [docs/COHORT.md](../docs/COHORT.md) reported:

- **38/38** maps pipelined (`reliability.npz`), including EMD-33736 (now manifest-`excluded`, RNA-only)
- **33** Q-score validation runs; **38** BlocRes `locres_blocres.mrc` files
- **26/26** narrative figures synced under `docs/figures/`
- **Still open:** spike conformation pair 11497↔11494; FSC-Q Xmipp cluster refresh; Results prose polish

Re-run `python scripts/run_cohort_gap_audit.py --write-run-list` after mounting to replace this section with live counts.

## Per-map status

| EMDB | Name | res (Å) | src | data | pipeline | BlocRes | Q | B-val |
|------|------|--------:|-----|:----:|:--------:|:-------:|:-:|:-:|
| 49450 | MgtA (E2P+E1) | 2.73 | b_factor | **no** | **no** | **no** | **no** | **no** |
| 11638 | Apoferritin | 1.22 | b_factor | **no** | **no** | **no** | **no** | **no** |
| 23129 | TRPV1 pH 6a | 3.70 | b_factor | **no** | **no** | **no** | **no** | **no** |
| 23130 | TRPV1 pH 6c | 3.66 | conformational_pair | **no** | **no** | **no** | **no** | **no** |
| 48923 | MgtA (E2.Mg.BeF3) | 2.59 | conformational_pair | **no** | **no** | **no** | **no** | **no** |
| 48534 | MgtA (E2P.Mg) | 3.07 | conformational_pair | **no** | **no** | **no** | **no** | **no** |
| 5995 | Beta-galactosidase (excluded - no half-maps) | 3.20 | _excluded_ | — | — | — | — | — |
| 16091 | Beta-galactosidase (5ms time-resolved) | 3.30 | b_factor | **no** | **no** | **no** | **no** | **no** |
| 41756 | LRRK2 CTD (+GZD824) | 2.90 | b_factor | **no** | **no** | **no** | **no** | **no** |
| 45261 | 70S Ribosome | 5.00 | literature_segment | **no** | **no** | **no** | **no** | — |
| 61596 | HSV-2 gB (+Fab16F9) | 2.18 | b_factor | **no** | **no** | **no** | **no** | **no** |
| 52525 | Complex III (mitochondrial) | 2.52 | b_factor | **no** | **no** | **no** | **no** | **no** |
| 52515 | UBR4-KCMF1 (N-terminal) | 5.70 | windowed_halfmap_correlation_only | **no** | **no** | **no** | **no** | — |
| 30207 | Beta-galactosidase+PEG v1 (excluded) | 3.10 | _excluded_ | — | — | — | — | — |
| 30208 | Beta-galactosidase+PEG v2 (excluded) | 3.20 | _excluded_ | — | — | — | — | — |
| 24120 | 70S pre-translocation (E. coli) | 2.33 | conformational_pair | **no** | **no** | **no** | **no** | **no** |
| 25418 | 70S post-translocation (E. coli) | 2.90 | conformational_pair | **no** | **no** | **no** | **no** | **no** |
| 41596 | MsbA outward-facing | 2.68 | conformational_pair | **no** | **no** | **no** | **no** | **no** |
| 41598 | MsbA inward-facing | 3.60 | conformational_pair | **no** | **no** | **no** | **no** | **no** |
| 45604 | beta2AR inactive (carazolol) | 3.50 | conformational_pair | **no** | **no** | **no** | **no** | **no** |
| 45603 | beta2AR apo | 3.95 | conformational_pair | **no** | **no** | **no** | **no** | **no** |
| 28498 | Eag1 voltage sensor down | 5.40 | conformational_pair | **no** | **no** | **no** | **no** | **no** |
| 28487 | Eag1 voltage sensor up | 3.90 | conformational_pair | **no** | **no** | **no** | **no** | **no** |
| 13308 | GroEL-GroES ADP (tight) | 3.43 | conformational_pair | **no** | **no** | **no** | **no** | **no** |
| 16119 | GroEL-GroES ADP (turnover) | 3.40 | conformational_pair | **no** | **no** | **no** | **no** | **no** |
| 11497 | SARS-CoV-2 spike 3 RBD-down | 3.50 | conformational_pair | **no** | **no** | **no** | **no** | **no** |
| 11494 | SARS-CoV-2 spike prefusion STA | 8.60 | conformational_pair | **no** | **no** | **no** | **no** | — |
| 4940 | ClpB WT-1 (ATPgammaS) | 6.20 | conformational_pair | **no** | **no** | **no** | **no** | **no** |
| 4941 | ClpB WT-2A (ATPgammaS) | 4.00 | conformational_pair | **no** | **no** | **no** | **no** | **no** |
| 37504 | GLP-1R + Gs (ligand-free) | 2.54 | b_factor | **no** | **no** | **no** | **no** | **no** |
| 9156 | NMDA GluN1-GluN2A Extended-1 | 6.84 | b_factor | **no** | **no** | **no** | **no** | **no** |
| 48311 | Yeast V-ATPase Vo | 3.70 | b_factor | **no** | **no** | **no** | **no** | **no** |
| 11149 | Bovine ATP synthase Fo | 3.61 | b_factor | **no** | **no** | **no** | **no** | **no** |
| 42603 | Human p97/VCP + inhibitor | 3.23 | b_factor | **no** | **no** | **no** | **no** | **no** |
| 6287 | T20S proteasome | 2.80 | windowed_halfmap_correlation_only | **no** | **no** | **no** | **no** | — |
| 44471 | Human nucleosome 3.0 A | 3.00 | b_factor | **no** | **no** | **no** | **no** | **no** |
| 33736 | Tetrahymena ribozyme | 4.14 | _excluded_ | — | — | — | — | — |
| 33734 | SARS-CoV-2 spike K202B bispecific | 3.10 | b_factor | **no** | **no** | **no** | **no** | **no** |
| 62841 | Thermophile spliceosome ILS | 2.80 | b_factor | **no** | **no** | **no** | **no** | **no** |
| 45266 | 70S ribosome (EMPIAR-10499) | 5.40 | windowed_halfmap_correlation_only | **no** | **no** | **no** | **no** | — |
| 7130 | Epithelial Na+ channel (ENaC) | 3.90 | b_factor | **no** | **no** | **no** | **no** | **no** |

## Recommended commands (after data volume is mounted)

```bash
source .venv/bin/activate
export PYTHONUNBUFFERED=1

# 1. Finish LH pipeline for any missing maps
python scripts/run_cohort_pipeline.py --pending

# 2. BlocRes local-resolution maps (cohort-wide)
python scripts/run_blocres_local_resolution.py --all

# 3. Q-score validation (maps with deposited PDB)
python scripts/run_qscore_validation.py --all --cohort-summary --cohort-figure

# 4. Regenerate cohort figures + sync into docs/figures/
python scripts/rerun_all_figures.py
python scripts/sync_thesis_narrative_figures.py
```

Machine-readable run list: [`COHORT_RUN_LIST.sh`](./COHORT_RUN_LIST.sh) (regenerate with `--write-run-list`).
