# Thesis Grid Schematic (Figma plugin)

Schematic thesis pipeline figure from synthetic 8×8 grids.

## Install

1. Figma Desktop → **Plugins → Development → Import plugin from manifest…**
2. Choose `figma-plugins/thesis-grid-schematic/manifest.json`.

## Generate

1. Run the plugin, set controls, click **Generate pipeline**.
2. Fixed flow:

```
Protein + slice → ρ slice → h₁,h₂ → ρ̄ → V → Reliability → Build zones
```

## Controls

| Control | Purpose |
|---------|---------|
| Grid size | Default **8×8** (4–16) |
| Seed | Reproducible layout; empty = random |
| Archetype | Tetramer, ion channel, membrane protein, or auto |
| Disagreement | Half-map split in flexible region (0–1) |
| Slope mesh | Isometric wireframe for h₁, h₂, ρ̄ (pipeline), or **from any selected grid** |

**Mesh from grid:** select a grid frame → **Generate mesh from selection**. Cell values are min–max normalized to 0–1 (lowest = base, highest = peak); wireframe slope follows the normalized surface.

**Grid operation:** select one grid (e.g. ρ̄) → pick operation → **Apply to selection**. Chains manually for the tight abstract figure:

| Step | Select | Operation |
|------|--------|-----------|
| V | ρ̄ | Constraint V |
| Reliability | V | Reliability score |
| Build zones | reliability | Build zones |

Mask threshold (default 0.12) applies to **Constraint V** on density maps. Reliability and build zones rank cells with value &gt; 0.

Install [Geist Mono](https://vercel.com/font) in Figma if cell labels fail to render.

## Color scales

| Map | Encoding |
|-----|----------|
| Half-maps | Gray + blue (h₁) / red (h₂) tint |
| V | Pale → deep cyan |
| Reliability | Pale → deep green (percentile rank) |
| Build zones | Red (omit) · amber (caution) · green (build) |

Slope mesh: one uniform-stroke vector with cubic-bezier row/column curves (4× upsample + smooth, 0–1 normalized).
