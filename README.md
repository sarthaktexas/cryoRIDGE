# cryoem-halfmap-qc

[![DOI](https://zenodo.org/badge/1262218538.svg)](https://doi.org/10.5281/zenodo.20618526)

Python tools for **local map reliability** in cryo-EM reconstructions: density statistics, half-map reproducibility, windowed local FSC (Å), a reproducibility score (H_repro), build/caution/omit zones, and optional deposited-model B-factor checks.

The goal is to test whether inexpensive map features track **half-map cross-correlation** and **local FSC** well enough to guide modeling. This is **not** a claim that density alone defines molecular flexibility.

All volumes use NumPy 3D arrays in `(Z, Y, X)` order (section, row, column), consistent with typical `mrcfile` layouts.

---

## Install

**PyPI:** not published yet (`pip install cryoem-halfmap-qc` will fail until the first release is uploaded). See [Publishing to PyPI](#publishing-to-pypi) below.

Until then, install from GitHub or a local checkout:

```bash
git clone https://github.com/sarthaktexas/cryoem-halfmap-qc.git
cd cryoem-halfmap-qc
pip install -e .

# or without cloning:
pip install "git+https://github.com/sarthaktexas/cryoem-halfmap-qc.git@v0.3.1"
```

This installs the **`halfmap-qc`** command on your PATH (PyPI package name will be `cryoem-halfmap-qc` once published).

**Help & interactive mode:**

```bash
halfmap-qc                  # interactive menu (when run in a terminal)
halfmap-qc help             # full command reference
halfmap-qc --help           # argparse summary + examples
halfmap-qc cohort --help    # flags for one subcommand
halfmap-qc interactive      # menu explicitly
```

**Dependencies:** NumPy, SciPy, mrcfile, Matplotlib, gemmi (mmCIF/PDB for residue-level validation).

---

## Data layout

Cryo-EM maps are **not** stored in this repository (too large for git). After cloning, create local directories:

```text
data/emd_<ID>-<label>/     # deposited map + half-maps (.map or .mrc)
outputs/emd_<ID>/           # pipeline products (created by scripts)
pdb/                        # fitted models (mmCIF) — sample models included for the cohort
cohort/manifest.csv         # EMDB IDs, relative paths, contours, validation labels
```

Download deposited and half maps from [EMDB](https://www.ebi.ac.uk/emdb/) and fitted models from [PDBe](https://www.ebi.ac.uk/pdbe/). Use the depositor-recommended contour for each entry (listed in `cohort/manifest.csv`). See [docs/COHORT.md](docs/COHORT.md) for download status and pipeline progress.

---

## Quick start

Run from the project root (where `data/` and `cohort/manifest.csv` live).

**Single-map features:**

```bash
halfmap-qc features path/to/map.mrc --out map_features.npz --float32
# shorthand (legacy): halfmap-qc path/to/map.mrc --out map_features.npz --float32
```

**Typical workflow** (features on avg-of-halves; reliability MRCs on deposited primary grid):

```bash
EMD=49450
CONTOUR=0.116
DATA=data/emd_${EMD}-mgtA_e2p+e1

halfmap-qc analyze \
  --features "${DATA}/emd_${EMD}_avg_features_t0116.npz" \
  --half1 "${DATA}/emd_${EMD}_half_map_1.map" \
  --half2 "${DATA}/emd_${EMD}_half_map_2.map" \
  --reference "${DATA}/emd_${EMD}.map" \
  --contour "${CONTOUR}" \
  --out-dir "outputs/emd_${EMD}/analysis"

halfmap-qc reliability --emd-id "${EMD}" --contour "${CONTOUR}" \
  --features "${DATA}/emd_${EMD}_avg_features_t0116.npz" \
  --halfmap-npz "outputs/emd_${EMD}/analysis/halfmap_metrics.npz"
```

**Cohort batch** (all active manifest entries with local data):

```bash
halfmap-qc cohort --pending --skip-bfactor
```

**ARC / SLURM** (one map per array task; save a local `*.sbatch` — not in git):

```bash
# After pip install -e . and rsync data/ + cohort/manifest.csv to $SCRATCH/thesis
N=$(($(halfmap-qc cohort-ids | wc -l) - 1))
sbatch --account=wrz135 --array=0-${N} --cpus-per-task=4 --mem=32G --time=00:45:00 \
  --wrap='halfmap-qc cohort --emd-id $(halfmap-qc cohort-ids | sed -n "$((SLURM_ARRAY_TASK_ID+1))p") --skip-bfactor'
```

Or save a multi-line script as e.g. `~/halfmap-qc_array.sbatch` (gitignored) and `sbatch --array=0-${N} ~/halfmap-qc_array.sbatch`.

---

## CLI (`halfmap-qc`)

| Command | Purpose |
| --- | --- |
| *(no args, TTY)* | Interactive menu |
| `halfmap-qc help` | Full reference + install notes |
| `halfmap-qc features` | Local density / multiscale features → `.npz` |
| `halfmap-qc analyze` | Windowed half-map CC + feature correlations |
| `halfmap-qc reliability` | Reliability score, build zones, MRC export |
| `halfmap-qc cohort` | Batch pipeline from `cohort/manifest.csv` |
| `halfmap-qc cohort-ids` | Print EMDB IDs (for SLURM array jobs) |
| `halfmap-qc interactive` | Interactive menu (same as bare `halfmap-qc`) |

Legacy: `python -m cryoem_mrc` still works (same as `halfmap-qc features`).

## Publishing to PyPI

One-time setup:

1. Create an account at [pypi.org](https://pypi.org/account/register/) (and optionally [test.pypi.org](https://test.pypi.org/) for a dry run).
2. On PyPI → **Your projects** → **Add new project** → name it `cryoem-halfmap-qc` (or claim it when uploading).
3. On PyPI → **Account settings** → **Publishing** → **Add a new pending publisher**:
   - PyPI project: `cryoem-halfmap-qc`
   - Owner: `sarthaktexas` (your GitHub user/org)
   - Repository: `cryoem-halfmap-qc`
   - Workflow: `publish.yml`
   - Environment: (leave blank unless you use one)

Release:

```bash
# bump version in pyproject.toml first, then:
git add pyproject.toml cryoem_mrc/__init__.py
git commit -m "Release v0.3.1"
git tag v0.3.1
git push origin main --tags
```

On GitHub → **Releases** → **Draft a new release** → choose tag `v0.3.1` → **Publish release**. The [`.github/workflows/publish.yml`](.github/workflows/publish.yml) workflow builds the wheel and uploads to PyPI.

Test install after publish:

```bash
pip install cryoem-halfmap-qc
halfmap-qc --version
```

Manual upload (without GitHub Actions):

```bash
pip install build twine
python -m build
twine upload dist/*
```

## Scripts (thesis / optional)

Thesis figure runners (`scripts/rerun_all_figures.py`, `scripts/run_cohort_summary_figures.py`, Figma export scripts, etc.) and `cryoem_mrc/thesis_figures.py` are **local-only** (gitignored) like `figma-plugins/`. Clone the repo on a machine that already has those files, or keep a local copy from before they were untracked.

---

## Python API (high level)

```python
import numpy as np
from cryoem_mrc import load_full_and_half_maps, run_pipeline, half_map_local_metrics
from cryoem_mrc.reliability import compute_reliability_maps, classify_build_zones

bundle = load_full_and_half_maps(
    "full.mrc", "half1.mrc", "half2.mrc", dtype=np.float32, resample_if_needed=True
)
metrics = half_map_local_metrics(bundle.half1, bundle.half2, window=5)
# metrics["windowed_halfmap_correlation"], etc.

features = run_pipeline("map.mrc", use_float32=True)
reliability = compute_reliability_maps(
    bundle.half1, bundle.half2,
    density_normalized=features["density_normalized"],
    window=5,
)
zones = classify_build_zones(reliability["reliability_score"])
```

**Package modules:** `io`, `map_grid`, `local_stats`, `multiscale`, `half_map_repro`, `local_fsc`, `mechanics`, `reliability`, `analysis`, `structure_validation`. Path helpers: `cryoem_mrc/repo_paths.py`.

---

## Methods summary

- **Windowed half-map correlation** is the fast internal reproducibility target for feature validation; **local FSC resolution (Å)** is the field-standard reference.
- **Local FSC** is computed in-repo (`cryoem_mrc.local_fsc`); external BlocRes / ResMap / MonoRes maps are not loaded.
- **H_repro** is the windowed gradient-constraint map *V* (legacy export name; ranked as **reliability_score**); **reliability_score** is an in-mask percentile used for build/caution/omit terciles. Resolvability gating uses windowed half-map CC or local FSC, not a separate disagreement map.
- **Local variance** is often the strongest single feature predictor of windowed half-map correlation; treat B-factor correlations as exploratory and report partial correlations when comparing scores.

**Thesis prose:** full narrative draft in [docs/THESIS_NARRATIVE.md](docs/THESIS_NARRATIVE.md). Writing guide and defense notes in [docs/THESIS_AND_PUBLICATION.md](docs/THESIS_AND_PUBLICATION.md).

---

## Tests

```bash
python -m unittest discover -s tests -v
```

---

## Citation

**Before the manuscript is published**, cite the software with the Zenodo concept DOI (resolves to the latest release; pin `v0.2.3` or a commit hash for exact reproducibility):

```bibtex
@software{mohanty2026cryoem_halfmap_qc,
  author = {Mohanty, Sarthak},
  title = {cryoem-halfmap-qc: local map reliability from cryo-EM density and half-maps},
  year = {2026},
  doi = {10.5281/zenodo.20618526},
  url = {https://doi.org/10.5281/zenodo.20618526},
  version = {0.2.3}
}
```

GitHub also reads [CITATION.cff](CITATION.cff) for the **Cite this repository** button.

**After publication**, cite the paper as the primary reference. Also cite this Zenodo archive when you need the exact pipeline version used in the work.

When the manuscript exists, add a `preferred-citation` block to `CITATION.cff` (template included there) and drop the BibTeX for the article into this section.

## License

MIT License. See [LICENSE](LICENSE).
