# cryoem-halfmap-qc

[![PyPI version](https://img.shields.io/pypi/v/cryoem-halfmap-qc)](https://pypi.org/project/cryoem-halfmap-qc/)
[![DOI](https://zenodo.org/badge/1262218538.svg)](https://doi.org/10.5281/zenodo.20618526)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Half-map reproducibility and **local reliability scores** for cryo-EM density maps: density features, windowed half-map correlation, **reliability_score**, and build / caution / omit zones.

**Requires Python 3.10+** (`python --version`).

Install the **`halfmap-qc`** command:

```bash
python3 -m pip install -U pip
pip install cryoem-halfmap-qc
halfmap-qc --version
```

From source:

```bash
pip install "git+https://github.com/sarthaktexas/cryoem-halfmap-qc.git@v0.3.3"
# or: git clone … && pip install -e .
```

---

## Quick start

Run from a working directory with your maps (and optionally `cohort/manifest.csv` for batch mode).

**Interactive menu** (terminal):

```bash
halfmap-qc
halfmap-qc help
```

**Single map — features:**

```bash
halfmap-qc features map.mrc --out features.npz --float32
```

**Half-map analysis + reliability export:**

```bash
halfmap-qc analyze \
  --features features.npz \
  --half1 half1.map --half2 half2.map \
  --reference deposited.map --contour 0.116 \
  --out-dir outputs/analysis

halfmap-qc reliability --emd-id 49450 --contour 0.116 \
  --features features.npz \
  --halfmap-npz outputs/analysis/halfmap_metrics.npz
```

**Cohort batch** (uses `cohort/manifest.csv` + local `data/` paths):

```bash
halfmap-qc cohort --pending
halfmap-qc cohort --emd-id 49450
halfmap-qc cohort-ids    # EMDB IDs for SLURM array jobs
```

---

## CLI

| Command | Purpose |
| --- | --- |
| `halfmap-qc` | Interactive menu (TTY) |
| `halfmap-qc help` | Command reference |
| `halfmap-qc features` | Local density / multiscale features → `.npz` |
| `halfmap-qc analyze` | Windowed half-map CC + feature correlations |
| `halfmap-qc reliability` | Reliability score, build zones, MRC export |
| `halfmap-qc cohort` | Batch pipeline from manifest |
| `halfmap-qc cohort-ids` | List cohort EMDB IDs |

Per-command flags: `halfmap-qc cohort --help`, etc.

---

## Data layout

Maps are not stored in this repository. Typical layout:

```text
data/emd_<ID>-<label>/     # deposited map + half-maps
outputs/emd_<ID>/           # pipeline outputs (created by halfmap-qc)
cohort/manifest.csv         # optional batch manifest
```

Download maps from [EMDB](https://www.ebi.ac.uk/emdb/). Use each entry's depositor-recommended contour in the manifest.

---

## Python API

```python
from cryoem_mrc import load_full_and_half_maps, run_pipeline, half_map_local_metrics
from cryoem_mrc.reliability import compute_reliability_maps, classify_build_zones

bundle = load_full_and_half_maps(
    "deposited.map", "half1.map", "half2.map",
    dtype="float32", resample_if_needed=True,
)
metrics = half_map_local_metrics(bundle.half1.data, bundle.half2.data, window=5)

features = run_pipeline("avg.map", use_float32=True)
delta = bundle.half1.data - bundle.half2.data
rel = compute_reliability_maps(
    features["density_normalized"], delta, window=5, mask=contour_mask
)
zones = classify_build_zones(rel["reliability_score"], contour_mask)
```

Main modules: `io`, `map_grid`, `local_stats`, `half_map_repro`, `reliability`, `analysis`.

---

## What the scores mean

- **Windowed half-map correlation** — local Pearson correlation between half-maps in a sliding window (fast reproducibility proxy).
- **reliability_score** — in-mask percentile rank of the constraint term *V* (gradient-based smoothness vs half-map disagreement); higher = more reliable for modeling.
- **Build zones** — tercile labels inside the density contour: omit / caution / build.

External local-resolution maps (BlocRes, ResMap) are optional comparison inputs, not required to run the pipeline.

---

## Tests

```bash
python -m unittest discover -s tests -q
```

---

## Citation

```bibtex
@software{mohanty2026cryoem_halfmap_qc,
  author = {Mohanty, Sarthak},
  title = {cryoem-halfmap-qc: local map reliability from cryo-EM density and half-maps},
  year = {2026},
  doi = {10.5281/zenodo.20618526},
  url = {https://doi.org/10.5281/zenodo.20618526},
  version = {0.3.3}
}
```

See also [CITATION.cff](CITATION.cff) for GitHub's **Cite this repository** button.

---

## Troubleshooting install

**`No matching distribution found` / `from versions:` empty**

1. **Check Python version** (most common cause):

   ```bash
   python3 --version   # must be 3.10 or newer
   ```

   macOS `/usr/bin/python3` is often **3.9** — it cannot install this package. Use Homebrew `python3.12`, `pyenv`, a venv, or on ARC `module load python/3.11`.

2. **Upgrade pip** (old pip hides available versions):

   ```bash
   python3 -m pip install -U pip
   ```

3. **HPC / offline index** — point at PyPI explicitly:

   ```bash
   pip install cryoem-halfmap-qc -i https://pypi.org/simple
   ```

4. **No PyPI access on compute nodes** — install on a login node, or from git:

   ```bash
   pip install "git+https://github.com/sarthaktexas/cryoem-halfmap-qc.git@v0.3.3"
   ```

Verify the package exists: https://pypi.org/project/cryoem-halfmap-qc/

---

## License

MIT — see [LICENSE](LICENSE).
