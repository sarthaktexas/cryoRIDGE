# cryoem-halfmap-qc

[![PyPI version](https://img.shields.io/pypi/v/cryoem-halfmap-qc)](https://pypi.org/project/cryoem-halfmap-qc/)
[![DOI](https://zenodo.org/badge/1262218538.svg)](https://doi.org/10.5281/zenodo.20618526)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Local **reliability scores** and **build zones** for cryo-EM maps from half-maps and density features. Python **3.10+**.

```bash
pip install cryoem-halfmap-qc
halfmap-qc          # interactive menu
halfmap-qc help     # CLI reference
```

## Pipeline

Three commands, in order. Pass your own map paths and output directories.

```bash
# 1. Features from a map (.mrc / .map)
halfmap-qc features map.mrc --out features.npz --float32

# 2. Half-map metrics + correlations
halfmap-qc analyze \
  --features features.npz \
  --half1 half1.map --half2 half2.map \
  --reference ref.map \
  --contour CONTOUR \
  --out-dir analysis_out

# 3. Reliability score + build-zone MRCs
halfmap-qc reliability \
  --reference ref.map --half1 half1.map --half2 half2.map \
  --features features.npz \
  --halfmap-npz analysis_out/halfmap_metrics.npz \
  --contour CONTOUR \
  --out-dir reliability_out
```

`--contour` is the density threshold for the analysis mask (same value in steps 2 and 3). Optional: `--local-res path.mrc` on reliability for a local-resolution comparison figure.

**Outputs:** `reliability_score` and omit / caution / build zones as MRC overlays on your reference grid, plus `.npz` and summary JSON under `--out-dir`.

Flag details: `halfmap-qc analyze --help`, `halfmap-qc reliability --help`.

## HPC (ARC)

Default login `python3` is too old (3.6). Conda modules load on **compute nodes only**:

```bash
srun -p compute1 -n 1 -t 02:00:00 --pty bash
module load miniconda/24.4.0
conda activate halfmap-qc    # after one-time: conda create -n halfmap-qc python=3.12 -y && pip install cryoem-halfmap-qc
```

Put `module load` + `conda activate` in every `sbatch` script.

If install fails with empty `(from versions:)`, check `python --version` (need ≥3.10) and `pip install -U pip`.

## Citation

```bibtex
@software{mohanty2026cryoem_halfmap_qc,
  author = {Mohanty, Sarthak},
  title = {cryoem-halfmap-qc: local map reliability from cryo-EM density and half-maps},
  year = {2026},
  doi = {10.5281/zenodo.20618526},
  url = {https://doi.org/10.5281/zenodo.20618526},
  version = {0.5.0}
}
```

MIT — see [LICENSE](LICENSE).
