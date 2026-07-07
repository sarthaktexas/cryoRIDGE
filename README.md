# cryoRIDGE

**RIDGE** = **R**eliability **I**nferred from **D**ensity **G**radient **E**nergy

[![PyPI version](https://img.shields.io/pypi/v/cryoridge)](https://pypi.org/project/cryoridge/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Local **reliability scores** and **build zones** for cryo-EM maps from half-maps and density features. Python **3.10+**.

<img width="772" height="681" alt="image" src="https://github.com/user-attachments/assets/d69251e6-bac9-4cb1-af28-5bacbad0b9f5" />

```bash
pip install cryoridge
cryoridge          # interactive: two half-map paths → two MRC outputs
cryoridge help     # CLI reference
```

Interactive mode asks for **half-map 1** and **half-map 2**, prompts you to set the contour in **ChimeraX**, warns when estimated resolution is outside the model-building band (>4 Å), then writes `{stem}_reliability.mrc` and `{stem}_build_zones.mrc` under `cryoridge_out/` next to half-map 1.

## Pipeline (non-interactive / advanced)

Three commands, in order. Pass your own map paths and output directories.

```bash
# 1. Features from a map (.mrc / .map)
cryoridge features map.mrc --out features.npz --float32

# 2. Half-map metrics + correlations
cryoridge analyze \
  --features features.npz \
  --half1 half1.map --half2 half2.map \
  --reference ref.map \
  --contour CONTOUR \
  --out-dir analysis_out

# 3. Reliability score + build-zone MRCs
cryoridge reliability \
  --reference ref.map --half1 half1.map --half2 half2.map \
  --features features.npz \
  --contour CONTOUR \
  --out-dir reliability_out
```

`--contour` is the density threshold for the analysis mask (same value in steps 2 and 3). Step 2 (analyze) is optional if you only need the reliability and build-zone MRCs.

**Reliability outputs:** `{label}_reliability.mrc` and `{label}_build_zones.mrc` on your reference grid.

Flag details: `cryoridge features --help`, `cryoridge analyze --help`, `cryoridge reliability --help`.

## HPC (ARC)

Default login `python3` is too old (3.6). Conda modules load on **compute nodes only**:

```bash
srun -p compute1 -n 1 -t 02:00:00 --pty bash
module load miniconda/24.4.0
conda activate cryoridge    # after one-time: conda create -n cryoridge python=3.12 -y && pip install cryoridge
```

**Batch job** — copy `scripts/cryoridge_cluster.sbatch.example`, set `MAP`, `HALF1`, `HALF2`, `REF`, `CONTOUR`, and submit. Each step aborts if the previous one fails.

```bash
cp scripts/cryoridge_cluster.sbatch.example run_my_map.sbatch
# edit paths in run_my_map.sbatch
sbatch run_my_map.sbatch
```

If install fails with empty `(from versions:)`, check `python --version` (need ≥3.10) and `pip install -U pip`.

## Citation

```bibtex
@software{mohanty2026cryoridge,
  author = {Mohanty, Sarthak},
  title = {cryoRIDGE: Reliability Inferred from Density Gradient Energy},
  year = {2026},
  doi = {10.5281/zenodo.20618526},
  url = {https://doi.org/10.5281/zenodo.20618526},
  version = {0.8.3}
}
```

MIT — see [LICENSE](LICENSE).
