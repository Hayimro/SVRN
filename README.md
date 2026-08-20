# SVRN — Stochastic Variational Relay Network

[![CI](https://github.com/<org>/svrn/actions/workflows/ci.yml/badge.svg)](https://github.com/<org>/svrn/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9–3.11](https://img.shields.io/badge/python-3.9%E2%80%933.11-blue)](pyproject.toml)

SVRN is a graph neural network for inferring **ligand–receptor mediated
cell–cell communication** from spatial transcriptomics data. It combines:

- a **Hill-kinetics interaction module** modeling cooperative ligand–receptor
  binding,
- **LR-aware gated attention** that fuses spatial context with LR
  co-expression,
- a **stochastic variational relay** that propagates a communication signal
  across the spatial graph over multiple hops, with an explicit
  reparameterized latent posterior and KL regularization,
- **edge-aware graph diffusion smoothing**, and
- a **multi-run, K-fold consensus-inference** procedure that yields
  reproducible per-cell-type influence scores, selection frequencies, and
  ranked communication corridors, with Monte-Carlo uncertainty estimates.
---

## Installation

`torch` and `torch-geometric` are hardware-specific (CPU vs. a particular
CUDA build), so install them first using the official instructions, then
install the rest of the dependencies.

### Option A — pip

```bash
python -m venv .venv && source .venv/bin/activate      # optional but recommended

# 1. Install PyTorch for your platform, e.g. CPU-only:
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu
# ...or for CUDA 12.1:
# pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121

# 2. Install torch-geometric (must match the torch build above):
pip install torch-geometric==2.5.3

# 3. Install SVRN and the remaining dependencies:
pip install -e .            # runtime only
pip install -e ".[dev]"     # + testing/linting tools
```

### Option B — conda / mamba

```bash
conda env create -f environment.yml
conda activate svrn
# adjust the `pytorch-cuda` line in environment.yml to match your GPU,
# or remove it entirely for a CPU-only environment
```

### Verify the install

```bash
python -m svrn --run_example
```

This generates a small synthetic AnnData object + LR table, runs the full
pipeline (preprocessing → training → Monte-Carlo uncertainty → plots) on
CPU in well under a minute, and writes results to `svrn_dummy_results/`.

---

## Quickstart — Python API

```python
from svrn import Config, SVRNPipeline

cfg = Config(
    DATA_PATH="data/your_dataset.h5ad",   # AnnData with adata.obsm["spatial"]
    LR_PATH="data/your_lr_pairs.csv",     # two-column ligand/receptor gene table
    OUTPUT_DIR="svrn_results",
    EPOCHS=100,
    HIDDEN_DIM=256,
    K_FOLDS=5,      # >=2 enables K-fold consensus inference; 0/1 disables it
    N_RUNS=5,       # independent seeded training runs pooled into consensus
    MC_SAMPLES=30,  # stochastic forward passes for uncertainty estimation
)

pipeline = SVRNPipeline(cfg)
pipeline.run()
```

## Quickstart — CLI

```bash
svrn \
  --data_path data/your_dataset.h5ad \
  --lr_path data/your_lr_pairs.csv \
  --output_dir svrn_results \
  --epochs 100 \
  --hidden_dim 256 \
  --k_folds 5 \
  --n_runs 5 \
  --mc_samples 30
```

(`python -m svrn ...` works identically.) See
[`scripts/run_full_pipeline.sh`](scripts/run_full_pipeline.sh) for a
copy-paste starting point, and [`scripts/run_example.sh`](scripts/run_example.sh)
for the synthetic-data smoke test.

## Project structure

```
svrn/
├── svrn/
│   ├── __init__.py         # public API (Config, SVRN, SVRNPipeline, ...)
│   ├── __main__.py         # `python -m svrn`
│   ├── utils.py            # Config, set_seed/get_device, metrics,
│   │                        # validation (SVRNValidator), consensus
│   │                        # aggregation (ConsensusInfluence)
│   ├── model.py             # network architecture: HillInteraction,
│   │                        # LRPAwareGatedAttention,
│   │                        # StochasticVariationalRelay,
│   │                        # GraphDiffusionSmoother, SVRN
│   ├── data.py              # data loading & preprocessing
│   │                        # (ScalableDataPreprocessor)
│   ├── visualization.py     # publication-quality plotting
│   │                        # (SVRNVisualizer, ConsensusPlotter)
│   └── pipeline.py          # SVRNPipeline orchestrator, `main()` CLI,
│                            # and the example_usage() smoke test
├── tests/
│   └── test_modules.py     # CPU-only unit tests for the core NN modules
├── scripts/
│   ├── run_example.sh      # synthetic-data smoke test
│   └── run_full_pipeline.sh
│ 
├── data/                   # local drop-zone for inputs (git-ignored)
├── requirements.txt        # pinned runtime dependencies
├── requirements-dev.txt    # + testing/linting/notebook tools
├── environment.yml         # conda equivalent
└── pyproject.toml          # packaging + console entry point
```

All public names are re-exported from `svrn/__init__.py`, so existing code
using `from svrn import Config, SVRNPipeline, ...` is unaffected by this
module split.

## Outputs

Each run writes to `--output_dir` (or `Config.OUTPUT_DIR`):

- `split_indices.npz` / `kfold_split_indices.npz` — cached, reusable train/val/test
  and K-fold partitions (spatial-KMeans-stratified) for exact reproducibility
  across separate process launches
- model checkpoints and a CSV training log (via PyTorch Lightning)
- per-cell influence scores with Monte-Carlo mean/std/95% CI
- consensus influence tables (per-cell-type percentile rank + selection
  frequency across all run × fold models) and ranked communication corridors
- evaluation metrics (AUROC, Spearman ρ, F1, Moran's I, Geary's C,
  balanced accuracy, influence entropy) and publication-style plots

---

## Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v --cov=svrn --cov-report=term-missing
```

CI (`.github/workflows/ci.yml`) runs these tests plus the synthetic smoke
test on every push/PR against CPU-only PyTorch.

## Reproducibility

`set_seed()` fixes every known RNG source (Python, NumPy, PyTorch CPU/CUDA,
PyTorch Lightning, `PYTHONHASHSEED`) and enables
`torch.use_deterministic_algorithms(True)`. `CUBLAS_WORKSPACE_CONFIG` is set
at import time, before any CUDA context is created, since it is only read
once at `libcublas` load time. Because spatial K-Means + `StratifiedKFold`
splitting is seeded but not bit-reproducible across separate process launches
(floating-point reduction order in multi-threaded clustering can flip a
handful of borderline cells), always reuse the saved `split_indices.npz` /
`kfold_split_indices.npz` (via `--split_path`/`--kfold_split_path`) rather
than recomputing splits when exact cross-run reproducibility matters.

## Contributing

Issues and pull requests are welcome. Please run `black svrn`, `isort svrn`,
and `pytest` before submitting a PR.
