
[![CI](https://github.com/<org>/svrn/actions/workflows/ci.yml/badge.svg)](https://github.com/<org>/svrn/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9–3.11](https://img.shields.io/badge/python-3.9%E2%80%933.11-blue)](pyproject.toml)


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
|── docs/
│   ├── algorithms.md       # pseudocode: training + consensus inference
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
└──LICENSE
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


## Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v --cov=svrn --cov-report=term-missing
```

CI (`.github/workflows/ci.yml`) runs these tests plus the synthetic smoke
test on every push/PR against CPU-only PyTorch.

```bibtex
@software{svrn2026,
  title   = {SVRN: Stochastic Variational Relay Network for Cell-Cell
             Communication Inference in Spatial Transcriptomics},
  author  = {{Hayimro Edemealem Merie, Zenebe Markos Lonseko, Helen Haile Hayeso, Dingcan Hu, Nini ‎Rao‎}},
  year    = {2026},
  url     = {https://github.com/Hayimro/SVRN},
  version = {1.0.0}
}
```
