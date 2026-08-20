# SVRN — Stochastic Variational Relay Network

[![CI](https://github.com/<org>/svrn/actions/workflows/ci.yml/badge.svg)](https://github.com/<org>/svrn/actions/workflows/ci.yml)
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


## Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v --cov=svrn --cov-report=term-missing
```

CI (`.github/workflows/ci.yml`) runs these tests plus the synthetic smoke
test on every push/PR against CPU-only PyTorch.
