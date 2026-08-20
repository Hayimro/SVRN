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

See [`docs/algorithms.md`](docs/algorithms.md) for full pseudocode of the
training procedure (Algorithm 1) and the consensus-inference procedure
(Algorithm 2).

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

### Key CLI arguments

| Flag | Default | Description |
|---|---|---|
| `--data_path` | — | Path to `.h5ad` spatial transcriptomics file (required unless `--run_example`) |
| `--lr_path` | — | Path to ligand–receptor CSV file (required unless `--run_example`) |
| `--output_dir` | `svrn_results` | Output directory for checkpoints, tables, and plots |
| `--epochs` | `100` | Training epochs per model |
| `--batch_size` | `100` | Mini-batch size |
| `--lr` | `3e-4` | Learning rate (CLI default; the Python `Config` default is `1.5e-4`) |
| `--hidden_dim` | `256` | Node encoder hidden dimension (CLI default; the Python `Config` default is `512`) |
| `--max_hops` | `5` | Relay propagation hops (`S` in Algorithm 1, Eq. 10) |
| `--k_folds` | `5` | K-fold cross-validation folds (`0`/`1` disables; `25` recommended for publication-grade consensus estimates) |
| `--n_runs` | `5` | Independent seeded training runs pooled into the consensus (20–25 recommended for ±10% selection-frequency precision) |
| `--consensus_k` | `2` | Top-K cell types per fold flagged as "selected" |
| `--mc_samples` | `30` | Monte-Carlo forward passes for uncertainty estimation (the Python `Config` default is `50`) |
| `--n_neighbors` | `14` | Spatial KNN graph degree (memory control) |
| `--edge_chunk_size` | `512` | Number of edges processed per LR encoder chunk (memory control) |
| `--max_edges_per_step` | `4000` | Edge subsampling cap per training step (memory control; `0` = use all) |
| `--split_path` / `--kfold_split_path` | auto | Reuse previously saved split indices for exact reproducibility across runs |
| `--run_example` | off | Ignore `--data_path`/`--lr_path` and run the synthetic smoke test |

Run `svrn --help` for the complete list.

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
├── docs/
│   ├── algorithms.md       # pseudocode: training + consensus inference
│   ├── MODEL_AND_CODE_AVAILABILITY.md
│   ├── DATA_AVAILABILITY.md
│   ├── REPRODUCIBILITY.md
│   └── data_provenance_template.csv
├── data/                   # local drop-zone for inputs (git-ignored)
├── requirements.txt        # pinned runtime dependencies
├── requirements-dev.txt    # + testing/linting/notebook tools
├── environment.yml         # conda equivalent
├── pyproject.toml          # packaging + console entry point
├── CITATION.cff
└── LICENSE
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

---

## Nature Communications reproducibility and availability

SVRN is custom computational code central to the reported method. The repository therefore includes installation instructions, an executable synthetic-data example, automated tests, algorithm documentation, reproducibility guidance, and data/model provenance templates. This structure follows Nature Communications guidance that custom code central to the conclusions should be available to editors and reviewers and that documentation should include installation/running instructions, tests and examples.

See:

- [`docs/MODEL_AND_CODE_AVAILABILITY.md`](docs/MODEL_AND_CODE_AVAILABILITY.md) for code/model release and archival requirements;
- [`docs/DATA_AVAILABILITY.md`](docs/DATA_AVAILABILITY.md) for dataset provenance and manuscript Data availability wording;
- [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for environment, preprocessing, split, validation and release requirements.

The final manuscript should cite the exact archived software release (preferably with a DOI) and state any restrictions on code, data, or trained model access.

### Manuscript Code availability statement template

> **Code availability**  The source code for SVRN is publicly available at `https://github.com/<org>/svrn` and the exact version used in this study is archived at [DOI]. The repository contains the model implementation, preprocessing and inference pipeline, training configuration, tests and reproducible synthetic example. Any restrictions on access to the study-specific data or trained model checkpoints are described separately in the Data availability statement.

## Citation

If you use SVRN in your research, please cite it (see
[`CITATION.cff`](CITATION.cff)):

```bibtex
@software{svrn2026,
  title   = {SVRN: Stochastic Variational Relay Network for Cell-Cell
             Communication Inference in Spatial Transcriptomics},
  author  = {{SVRN Authors}},
  year    = {2026},
  url     = {https://github.com/<org>/svrn},
  version = {1.0.0}
}
```

## License

MIT — see [`LICENSE`](LICENSE). Update the copyright holder name in
`LICENSE`, `CITATION.cff`, and `pyproject.toml` before publishing.

## Contributing

Issues and pull requests are welcome. Please run `black svrn`, `isort svrn`,
and `pytest` before submitting a PR.
