#!/usr/bin/env bash
# Example invocation of the full SVRN training + consensus-inference
# pipeline against real data. Adjust paths and hyperparameters as needed.
set -euo pipefail
cd "$(dirname "$0")/.."

python -m svrn \
  --data_path data/your_dataset.h5ad \
  --lr_path data/your_lr_pairs.csv \
  --output_dir svrn_results \
  --epochs 100 \
  --batch_size 100 \
  --hidden_dim 256 \
  --k_folds 5 \
  --n_runs 5 \
  --mc_samples 30
