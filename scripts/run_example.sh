#!/usr/bin/env bash
# Quick smoke test: generates synthetic data and runs the full SVRN
# pipeline (preprocessing -> training -> MC uncertainty -> plots) end to
# end on CPU in a few seconds. Useful to confirm a fresh install works.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m svrn --run_example
echo "Done. See ./svrn_dummy_results/ for outputs."
