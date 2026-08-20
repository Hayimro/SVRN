"""
SVRN — Stochastic Variational Relay Network
============================================

A graph neural network for modeling ligand-receptor mediated cell-cell
communication in spatial transcriptomics data.

Typical usage
-------------
    from svrn import Config, SVRNPipeline

    cfg = Config(
        DATA_PATH="path/to/adata.h5ad",
        LR_PATH="path/to/lr_pairs.csv",
        OUTPUT_DIR="svrn_results",
    )
    pipeline = SVRNPipeline(cfg)
    pipeline.run()

See README.md for the full CLI and Python API documentation.

Package layout
--------------
The implementation is split across four submodules (previously a
single monolithic ``pipeline.py``):

- :mod:`svrn.utils`          -- Config, reproducibility helpers, metrics,
                                 validation, and consensus aggregation
- :mod:`svrn.model`          -- the SVRN network architecture
- :mod:`svrn.data`           -- data loading and preprocessing
- :mod:`svrn.visualization`  -- publication-quality plotting
- :mod:`svrn.pipeline`       -- the SVRNPipeline orchestrator, CLI, and
                                 example_usage() smoke test

Everything below is re-exported at the top level so existing imports
(``from svrn import ...``) keep working unchanged.
"""

from .utils import (
    Config,
    UnifiedMetrics,
    SVRNValidator,
    ConsensusInfluence,
    set_seed,
    get_device,
)
from .model import (
    SVRN,
    HillInteraction,
    LRPAwareGatedAttention,
    StochasticVariationalRelay,
    GraphDiffusionSmoother,
)
from .data import ScalableDataPreprocessor
from .visualization import SVRNVisualizer, ConsensusPlotter
from .pipeline import SVRNPipeline, example_usage, main

__version__ = "1.1.0"

__all__ = [
    "Config",
    "SVRN",
    "SVRNPipeline",
    "ScalableDataPreprocessor",
    "UnifiedMetrics",
    "SVRNVisualizer",
    "SVRNValidator",
    "ConsensusInfluence",
    "ConsensusPlotter",
    "HillInteraction",
    "LRPAwareGatedAttention",
    "StochasticVariationalRelay",
    "GraphDiffusionSmoother",
    "set_seed",
    "get_device",
    "example_usage",
    "main",
    "__version__",
]
