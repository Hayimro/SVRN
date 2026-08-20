# -*- coding: utf-8 -*-
"""
svrn.utils
==========
Cross-cutting support code that doesn't belong to the data, model, or
visualization layers: reproducibility helpers, the global :class:`Config`
dataclass, evaluation metrics (:class:`UnifiedMetrics`), the post-hoc
model validation suite (:class:`SVRNValidator`), and multi-run/K-fold
consensus aggregation (:class:`ConsensusInfluence`).

"""

from __future__ import annotations

import os
import sys
import json

# ---------------------------------------------------------------
# REPRODUCIBILITY: env vars MUST be set before any CUDA import.
# CUBLAS_WORKSPACE_CONFIG is read once at libcublas load time;
# setting it after torch is imported has no effect. Because this
# module is imported first by ``svrn/__init__.py``, this is the
# single place in the package where these env vars are set.
# ---------------------------------------------------------------
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import random
import warnings
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Tuple, TYPE_CHECKING

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

import pytorch_lightning as pl

from scipy.sparse import csr_matrix, issparse
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, average_precision_score, r2_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torch_geometric.loader import DataLoader as PyGDataLoader

if TYPE_CHECKING:
    # Only needed for the ``model: SVRN`` type hint on SVRNValidator; kept
    # behind TYPE_CHECKING to avoid a runtime circular import (model.py
    # imports Config from this module).
    from .model import SVRN


def set_seed(seed: int = 42) -> None:
    """Fix ALL known RNG sources for full cross-run reproducibility.

    Call order matters: this must run before any model weight init,
    DataLoader construction, or forward pass.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
   
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Hash-based ops (e.g. torch.unique) use this env var for determinism.
    os.environ["PYTHONHASHSEED"] = str(seed)
    pl.seed_everything(seed, workers=True)

_DEFAULT_SEED = 42
set_seed(_DEFAULT_SEED)

def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        device = torch.device("cpu")
        print("⚠️ CUDA not available. Using CPU.")
    return device


DEVICE = get_device()


@dataclass

class Config:
    DATA_PATH: str
    LR_PATH: str
    OUTPUT_DIR: str

    EPOCHS: int = 100
    BATCH_SIZE: int = 100
    LR: float = 1.5e-4
    HIDDEN_DIM: int = 512
    MULTI_HOP_STEPS: int = 5
    DROPOUT: float = 0.1
    SEED: int = 42

    MIN_CELLS: int = 20
    MIN_GENES: int = 10
    N_HVGS: int = 1122

    # Memory controls
    N_NEIGHBORS: int = 14
    EDGE_CHUNK_SIZE: int = 64
    MAX_EDGES_PER_STEP: int = 4000

    # Monte-Carlo uncertainty estimation (number of stochastic forward passes)
    MC_SAMPLES: int = 50

    N_GENES: int = 0
    N_CELLS: int = 0
    N_LR: int = 0
    DEVICE: str = "cpu"

    CT_PRIOR: Optional[torch.Tensor] = None
    CT_TYPE_TO_IDX: Optional[dict] = None  # maps cell-type string → integer index
    N_CT: int = 0  # number of unique cell types (set by preprocessor)

    VAL_RATIO: float = 0.15
    TEST_RATIO: float = 0.15

    # Species for mygene pathway annotation in C9 plot ("mouse" or "human")
    SPECIES: str = "mouse"

    K_FOLDS: int = 5

    N_RUNS: int = 5

    # Top-K cell types counted as "selected" when computing selection frequency
    # in ConsensusInfluence (K=2 → top-2 cell types per run are flagged)
    CONSENSUS_K: int = 2

    # Path to a saved split (.npz).  Empty string = compute & save a new one.
    SPLIT_PATH: str = ""

    KFOLD_SPLIT_PATH: str = ""

    def __post_init__(self):
        self.DEVICE = DEVICE.type

        def _normalize(path: str) -> str:
            return os.path.abspath(os.path.expanduser(path)) if path else path

        self.DATA_PATH = _normalize(self.DATA_PATH)
        self.LR_PATH = _normalize(self.LR_PATH)
        self.OUTPUT_DIR = _normalize(self.OUTPUT_DIR)
        self.SPLIT_PATH = _normalize(self.SPLIT_PATH)
        self.KFOLD_SPLIT_PATH = _normalize(self.KFOLD_SPLIT_PATH)

        try:
            os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        except OSError as exc:
            raise OSError(
                f"Could not create OUTPUT_DIR={self.OUTPUT_DIR!r}: {exc}. "
                f"Check that the parent directory exists and is writable."
            ) from exc

        for path, label in ((self.DATA_PATH, "DATA_PATH/--data_path"),
                             (self.LR_PATH, "LR_PATH/--lr_path")):
            if not path:
                raise ValueError(f"{label} is required and was not provided.")
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"{label} does not point to an existing file: {path!r}. "
                    f"Check the path (and that it is a file, not a directory)."
                )

        # ── Optional cache files: validate only if explicitly supplied ─
        for path, label, hint in (
            (self.SPLIT_PATH, "SPLIT_PATH/--split_path",
             "Leave it empty to compute and save a fresh split."),
            (self.KFOLD_SPLIT_PATH, "KFOLD_SPLIT_PATH/--kfold_split_path",
             "Leave it empty to compute and save fresh k-fold splits."),
        ):
            if path and not os.path.isfile(path):
                raise FileNotFoundError(
                    f"{label} does not point to an existing file: {path!r}. {hint}"
                )

    def save_json(self, path: str) -> None:
        d = asdict(self)
        # CT_PRIOR is a Tensor; CT_TYPE_TO_IDX may contain non-JSON types – skip both.
        d.pop("CT_PRIOR", None)
        d.pop("CT_TYPE_TO_IDX", None)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2)

# =====================================================================
#  Hill Interaction
# =====================================================================

class UnifiedMetrics:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def compute_all(
        self,
        influence_scores: np.ndarray,
        spatial_coords: np.ndarray,
        cell_types: np.ndarray,
        adj_matrix: csr_matrix,
    ) -> Dict[str, float]:
        # Winsorise [2nd, 98th pct] then min-max to [0, 1].
        # All metrics receive y_proc so that spatially-concentrated outlier
        # cells (high-influence hotspots present in some folds but not others)
        # do not dominate any single statistic and inflate cross-fold CV.
        y_proc = self._preprocess(influence_scores)

        metrics = {}

        try:
            metrics["morans_i"] = self.morans_i(
                y_proc,
                adj_matrix,
            )
        except Exception as e:
            print(f"Could not compute Moran's I: {e}")
            metrics["morans_i"] = np.nan

        try:
            metrics["gearys_c"] = self.gearys_c(
                y_proc,
                adj_matrix,
            )
        except Exception as e:
            print(f"Could not compute Geary's C: {e}")
            metrics["gearys_c"] = np.nan

        metrics["gini"] = self.gini_coefficient(y_proc)

        metrics["influence_entropy"] = self.shannon_entropy(y_proc)

        try:
            spec = self.pathway_specificity(y_proc, cell_types)
            metrics["tau_specificity"] = spec["tau_specificity"]
            metrics["js_specificity"]  = spec["js_specificity"]
        except Exception as e:
            print(f"Could not compute pathway specificity: {e}")
            metrics["tau_specificity"] = np.nan
            metrics["js_specificity"]  = np.nan

        try:
            y_rank = self._rank_transform(y_proc)
            metrics["r2_cell_type"] = self.r2_cell_type(
                y_rank,
                cell_types,
            )
        except Exception as e:
            print(f"Could not compute R²: {e}")
            metrics["r2_cell_type"] = np.nan

        return metrics

    # ------------------------------------------------------------------
    # Shared preprocessing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_normalize(W: csr_matrix) -> csr_matrix:
        """Row-normalise W so every row sums to 1.

        The original (n/W_sum) prefactor is fold-dependent: W_sum changes
        every time the adjacency is sliced (different cells = different KNN
        edges).  Row-normalisation makes W_sum == n exactly, collapsing the
        prefactor to 1 and removing this variance source entirely.

        Isolated nodes (row-sum == 0) are left as zero rows.
        """
        row_sums = np.asarray(W.sum(axis=1), dtype=np.float64).flatten()
        row_sums[row_sums < 1e-12] = 1.0
        inv = csr_matrix(
            (1.0 / row_sums,
             (np.arange(len(row_sums)), np.arange(len(row_sums)))),
            shape=(len(row_sums), len(row_sums)),
        )
        return inv.dot(W)

    @staticmethod
    def _preprocess(y: np.ndarray) -> np.ndarray:
        """Winsorise at [2nd, 98th] percentile, then min-max to [0, 1].

        Applied once at the top of compute_all and passed to every metric,
        so all statistics see the same outlier-robust score representation.

        **Why this reduces CV for all metrics**

        Spatial KMeans folds are contiguous tissue regions.  Some regions
        contain spatially concentrated high-influence cell clusters (hot-
        spots); others do not.  Without winsorising, these extreme cells
        dominate:

        - gini / tau / js: one extreme cell can double the per-cell-type
          mean, spiking Tau and JS specificity for that fold only.
        - morans_i / gearys_c: the large z = y - ȳ term from an outlier
          cell inflates the quadratic form z′W_rn z unpredictably.

        Winsorising at [2, 98] caps the maximum single-cell contribution
        to the 98th-percentile value, making all metrics insensitive to
        whether the top ~2 % of cells land in a given fold.

        Min-max after winsorising re-establishes a [0, 1] range so that
        Gini and specificity indices remain on their standard scales.
        """
        y = np.asarray(y, dtype=np.float64).flatten()
        lo, hi = np.percentile(y, [2, 98])
        y_w = np.clip(y, lo, hi)
        y_range = y_w.max() - y_w.min()
        if y_range < 1e-12:
            return np.zeros_like(y_w)
        return (y_w - y_w.min()) / y_range

    @staticmethod
    def _rank_transform(y: np.ndarray) -> np.ndarray:
        """Convert to fractional ranks in (0, 1].

        Called on the already-winsorised output of _preprocess(), so the
        [2, 98] clip is not repeated.  Fractional ranking makes morans_i
        and gearys_c scale-invariant and ensures they are exact duals:
        C = (n-1)/n * (1 - I).
        """
        from scipy.stats import rankdata
        n = len(y)
        return rankdata(y, method="average") / max(n - 1, 1)

    # ------------------------------------------------------------------
    # Spatial autocorrelation statistics
    # ------------------------------------------------------------------

    @staticmethod
    def morans_i(
        y: np.ndarray,
        adj_matrix: csr_matrix,
    ) -> float:
        y = np.asarray(y, dtype=np.float64).flatten()
        W = adj_matrix.tocsr() if issparse(adj_matrix) else csr_matrix(adj_matrix)
        n = len(y)
        if n < 2:
            return 0.0
        y_r  = UnifiedMetrics._rank_transform(y)
        W_rn = UnifiedMetrics._row_normalize(W)
        z = y_r - np.mean(y_r)
        denom = float(z @ z)
        if denom <= 1e-12:
            return 0.0
        return float(z @ (W_rn @ z)) / denom

    @staticmethod
    def gearys_c(
        y: np.ndarray,
        adj_matrix: csr_matrix,
    ) -> float:
       
        y = np.asarray(y, dtype=np.float64).flatten()
        W = adj_matrix.tocsr() if issparse(adj_matrix) else csr_matrix(adj_matrix)
        n = len(y)
        if n < 2:
            return 1.0
        y_r  = UnifiedMetrics._rank_transform(y)
        W_rn = UnifiedMetrics._row_normalize(W)
        z = y_r - np.mean(y_r)
        denom = float(z @ z)
        if denom <= 1e-12:
            return 1.0
        I = float(z @ (W_rn @ z)) / denom
        return float((n - 1) / n * (1.0 - I))

    @staticmethod
    def de_effect_size(
        influence_scores: np.ndarray,
        expression_matrix: np.ndarray,
        high_quantile: float = 0.75,
        low_quantile: float = 0.25,
    ) -> np.ndarray:
      
        scores = np.asarray(influence_scores, dtype=np.float64).flatten()

        if issparse(expression_matrix):
            expr = expression_matrix.toarray().astype(np.float64)
        else:
            expr = np.asarray(expression_matrix, dtype=np.float64)

        n_cells, n_genes = expr.shape
        if len(scores) != n_cells:
            raise ValueError(
                f"influence_scores length ({len(scores)}) does not match "
                f"expression_matrix rows ({n_cells})."
            )

        thr_hi = np.quantile(scores, high_quantile)
        thr_lo = np.quantile(scores, low_quantile)

        mask_hi = scores >= thr_hi
        mask_lo = scores <= thr_lo

        if mask_hi.sum() < 2 or mask_lo.sum() < 2:
            return np.zeros(n_genes, dtype=np.float64)

        mu_high = expr[mask_hi].mean(axis=0)   # (G,)
        mu_low  = expr[mask_lo].mean(axis=0)   # (G,)

        return (mu_high - mu_low).astype(np.float64)

    @staticmethod
    def balanced_accuracy_influence(
        influence_scores: np.ndarray,
        cell_types: np.ndarray,
        spatial_coords: Optional[np.ndarray] = None,
        k_smooth: int = 6,
        min_class_size: int = 3,
    ) -> float:
       
        from sklearn.metrics import balanced_accuracy_score, roc_auc_score
        from scipy.stats import rankdata

        scores = np.asarray(influence_scores, dtype=np.float64).flatten()
        ct_arr = np.asarray(cell_types).flatten()
        assert len(scores) == len(ct_arr), \
            f"length mismatch: scores={len(scores)} ct={len(ct_arr)}"

        if len(scores) == 0:
            return float("nan")

        # ── Step 0: spatial kNN smoothing (LARIS / DeepTalk approach) ─────
        if spatial_coords is not None and len(spatial_coords) == len(scores) and len(scores) > 1:
            try:
                from sklearn.neighbors import NearestNeighbors as _NNS
                coords = np.asarray(spatial_coords, dtype=np.float64)
                n_nbrs = min(k_smooth, len(scores) - 1)
                if n_nbrs >= 1:
                    _nbrs = _NNS(n_neighbors=n_nbrs + 1).fit(coords)
                    dists, idxs = _nbrs.kneighbors(coords)
                    sigma = float(np.median(dists[:, 1:])) + 1e-8
                    w_gauss = np.exp(-(dists[:, 1:] ** 2) / (2 * sigma ** 2))
                    w_gauss /= w_gauss.sum(axis=1, keepdims=True) + 1e-12
                    own_w = 1.0 / (n_nbrs + 1)
                    nbr_contrib = ((1.0 - own_w) * w_gauss * scores[idxs[:, 1:]]).sum(axis=1)
                    scores = own_w * scores + nbr_contrib
            except Exception:
                pass   # graceful fallback to unsmoothed scores

        # ── Step 1: global rank normalisation (ties get the average rank, ──
        # ── so duplicate influence scores don't bias the ranking) ─────────
        normed = (rankdata(scores, method="average") - 1.0) / max(len(scores) - 1, 1)

        # ── Step 2: per-cell-type one-vs-rest AUROC, macro-averaged ───────
        unique_cts = np.unique(ct_arr)
        per_class_scores: List[float] = []

        for ct in unique_cts:
            true_bin = (ct_arr == ct).astype(int)
            n_pos = int(true_bin.sum())
            n_neg = int(len(true_bin) - n_pos)
            if n_pos < min_class_size or n_neg < min_class_size:
                continue   # too few cells for a stable per-class estimate

            # Primary: threshold-free AUROC (BASS / RECCIPE)
            try:
                auc_val = float(roc_auc_score(true_bin, normed))
             
                per_class_scores.append(max(auc_val, 1.0 - auc_val))
                continue
            except Exception:
                pass

            # Fallback: 30-threshold percentile sweep (used only when AUROC
            # is undefined, e.g. `normed` is constant for this subset).
            best_ba: Optional[float] = None
            thresholds = np.percentile(normed, np.linspace(5, 95, 30))
            for thr in thresholds:
                pred = (normed >= thr).astype(int)
                if pred.sum() == 0 or pred.sum() == len(pred):
                    continue
                try:
                    ba = balanced_accuracy_score(true_bin, pred)
                    if best_ba is None or ba > best_ba:
                        best_ba = ba
                except Exception:
                    continue
            # No floor at 0.5 here: let a genuinely uninformative sweep pull
            # the macro average down rather than masking it as "at least chance".
            per_class_scores.append(best_ba if best_ba is not None else 0.5)

        if not per_class_scores:
            return float("nan")

        # Macro average: every cell type counts equally. See docstring note
        # above for why this replaces the previous size-weighted average.
        return float(np.mean(per_class_scores))

# =====================================================================
# Visualizer
# =====================================================================

class SVRNValidator:
    def __init__(self, model: SVRN):
        self.model = model.eval()

    @torch.no_grad()
    def compute_core_metrics(self, test_loader: PyGDataLoader) -> Dict[str, float]:
        print("\n--- Running Stable SVRN Validation Suite ---")
        device = next(self.model.parameters()).device

        _saved_max_edges = self.model.cfg.MAX_EDGES_PER_STEP
        self.model.cfg.MAX_EDGES_PER_STEP = 0

        total_kl = 0.0
        n_batches = 0

        all_influence = []
        all_edges = []
        all_edge_probs_mean = []   # per-edge mean LR probability (directionality signal)
        all_edge_probs_lr  = []    # (E, N_LR) — for Spearman ρ and F1
        all_raw_lr         = []    # (E, N_LR) — for Spearman ρ and F1

        batch_aurocs = []
        batch_auprcs = []

        for batch in test_loader:
            batch = batch.to(device)

            influence, kl_div, edge_logits_lr, edge_probs_lr, edge_index_used = self.model(
                batch.x, batch.edge_index, batch.lr_features, batch.spatial_coords,
                getattr(batch, "cell_type_idx", None),
            )

            total_kl += float(kl_div.detach().cpu())
            n_batches += 1

            all_influence.append(influence.detach().cpu().numpy().flatten())
            all_edges.append(edge_index_used.detach().cpu().numpy())
            # Mean LR probability per edge — used as directional edge weight in TE
            all_edge_probs_mean.append(
                edge_probs_lr.mean(dim=1).detach().cpu().numpy()
            )

            # Collect per-LR arrays for Spearman ρ and F1
            num_nodes_b = batch.x.size(0)
            N_LR_b = edge_probs_lr.shape[1]
            lr_reshaped_b = batch.lr_features.detach().cpu().view(num_nodes_b, N_LR_b, 2)
            L_b = lr_reshaped_b[:, :, 0]
            R_b = lr_reshaped_b[:, :, 1]
            src_b = edge_index_used.detach().cpu()[0]
            dst_b = edge_index_used.detach().cpu()[1]
            raw_lr_b = (L_b[src_b] * R_b[dst_b]).numpy()
            all_edge_probs_lr.append(edge_probs_lr.detach().cpu().numpy())
            all_raw_lr.append(raw_lr_b)

            auroc, auprc = self._biological_accuracy(
                edge_probs_lr.detach().cpu(),
                batch.cpu(),
                batch.edge_index.detach().cpu(),
                edge_index_used.detach().cpu(),
            )

            if not np.isnan(auroc):
                batch_aurocs.append(auroc)
            if not np.isnan(auprc):
                batch_auprcs.append(auprc)

        kl_val = total_kl / max(n_batches, 1)

        influence_scores = np.concatenate(all_influence)
        edge_index = np.concatenate(all_edges, axis=1)
        edge_weights = np.concatenate(all_edge_probs_mean) if all_edge_probs_mean else None

        path_robustness = self._path_robustness(influence_scores, edge_index)
        geary_c = self._local_geary_c(influence_scores, edge_index)

        auroc = float(np.mean(batch_aurocs)) if len(batch_aurocs) > 0 else np.nan
        auprc = float(np.mean(batch_auprcs)) if len(batch_auprcs) > 0 else np.nan

        # ── Spearman rho, F1, and Information Flow Score (all need cat_raw_lr) ─────
        cat_raw_lr = None
        if all_edge_probs_lr and all_raw_lr:
            try:
                cat_probs_lr = np.concatenate(all_edge_probs_lr, axis=0)
                cat_raw_lr   = np.concatenate(all_raw_lr,        axis=0)
                spearman_rho = self._spearman_lr(cat_probs_lr, cat_raw_lr)
                f1_lr        = self._f1_lr(cat_probs_lr, cat_raw_lr)
            except Exception as e:
                print(f"  [warn] Spearman/F1 computation failed: {e}")
                spearman_rho = float("nan")
                f1_lr        = float("nan")
        else:
            spearman_rho = float("nan")
            f1_lr        = float("nan")

        # Information Flow Score: primary path uses raw_lr for true directional asymmetry
        information_flow_score = self._information_flow_score(
            influence_scores, edge_index,
            edge_weights=edge_weights,
            raw_lr=cat_raw_lr,
        )

       
        # Re-run inference to collect aligned (influence, cell_type, spatial) triples
        all_inf_ba:    list = []
        all_ct_ba:     list = []
        all_coords_ba: list = []

        # idx_to_type for integer → string reconstruction
        _type_to_idx = getattr(self.model.cfg, "CT_TYPE_TO_IDX", None) or {}
        _idx_to_type = {v: k for k, v in _type_to_idx.items()}

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                inf_b, _, _, _, _ = self.model(
                    batch.x, batch.edge_index,
                    batch.lr_features, batch.spatial_coords,
                )
                inf_np = inf_b.detach().cpu().numpy().flatten()
                all_inf_ba.append(inf_np)

                # Collect spatial coordinates (for kNN smoothing in BA)
                if hasattr(batch, "spatial_coords"):
                    sc_np = batch.spatial_coords.detach().cpu().numpy()
                    if len(sc_np) == len(inf_np):
                        all_coords_ba.append(sc_np)

                # Strategy 1: numpy string array stored on Data object
                ct_raw = getattr(batch, "cell_types", None)
                if ct_raw is not None:
                    arr = np.asarray(ct_raw).flatten()
                    if len(arr) != len(inf_np):
                        arr = arr[:len(inf_np)] if len(arr) > len(inf_np) \
                              else np.pad(arr.astype(str),
                                          (0, len(inf_np) - len(arr)),
                                          constant_values="unknown")
                    all_ct_ba.append(arr)
                    continue

                # Strategy 2: integer cell_type_idx → reconstruct string labels
                ct_idx = getattr(batch, "cell_type_idx", None)
                if ct_idx is not None:
                    idx_arr = ct_idx.detach().cpu().numpy().flatten()
                    if _idx_to_type:
                        str_arr = np.array([_idx_to_type.get(int(i), str(i))
                                            for i in idx_arr])
                    else:
                        str_arr = idx_arr.astype(str)
                    all_ct_ba.append(str_arr)
                    continue

                all_ct_ba.append(np.full(len(inf_np), "unknown", dtype=object))

        balanced_acc = float("nan")
        ba_note       = "no cell-type labels found"

        if all_inf_ba and all_ct_ba:
            inf_concat = np.concatenate(all_inf_ba)
            ct_concat  = np.concatenate(all_ct_ba)
            min_len    = min(len(inf_concat), len(ct_concat))
            inf_use    = inf_concat[:min_len]
            ct_use     = ct_concat[:min_len]
            coords_use = (np.concatenate(all_coords_ba)[:min_len]
                          if len(all_coords_ba) == len(all_inf_ba)
                          else None)

            # Remove "unknown" sentinel entries
            known_mask = ct_use != "unknown"
            if known_mask.sum() > 0:
                inf_use    = inf_use[known_mask]
                ct_use     = ct_use[known_mask]
                coords_use = coords_use[known_mask] if coords_use is not None else None

            unique_cts, ct_counts = np.unique(ct_use, return_counts=True)
            n_classes = len(unique_cts)
            ba_note = (f"{min_len} cells, {n_classes} cell types; "
                       f"counts: {dict(zip(unique_cts, ct_counts))}")
            smooth_note = " | spatial smoothing applied (LARIS kNN)" if coords_use is not None else ""

            if n_classes >= 2:
                inf_std = float(np.std(inf_use))
                if inf_std < 1e-8:
                    balanced_acc = 0.5
                    ba_note += " | scores constant → BA fixed at 0.50 (random)"
                else:
                    try:
                        _um_tmp = UnifiedMetrics(self.model.cfg)
                        balanced_acc = _um_tmp.balanced_accuracy_influence(
                            inf_use, ct_use,
                            spatial_coords=coords_use,  # enables LARIS kNN smoothing
                            k_smooth=6,
                        )
                        ba_note += smooth_note
                        if not np.isnan(balanced_acc):
                            ba_note += f" | AUROC-based BA = {balanced_acc:.4f}"
                    except Exception as e:
                        ba_note += f" | exception in balanced_accuracy_influence: {e}"
            else:
                ba_note += " | only 1 unique cell type → skipped"

        print(f"  [balanced_acc] {ba_note}")
        if not np.isnan(balanced_acc):
            print(f"  Balanced_Accuracy_Influence: {balanced_acc:.4f}")

        results = {
            "KL_Divergence": kl_val,
            "Information_Flow_Score": information_flow_score,
            "Path_Robustness": path_robustness,
            "AUROC_pseudo_LR": auroc,
            "AUPRC_pseudo_LR": auprc,
            "AUPRC_random_baseline": "see [diag] positive rate above (~1-5% for MERFISH)",
            "Local_Geary_C": geary_c,
            "Balanced_Accuracy_Influence": balanced_acc,
            "Spearman_rho_LR": spearman_rho,
            "F1_LR": f1_lr,
        }

        print(f"\n  [Note] Labels = top-50% of ACTIVE (non-zero L*R) edges per LR pair.")
        print(f"         AUROC random baseline = 0.50.")
        print(f"         AUPRC random baseline approx positive rate (shown in [diag] above, typically 1-5%).")
        print(f"         AUPRC >> positive_rate indicates genuine LR-guided spatial signal.")
        print(f"         Balanced_Accuracy_Influence: niche-stratified multi-threshold BA-AUC;")
        print(f"           0.50=random, >=0.60=good, >=0.70=strong cell-type structure.")
        print(f"         Spearman_rho_LR: count-weighted macro-avg Spearman rho (pred vs L*R expression);")
        print(f"           >0.30=meaningful rank agreement, complements AUROC/AUPRC.")
        print(f"         F1_LR: count-weighted macro-avg F1 at data-driven threshold (top-k predictions")
        print(f"           where k=n_pos per pair); random baseline = 2*p/(1+p) ~ 0.02-0.10.")
        print(f"           No longer artificially suppressed by fixed 0.5 cut-off.\n")
        for k, v in results.items():
            if isinstance(v, float) and np.isnan(v):
                if k == "Balanced_Accuracy_Influence":
                    print(f"  {k}: nan  — see [balanced_acc] diagnostic above")
                else:
                    print(f"  {k}: nan  (insufficient data for this metric)")
            else:
                print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

        self.model.cfg.MAX_EDGES_PER_STEP = _saved_max_edges  # restore after full-graph pass
        return results

    @staticmethod
    def _histogram_nmi(x: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
        """Normalized mutual information between x and y via a 2D histogram.

        MI(x,y) = sum_{i,j} p(i,j) * log( p(i,j) / (p(i)*p(j)) )

        Normalized by min(H(x), H(y)) so the result lies in [0, 1]:
          - 0 -> x and y are (histogram-)independent
          - 1 -> one variable is a deterministic function of the other
                 (within binning resolution)
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if x.size < 4 or x.std() < 1e-12 or y.std() < 1e-12:
            return 0.0

        c_xy, _, _ = np.histogram2d(x, y, bins=n_bins)
        p_xy = c_xy / c_xy.sum()
        p_x = p_xy.sum(axis=1)
        p_y = p_xy.sum(axis=0)

        nz = p_xy > 0
        outer = np.outer(p_x, p_y)
        mi = float(np.sum(p_xy[nz] * np.log(p_xy[nz] / outer[nz])))

        h_x = float(-np.sum(p_x[p_x > 0] * np.log(p_x[p_x > 0])))
        h_y = float(-np.sum(p_y[p_y > 0] * np.log(p_y[p_y > 0])))
        denom = min(h_x, h_y)

        if denom <= 1e-12:
            return 0.0
        return float(np.clip(mi / denom, 0.0, 1.0))

  
    @staticmethod
    def _path_robustness(scores: np.ndarray, edge_index: np.ndarray) -> float:
        E = edge_index.shape[1]
        if E < 5: return 0.0
        keep = np.random.choice(E, int(E * 0.8), replace=False)
        src, dst = edge_index[0], edge_index[1]
        reduced_src, reduced_dst = src[keep], dst[keep]
        score_agg_full = np.bincount(dst, weights=scores[src], minlength=len(scores))
        score_agg_reduced = np.bincount(reduced_dst, weights=scores[reduced_src], minlength=len(scores))
        corr = np.corrcoef(score_agg_full, score_agg_reduced)[0, 1]
        return float(abs(corr)) if not np.isnan(corr) else 0.0
    @staticmethod
    def _biological_accuracy(
        edge_probs_lr: torch.Tensor,
        batch,
        full_edge_index: torch.Tensor,
        used_edge_index: torch.Tensor,
    ) -> Tuple[float, float]:
        from scipy.stats import spearmanr

        N_LR = edge_probs_lr.shape[1]
        num_nodes = batch.x.size(0)
        lr_reshaped = batch.lr_features.detach().cpu().view(num_nodes, N_LR, 2)
        L = lr_reshaped[:, :, 0]
        R = lr_reshaped[:, :, 1]

        src = used_edge_index[0].detach().cpu()
        dst = used_edge_index[1].detach().cpu()

        raw_lr = (L[src] * R[dst]).numpy()                        
        pred_lr = edge_probs_lr.detach().cpu().numpy()            


        E = raw_lr.shape[0]   # number of edges in the test graph
        MIN_ACTIVE = 10       # minimum active (non-zero L*R) edges to score a pair

        labels_lr = np.zeros_like(raw_lr, dtype=np.int32)
        valid_spearmans = []

        for j in range(N_LR):
            col = raw_lr[:, j]
            col_pred = pred_lr[:, j]

            active_mask = col > 1e-9
            n_active = int(active_mask.sum())

            # Skip: gene absent from panel entirely
            if col.max() < 1e-9 or n_active < MIN_ACTIVE:
                continue

            active_vals = col[active_mask]
            thresh_j = np.median(active_vals)
            labels_lr[(col >= thresh_j) & active_mask, j] = 1

            # Spearman for monitoring (full column, including zeros)
            if col_pred.max() - col_pred.min() > 1e-12:
                rho, _ = spearmanr(col, col_pred)
                if not np.isnan(rho):
                    valid_spearmans.append(rho)

        MIN_PER_CLASS = MIN_ACTIVE // 2

        n_scoreable = int(labels_lr.any(axis=0).sum())
        n_skipped   = N_LR - n_scoreable
        # Compute actual positive rate for AUPRC baseline reporting
        total_pos   = int(labels_lr.sum())
        total_edges = E * N_LR
        pos_rate    = total_pos / max(total_edges, 1)
        print(f"  [diag] LR pairs scored: {n_scoreable}/{N_LR} "
              f"({n_skipped} skipped: all-zero or < {MIN_ACTIVE} active edges).")
        print(f"  [diag] Positive rate: {pos_rate*100:.2f}%  "
              f"← AUPRC random baseline ≈ {pos_rate:.4f}")

        if labels_lr.any():
            pair_aurocs, pair_auprcs, pair_weights = [], [], []
            for j in range(N_LR):
                col_labels = labels_lr[:, j]
                col_pred   = pred_lr[:, j]
                col_expr   = raw_lr[:, j]

                n_pos = int(col_labels.sum())
                n_neg = int(len(col_labels) - n_pos)
                if n_pos < MIN_PER_CLASS or n_neg < MIN_PER_CLASS:
                    continue
                if col_pred.max() - col_pred.min() < 1e-12:
                    continue

                p = col_expr / (col_expr.sum() + 1e-9)
                p = p[p > 0]
                entropy_w = float(-np.sum(p * np.log(p + 1e-12))) + 1e-6  # always > 0

                try:
                    pair_aurocs.append(roc_auc_score(col_labels, col_pred))
                    pair_auprcs.append(average_precision_score(col_labels, col_pred))
                    pair_weights.append(entropy_w)
                except Exception:
                    continue

            if pair_auprcs:
                weights = np.array(pair_weights, dtype=np.float64)
                weights /= weights.sum()
                w_auroc = float(np.sum(np.array(pair_aurocs) * weights))
                w_auprc = float(np.sum(np.array(pair_auprcs) * weights))
                return w_auroc, w_auprc

        if valid_spearmans:
            mean_rho = float(np.mean(valid_spearmans))
            auroc_approx = float(np.clip(0.5 + mean_rho / 2, 0.0, 1.0))
            return auroc_approx, float("nan")  # AUPRC cannot be approximated from rho

        # Geometric distance fallback
        src_c = batch.spatial_coords[src].detach().cpu().numpy()
        dst_c = batch.spatial_coords[dst].detach().cpu().numpy()
        dist  = np.linalg.norm(src_c - dst_c, axis=1)
        pred_edge = pred_lr.mean(axis=1)
        if pred_edge.std() < 1e-12 or dist.std() < 1e-12:
            return 0.5, float("nan")
        corr, _ = spearmanr(dist, pred_edge)
        return float(np.clip(abs(corr), 0.0, 1.0)), float("nan")
  
    # ------------------------------------------------------------------
    # Spearman rank correlation — predicted edge prob vs raw L×R expression
    # ------------------------------------------------------------------
    @staticmethod
    def _spearman_lr(
        edge_probs_lr: np.ndarray,   # (E, N_LR) model predictions ∈ (0,1)
        raw_lr: np.ndarray,          # (E, N_LR) L[src]*R[dst] ground-truth
        min_active: int = 10,
    ) -> float:
     
        from scipy.stats import spearmanr
        N_LR = raw_lr.shape[1]
        rhos, weights = [], []
        for j in range(N_LR):
            col_raw  = raw_lr[:, j]
            col_pred = edge_probs_lr[:, j]
            active = col_raw > 1e-9
            n_active = int(active.sum())
            if n_active < min_active:
                continue
            if col_pred.max() - col_pred.min() < 1e-12:
                continue
            if col_raw.max() - col_raw.min() < 1e-12:
                continue
            try:
                rho, _ = spearmanr(col_raw, col_pred)
                if not np.isnan(rho):
                    rhos.append(float(rho))
                    weights.append(float(n_active))
            except Exception:
                continue
        if not rhos:
            return float("nan")
        w = np.array(weights, dtype=np.float64)
        w /= w.sum()
        return float(np.dot(rhos, w))

    # ------------------------------------------------------------------
    # F1 score — predicted edge labels vs raw L*R expression labels
    # ------------------------------------------------------------------
    @staticmethod
    def _f1_lr(
        edge_probs_lr: np.ndarray,   # (E, N_LR) model predictions in (0,1)
        raw_lr: np.ndarray,          # (E, N_LR) L[src]*R[dst] ground-truth
        min_active: int = 10,
    ) -> float:
        
        from sklearn.metrics import f1_score
        N_LR = raw_lr.shape[1]
        f1s, weights = [], []
        for j in range(N_LR):
            col_raw  = raw_lr[:, j]
            col_pred = edge_probs_lr[:, j]

            active_mask = col_raw > 1e-9
            n_active = int(active_mask.sum())
            if n_active < min_active:
                continue

            # Ground-truth label: top-50% of active edges are positive
            # (matches AUROC/AUPRC labelling in _biological_accuracy)
            thresh_raw = float(np.median(col_raw[active_mask]))
            y_true = ((col_raw >= thresh_raw) & active_mask).astype(np.int32)

            n_pos   = int(y_true.sum())
            n_total = len(y_true)
            if n_pos == 0 or n_pos == n_total:
                continue  # degenerate: only one class

            # Data-driven prediction threshold: predict the top-k edges as
            # positive where k = n_pos (match empirical positive count).
            # np.partition is O(E) and avoids a full sort.
            pivot = n_total - n_pos
            pred_threshold = float(np.partition(col_pred, pivot)[pivot])

            y_pred = (col_pred >= pred_threshold).astype(np.int32)

            # Guard against all-same predictions after thresholding
            if y_pred.sum() == 0 or y_pred.sum() == n_total:
                continue

            try:
                score = f1_score(y_true, y_pred, zero_division=0)
                f1s.append(float(score))
                weights.append(float(n_active))
            except Exception:
                continue

        if not f1s:
            return float("nan")
        w = np.array(weights, dtype=np.float64)
        w /= w.sum()
        return float(np.dot(f1s, w))

    # ------------------------------------------------------------------
    # Model evaluation visual panel
    # ------------------------------------------------------------------
    @staticmethod
    def plot_model_evaluation_panel(
        results: Dict[str, float],
        output_dir: str,
        split_name: str = "test",
    ) -> None:
      
        import matplotlib.gridspec as mgridspec

        def _safe(k, default=float("nan")):
            v = results.get(k, default)
            try:
                return float(v)
            except (TypeError, ValueError):
                return default

        auroc    = _safe("AUROC_pseudo_LR")
        auprc    = _safe("AUPRC_pseudo_LR")
        bal_acc  = _safe("Balanced_Accuracy_Influence")
        kl       = _safe("KL_Divergence")
        ifs      = _safe("Information_Flow_Score")
        pr       = _safe("Path_Robustness")

        # ── Colour helpers ────────────────────────────────────────────────
        def _gauge_color(val, lo=0.5, hi=0.8):
            """Red→amber→green gradient mapped to [lo, hi]."""
            if np.isnan(val):
                return "#aaaaaa"
            t = np.clip((val - lo) / (hi - lo), 0.0, 1.0)
            r = int(255 * (1 - t))
            g = int(200 * t)
            return f"#{r:02x}{g:02x}40"

        def _ba_color(ba):
            if np.isnan(ba):           return "#aaaaaa"
            if ba >= 0.65:             return "#22c55e"   # green — strong
            if ba >= 0.55:             return "#f59e0b"   # amber — moderate
            return                            "#ef4444"   # red — near random

        def _kl_color(kl_v):
            if np.isnan(kl_v):         return "#aaaaaa"
            if kl_v < 0.5:             return "#22c55e"
            if kl_v < 1.5:             return "#f59e0b"
            return                            "#ef4444"

        # ── Figure layout ─────────────────────────────────────────────────
        fig = plt.figure(figsize=(18, 9))
        outer = mgridspec.GridSpec(2, 1, figure=fig,
                                   height_ratios=[1.0, 1.0], hspace=0.55)

        # Row 1: three gauge panels
        top_gs = mgridspec.GridSpecFromSubplotSpec(
            1, 3, subplot_spec=outer[0], wspace=0.35
        )
        ax_auroc  = fig.add_subplot(top_gs[0])
        ax_auprc  = fig.add_subplot(top_gs[1])
        ax_ba     = fig.add_subplot(top_gs[2])

        # Row 2: secondary metrics bar chart
        bot_gs = mgridspec.GridSpecFromSubplotSpec(
            1, 1, subplot_spec=outer[1]
        )
        ax_bar = fig.add_subplot(bot_gs[0])

        # ── Helper: draw a half-donut gauge ──────────────────────────────
        def _draw_gauge(ax, value, title, lo=0.5, hi=1.0,
                        fmt=".3f", highlight=False):
            """Draw a semicircular gauge with the value printed in the centre."""
            ax.set_aspect("equal")
            ax.axis("off")

            # Background arc (grey)
            theta_bg = np.linspace(np.pi, 0, 200)
            ax.plot(np.cos(theta_bg), np.sin(theta_bg),
                    color="#e5e7eb", linewidth=18, solid_capstyle="butt")

            # Foreground arc (coloured, proportion of [lo, hi])
            if not np.isnan(value):
                frac = np.clip((value - lo) / max(hi - lo, 1e-8), 0.0, 1.0)
                theta_fg = np.linspace(np.pi, np.pi - frac * np.pi, 200)
                col = _gauge_color(value, lo, hi) if not highlight else _ba_color(value)
                ax.plot(np.cos(theta_fg), np.sin(theta_fg),
                        color=col, linewidth=18, solid_capstyle="butt")

            # Centre text
            val_str = f"{value:{fmt}}" if not np.isnan(value) else "N/A"
            ax.text(0, -0.05, val_str, ha="center", va="center",
                    fontsize=22, fontweight="bold", color="#1f2937")
            ax.text(0, -0.35, title, ha="center", va="center",
                    fontsize=10, color="#374151")

            # Threshold markers
            if not np.isnan(value) and hi - lo > 0:
                # 0.5 marker (random baseline for AUROC/BA)
                frac_05 = np.clip((0.5 - lo) / (hi - lo), 0.0, 1.0)
                theta_m = np.pi - frac_05 * np.pi
                ax.annotate("", xy=(np.cos(theta_m) * 0.62, np.sin(theta_m) * 0.62),
                             xytext=(np.cos(theta_m) * 0.78, np.sin(theta_m) * 0.78),
                             arrowprops=dict(arrowstyle="-", color="#9ca3af", lw=1.2))

            # Range labels
            ax.text(-1.05, -0.12, f"{lo:.2f}", ha="center", fontsize=7.5,
                    color="#6b7280")
            ax.text( 1.05, -0.12, f"{hi:.2f}", ha="center", fontsize=7.5,
                    color="#6b7280")

            if highlight:
                ax.set_facecolor("#fefce8")
                for spine in ax.spines.values():
                    spine.set_visible(False)

            ax.set_xlim(-1.25, 1.25)
            ax.set_ylim(-0.55, 1.15)

        _draw_gauge(ax_auroc, auroc,   "AUROC (LR pseudo-label)", lo=0.5, hi=1.0)
        _draw_gauge(ax_auprc, auprc,   "AUPRC (LR pseudo-label)", lo=0.0, hi=1.0)
        _draw_gauge(ax_ba,    bal_acc,
                    "Balanced Accuracy (influence vs cell type)",
                    lo=0.5, hi=1.0, highlight=True)

        # Balanced Accuracy interpretation box below gauge
        if not np.isnan(bal_acc):
            if bal_acc >= 0.65:
                interp = "Strong spatial structure in influence scores"
                icol   = "#15803d"
            elif bal_acc >= 0.55:
                interp = "Moderate spatial structure"
                icol   = "#92400e"
            else:
                interp = "Near-random (≈0.50 baseline)"
                icol   = "#991b1b"
            ax_ba.text(0, -0.52, interp, ha="center", va="center",
                       fontsize=8, color=icol, style="italic")

        # ── Row 2: secondary metrics horizontal bar chart ─────────────────
        sec_metrics = {
            "KL Divergence (down better)":       (kl,      _kl_color(kl),                              True),
            "Information Flow Score (up better, >0.5=directional)": (ifs, _gauge_color(ifs, 0.5, 1.0), False),
            "Path Robustness (up better)":       (pr,      _gauge_color(pr,  0.5, 1.0),               False),
            "Local Geary C (near 1.0 = random)": (gc,      _gauge_color(1.0 - abs(gc - 1.0), 0.0, 1.0), False),
            "Balanced Accuracy (up better, 0.55+ good)": (bal_acc, _ba_color(bal_acc),               False),
        }

        labels_bar = list(sec_metrics.keys())
        vals_bar   = [v[0] for v in sec_metrics.values()]
        cols_bar   = [v[1] for v in sec_metrics.values()]

        y_b = np.arange(len(labels_bar))
        bars_b = ax_bar.barh(
            y_b, [v if not np.isnan(v) else 0 for v in vals_bar],
            color=cols_bar, edgecolor="white", linewidth=0.4, height=0.6
        )
        ax_bar.set_yticks(y_b)
        ax_bar.set_yticklabels(labels_bar, fontsize=9)
        ax_bar.invert_yaxis()
        ax_bar.set_xlabel("Metric value", fontsize=10)
        ax_bar.set_title("All model evaluation metrics  (Balanced Accuracy highlighted in yellow)",
                         fontsize=10, fontweight="bold")
        ax_bar.spines[["top", "right"]].set_visible(False)

        # Annotate bar values
        for bar_b, val in zip(bars_b, vals_bar):
            if np.isnan(val):
                ax_bar.text(0.002, bar_b.get_y() + bar_b.get_height() / 2,
                            "N/A", va="center", fontsize=8, color="#6b7280")
            else:
                ax_bar.text(bar_b.get_width() + 0.002,
                            bar_b.get_y() + bar_b.get_height() / 2,
                            f"{val:.4f}", va="center", fontsize=8.5,
                            color="#1f2937", fontweight="bold")

        # Highlight balanced accuracy bar with background band
        ba_idx = list(sec_metrics.keys()).index(
            "Balanced Accuracy (up better, 0.55+ good)")
        ax_bar.axhspan(ba_idx - 0.45, ba_idx + 0.45,
                       color="#fef9c3", alpha=0.6, zorder=0)
        # Reference line at 0.5 (random baseline for classification metrics)
        ax_bar.axvline(0.5, color="#9ca3af", lw=0.9, ls="--", alpha=0.7,
                       label="0.50 (random baseline)")
        ax_bar.legend(fontsize=8, loc="lower right", framealpha=0.7)

        fig.suptitle(
            f"SVRN Model Evaluation Dashboard  [{split_name} split]",
            fontsize=14, fontweight="bold", y=1.01
        )
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"model_evaluation_panel_{split_name}.png")
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"✓ Model evaluation panel saved: {out_path}")

   
# =====================================================================
#  Consensus Influence
# =====================================================================

class ConsensusInfluence:
    """Aggregate per-cell influence scores across K folds into a stable,
    rank-based consensus per cell type.

    Algorithm (mirrors the function supplied by the user, extended with
    saving and diagnostics):

    Step 1 — Per-fold percentile rank
        For each fold, compute the mean raw influence score per cell type,
        then convert to a percentile rank in [0, 100] using ``scipy.stats.rankdata``.
        This makes scores comparable across folds even when absolute scales differ.

    Step 2 — Consensus score
        Take the **median** of the percentile ranks across all folds →
        consensus ∈ [0, 100].  Median is used (not mean) so that a single
        aberrant fold — e.g. a tissue region where a cell type is absent or
        grossly over-represented — cannot pull the consensus rank away from
        where the cell type ranks in the majority of folds.  This matters
        especially with stratified spatial folds, where some folds may lack
        specific microenvironments.  Higher consensus = more consistently
        highly ranked cell type across tissue regions.

    Step 3 — Selection frequency
        Count how many folds placed a cell type in the top-K by mean influence.
        Expressed as a fraction of total folds (0–1) for comparability regardless
        of K_FOLDS.  A cell type with sel_freq=1.0 was top-K in every fold.

    Parameters
    ----------
    K : int
        Number of top cell types per fold that count as "selected".
        Corresponds to ``cfg.CONSENSUS_K``.
    """

    def __init__(self, K: int = 2):
        self.K = K

    def compute(
        self,
        scores_by_fold: Dict[str, np.ndarray],
        cell_type_labels: np.ndarray,
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        scores_by_fold   : {fold_id_str: (n_test_cells,) influence array}
                           Keys are arbitrary (e.g. "fold_1", "fold_2", …).
                           Each array covers ONLY that fold's test cells, so
                           ``cell_type_labels`` must match the FULL dataset and
                           the mapping is done via ``fold_test_indices``.
        cell_type_labels : (N_total,) cell-type string array for the whole dataset.

        Note: call ``compute_from_full`` (below) which handles index alignment.

        Returns
        -------
        pd.DataFrame with columns:
            cell_type, consensus_pct_rank, selection_frequency, n_folds
        sorted descending by consensus_pct_rank.
        """
        from scipy.stats import rankdata

        cell_types = np.unique(cell_type_labels)
        n_folds    = len(scores_by_fold)

        pct_ranks  = []          # list of (n_cell_types,) arrays
        sel_counts = np.zeros(len(cell_types), dtype=np.float64)

        for run_scores in scores_by_fold.values():
            ct_means = np.array([
                run_scores[cell_type_labels == ct].mean()
                if np.any(cell_type_labels == ct) else 0.0
                for ct in cell_types
            ])
            # Percentile rank within this fold: ties averaged
            pct_ranks.append(rankdata(ct_means) / len(cell_types) * 100)

            # Selection frequency: which cell types land in top-K this fold?
            top_k_idx = np.argsort(ct_means)[-self.K:]
            sel_counts[top_k_idx] += 1

        # Median across folds is more robust than mean: a single aberrant fold
        # (e.g. a tissue region where a cell type is absent or grossly over-
        # represented) cannot pull the consensus rank far from where it sits
        # in the majority of folds.  With K_FOLDS ≥ 5 the median is also
        # more statistically efficient than the mean under heavy-tailed fold
        # variance (Huber 1981; Leys et al. 2013).
        pct_ranks_arr = np.array(pct_ranks)          # (n_folds, n_cell_types)
        consensus     = np.median(pct_ranks_arr, axis=0)   # (n_cell_types,)
        sel_freq      = sel_counts / n_folds               # normalised to [0,1]

        df = pd.DataFrame({
            "cell_type":            cell_types,
            "consensus_pct_rank":   np.round(consensus, 4),
            "selection_frequency":  np.round(sel_freq,  4),
            "n_folds":              n_folds,
        }).sort_values("consensus_pct_rank", ascending=False).reset_index(drop=True)

        return df

    def compute_from_full(
        self,
        fold_scores:        Dict[str, np.ndarray],   # {fold_id: (n_test,) scores}
        fold_test_indices:  Dict[str, np.ndarray],   # {fold_id: (n_test,) global idx}
        all_cell_types:     np.ndarray,              # (N_total,) global labels
    ) -> pd.DataFrame:
        """Align each fold's test-cell scores to global cell-type labels, then
        call ``compute``.

        Each fold's scores array covers only test cells; this method projects them
        onto the global label array so the per-cell-type means are computed correctly.
        """
        # Build fold-local label arrays (same length as each score array)
        scores_by_fold_local: Dict[str, np.ndarray] = {}
        labels_by_fold:       Dict[str, np.ndarray] = {}

        for fold_id, scores in fold_scores.items():
            idx = fold_test_indices[fold_id]
            labels_by_fold[fold_id] = all_cell_types[idx]

        # Determine the union of all cell types seen across folds
        all_seen_types = np.unique(np.concatenate(list(labels_by_fold.values())))

        from scipy.stats import rankdata

        pct_ranks  = []
        sel_counts = np.zeros(len(all_seen_types), dtype=np.float64)
        n_folds    = len(fold_scores)

        for fold_id, scores in fold_scores.items():
            local_labels = labels_by_fold[fold_id]
            ct_means = np.array([
                scores[local_labels == ct].mean()
                if np.any(local_labels == ct) else 0.0
                for ct in all_seen_types
            ])
            pct_ranks.append(rankdata(ct_means) / len(all_seen_types) * 100)
            top_k_idx = np.argsort(ct_means)[-self.K:]
            sel_counts[top_k_idx] += 1

        pct_ranks_arr = np.array(pct_ranks)                    # (n_folds, n_cell_types)
        consensus     = np.median(pct_ranks_arr, axis=0)       # robust to outlier folds
        sel_freq      = sel_counts / n_folds

        df = pd.DataFrame({
            "cell_type":           all_seen_types,
            "consensus_pct_rank":  np.round(consensus, 4),
            "selection_frequency": np.round(sel_freq,  4),
            "n_folds":             n_folds,
        }).sort_values("consensus_pct_rank", ascending=False).reset_index(drop=True)

        return df

    def save(self, df: pd.DataFrame, output_dir: str) -> str:
        """Save consensus results to CSV and print a summary."""
        path = os.path.join(output_dir, "consensus_influence.csv")
        df.to_csv(path, index=False)
        print(f"\n✓ Consensus influence saved: {path}")
        print(f"\n{'─'*55}")
        print(f"{'Cell Type':<30} {'Consensus':>10} {'Sel.Freq':>10}")
        print(f"{'─'*55}")
        for _, row in df.iterrows():
            bar_len = int(row["consensus_pct_rank"] / 5)
            bar = "█" * bar_len
            print(f"{row['cell_type']:<30} {row['consensus_pct_rank']:>9.1f}%"
                  f" {row['selection_frequency']:>9.2f}  {bar}")
        print(f"{'─'*55}")
        return path
