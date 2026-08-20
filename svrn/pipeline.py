# -*- coding: utf-8 -*-
"""
svrn.pipeline
=============
Top-level orchestrator: :class:`SVRNPipeline` wires together data
preprocessing (:mod:`svrn.data`), the model (:mod:`svrn.model`),
metrics/validation/consensus (:mod:`svrn.utils`), and plotting
(:mod:`svrn.visualization`) into the full train -> evaluate ->
Monte-Carlo uncertainty -> K-fold consensus -> plot workflow, plus the
``example_usage()`` synthetic-data smoke test and the ``main()`` CLI
entry point.

"""

import os
import argparse
import warnings
from typing import Dict, Any, List, Optional, Tuple

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger

from anndata import AnnData
from tqdm import tqdm

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader

from .utils import (
    Config,
    set_seed,
    get_device,
    UnifiedMetrics,
    SVRNValidator,
    ConsensusInfluence,
)
from .model import SVRN
from .data import ScalableDataPreprocessor
from .visualization import SVRNVisualizer, ConsensusPlotter


class SVRNPipeline:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.preprocessor = ScalableDataPreprocessor(cfg)
        self.model: Optional[SVRN] = None

        self.train_idx = None
        self.val_idx = None
        self.test_idx = None

        self.train_results = None
        self.val_results = None
        self.test_results = None
        self.test_results_std: Optional[np.ndarray] = None
        self.test_results_ci_lo: Optional[np.ndarray] = None
        self.test_results_ci_hi: Optional[np.ndarray] = None

    @staticmethod
    def create_isolated_subgraph(pyg_data: Data, indices: np.ndarray) -> Data:
        indices = torch.as_tensor(indices, dtype=torch.long)

        node_mask = torch.zeros(pyg_data.x.size(0), dtype=torch.bool)
        node_mask[indices] = True

        edge_mask = node_mask[pyg_data.edge_index[0]] & node_mask[pyg_data.edge_index[1]]
        sub_edge_index = pyg_data.edge_index[:, edge_mask]

        mapping = -torch.ones(pyg_data.x.size(0), dtype=torch.long)
        mapping[indices] = torch.arange(indices.size(0), dtype=torch.long)
        local_edge_index = mapping[sub_edge_index]

        kwargs = dict(
            x=pyg_data.x[indices],
            edge_index=local_edge_index,
            lr_features=pyg_data.lr_features[indices],
            spatial_coords=pyg_data.spatial_coords[indices],
        )
        if hasattr(pyg_data, "cell_type_idx") and pyg_data.cell_type_idx is not None:
            kwargs["cell_type_idx"] = pyg_data.cell_type_idx[indices]

        sub = Data(**kwargs)
        # Carry string cell-type labels (numpy array) into the subgraph so
        # that SVRNValidator.compute_core_metrics() can read batch.cell_types.
        if hasattr(pyg_data, "cell_types") and pyg_data.cell_types is not None:
            sub.cell_types = pyg_data.cell_types[indices.cpu().numpy()]
        return sub

    def run(self) -> None:
        set_seed(self.cfg.SEED)
        self.cfg.save_json(os.path.join(self.cfg.OUTPUT_DIR, "config.json"))

        pyg_data, full_data = self.preprocessor.get_pyg_data()

        # ── K-fold cross-validation (when K_FOLDS >= 2) ──────────────────
        if self.cfg.K_FOLDS >= 2:
            self.run_kfold(pyg_data, full_data)
            return

        self.train_idx, self.val_idx, self.test_idx = self.preprocessor.spatial_train_val_test_split(
            full_data["spatial_coords"],
            val_ratio=self.cfg.VAL_RATIO,
            test_ratio=self.cfg.TEST_RATIO,
        )

        train_graph = self.create_isolated_subgraph(pyg_data, self.train_idx)
        val_graph = self.create_isolated_subgraph(pyg_data, self.val_idx)
        test_graph = self.create_isolated_subgraph(pyg_data, self.test_idx)

        print("\nSpatial split completed:")
        print(f"  Train cells: {len(self.train_idx)} | edges: {train_graph.edge_index.size(1)}")
        print(f"  Val cells  : {len(self.val_idx)} | edges: {val_graph.edge_index.size(1)}")
        print(f"  Test cells : {len(self.test_idx)} | edges: {test_graph.edge_index.size(1)}")

        _dl_gen = torch.Generator()
        _dl_gen.manual_seed(self.cfg.SEED)
        def _worker_init(worker_id: int) -> None:
            set_seed(self.cfg.SEED + worker_id)

        self.train_loader = PyGDataLoader(
            [train_graph], batch_size=self.cfg.BATCH_SIZE, shuffle=True,
            generator=_dl_gen, worker_init_fn=_worker_init,
        )
        self.val_loader = PyGDataLoader(
            [val_graph], batch_size=self.cfg.BATCH_SIZE, shuffle=False,
            worker_init_fn=_worker_init,
        )
        self.test_loader = PyGDataLoader(
            [test_graph], batch_size=self.cfg.BATCH_SIZE, shuffle=False,
            worker_init_fn=_worker_init,
        )

        self.model = SVRN(self.cfg)

        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            dirpath=self.cfg.OUTPUT_DIR,
            filename="best_svrn_model_{epoch:03d}",
            save_top_k=1,
            save_last=True,
        )

        early_stop = EarlyStopping(
            monitor="val_loss",
           
            patience=20,
            mode="min",
            verbose=True,
        )

        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        trainer = pl.Trainer(
            max_epochs=self.cfg.EPOCHS,
            accelerator="gpu" if self.cfg.DEVICE == "cuda" else "cpu",
            devices=1,
            precision="32",
            num_sanity_val_steps=0,
            callbacks=[checkpoint_callback, early_stop, lr_monitor],
            logger=CSVLogger(self.cfg.OUTPUT_DIR, name="logs"),
            gradient_clip_val=1.0,
            gradient_clip_algorithm="norm",
            log_every_n_steps=1,
            enable_checkpointing=True,
            deterministic=True,
        )
         
        trainer.fit(self.model, self.train_loader, val_dataloaders=self.val_loader)

        self.train_results = self.get_split_influence(self.train_loader)
        self.val_results = self.get_split_influence(self.val_loader)

        # Test split: use MC sampling to get mean + uncertainty
        (
            self.test_results,
            self.test_results_std,
            self.test_results_ci_lo,
            self.test_results_ci_hi,
        ) = self.get_split_influence_with_uncertainty(
            self.test_loader, n_samples=self.cfg.MC_SAMPLES
        )

        validator = SVRNValidator(self.model)
        val_suite_results = validator.compute_core_metrics(self.test_loader)

        self.save_results(full_data, val_suite_results)
        self.make_plots(full_data)

    def run_kfold(self, pyg_data: Data, full_data: Dict[str, Any]) -> None:
        """Train and evaluate SVRN across N_RUNS independent repetitions, each with
        K_FOLDS tissue-stratified spatial folds.

        Two axes of variance are separated:
          - Fold variance   : which cells are in train/test (data partitioning)
          - Run variance    : model weight initialisation + training stochasticity

        Consensus influence is pooled across all n_runs × n_folds models, giving
        stable selection frequencies.  Wilson 95% CIs are reported per cell type.

        Directory layout::

            output_dir/
              run_1/
                fold_1/  split_indices.npz, best_svrn_*, logs/, plots/, *.csv
                fold_2/  ...
                kfold_metrics_summary.csv
              run_2/  ...
              consensus_influence.csv      ← pooled across ALL runs and folds
              all_runs_summary.csv         ← per-run mean±std metrics
        """
        k      = self.cfg.K_FOLDS
        n_runs = self.cfg.N_RUNS

        print(f"\n{'='*70}")
        print(f"MULTI-RUN K({k})-FOLD CROSS-VALIDATION  ({n_runs} run(s))")
        print(f"  Total models trained: {n_runs * k}")
        print(f"  Consensus K         : {self.cfg.CONSENSUS_K}")
        print(f"{'='*70}")

        _RUN_HASH = 0x9E3779B97F4A7C15   # Fibonacci hashing constant (64-bit)
        run_seeds = [
            int(self.cfg.SEED) ^ ((_RUN_HASH * (r + 1)) & 0xFFFFFFFF)
            for r in range(n_runs)
        ]

        # Folds are shared across runs — same data partitions, different model inits.
        # This cleanly isolates model variance from data-partition variance.
        folds = self.preprocessor.get_kfold_splits(
            full_data["spatial_coords"], k=k,
            cell_types=full_data["cell_types"],
        )

        # Accumulate across ALL runs × folds for final consensus
        all_fold_scores:       Dict[str, np.ndarray] = {}
        all_fold_test_indices: Dict[str, np.ndarray] = {}

        # Per-fold Hill parameters and LR edge probs for ConsensusPlotter
        all_fold_log_n:        Dict[str, np.ndarray] = {}
        all_fold_log_K:        Dict[str, np.ndarray] = {}
        all_fold_edge_probs:   Dict[str, np.ndarray] = {}
        all_fold_edge_indices: Dict[str, np.ndarray] = {}

        all_runs_metrics: List[Dict[str, Any]] = []
        # Accumulate UnifiedMetrics (morans_i, gearys_c, gini, etc.) per split per fold
        all_fold_unified_metrics: List[Dict[str, Any]] = []
        # Accumulate per-fold communication-corridor statistics
        # (4b_corridor_summary.csv written by SVRNVisualizer.plot_corridor_summary
        # for each fold's test split) for the consensus C4 plot.
        all_fold_corridor_dfs: List[pd.DataFrame] = []
        best_global_val_loss = float("inf")

        for run_i, run_seed in enumerate(run_seeds):
            print(f"\n{'#'*70}")
            print(f"  RUN {run_i + 1}/{n_runs}  (model seed: {run_seed})")
            print(f"{'#'*70}")

            run_output_dir = os.path.join(
                self.cfg.OUTPUT_DIR, f"run_{run_i + 1}"
            ) if n_runs > 1 else self.cfg.OUTPUT_DIR

            run_fold_metrics: List[Dict[str, Any]] = []
            best_run_val_loss = float("inf")
            best_run_fold_idx = 0

            for fold_i, (train_idx, val_idx, test_idx) in enumerate(folds):
                print(f"\n── Run {run_i+1}/{n_runs}  Fold {fold_i+1}/{k} "
                      f"──────────────────────────────")
                print(f"  Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

                self.train_idx = train_idx
                self.val_idx   = val_idx
                self.test_idx  = test_idx

                train_graph = self.create_isolated_subgraph(pyg_data, train_idx)
                val_graph   = self.create_isolated_subgraph(pyg_data, val_idx)
                test_graph  = self.create_isolated_subgraph(pyg_data, test_idx)

                # Model seed = run_seed XOR fold position — keeps fold variation
                # independent of run variation while remaining deterministic.
                model_seed = run_seed ^ (fold_i * 0x6B43A9B5 & 0xFFFFFFFF)
                set_seed(model_seed)

                _dl_gen = torch.Generator()
                _dl_gen.manual_seed(model_seed)
                def _worker_init_kf(worker_id: int, _seed: int = model_seed) -> None:
                    set_seed(_seed + worker_id)

                train_loader = PyGDataLoader(
                    [train_graph], batch_size=self.cfg.BATCH_SIZE, shuffle=True,
                    generator=_dl_gen, worker_init_fn=_worker_init_kf,
                )
                val_loader   = PyGDataLoader(
                    [val_graph],   batch_size=self.cfg.BATCH_SIZE, shuffle=False,
                    worker_init_fn=_worker_init_kf,
                )
                test_loader  = PyGDataLoader(
                    [test_graph],  batch_size=self.cfg.BATCH_SIZE, shuffle=False,
                    worker_init_fn=_worker_init_kf,
                )

                fold_model = SVRN(self.cfg)

                fold_output_dir = os.path.join(run_output_dir, f"fold_{fold_i + 1}")
                os.makedirs(fold_output_dir, exist_ok=True)

                # Save split indices (same across runs; only written on run 1)
                fold_split_path = os.path.join(fold_output_dir, "split_indices.npz")
                np.savez(fold_split_path,
                         train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
                print(f"  ✓ Split indices saved: {fold_split_path}")

                checkpoint_callback = ModelCheckpoint(
                    monitor="val_loss",
                    mode="min",
                    dirpath=fold_output_dir,
                    filename=f"best_svrn_r{run_i+1}_f{fold_i+1}_{{epoch:03d}}",
                    save_top_k=1,
                    save_last=True,
                )
                early_stop  = EarlyStopping(monitor="val_loss", patience=20,
                                            mode="min", verbose=False)
                lr_monitor  = LearningRateMonitor(logging_interval="epoch")

                trainer = pl.Trainer(
                    max_epochs=self.cfg.EPOCHS,
                    accelerator="gpu" if self.cfg.DEVICE == "cuda" else "cpu",
                    devices=1,
                    precision="32",
                    num_sanity_val_steps=0,
                    callbacks=[checkpoint_callback, early_stop, lr_monitor],
                    logger=CSVLogger(fold_output_dir, name="logs"),
                    gradient_clip_val=1.0,
                    gradient_clip_algorithm="norm",
                    log_every_n_steps=1,
                    enable_checkpointing=True,
                    deterministic=True,
                    enable_progress_bar=True,
                )

                trainer.fit(fold_model, train_loader, val_dataloaders=val_loader)

                fold_val_losses = getattr(fold_model, "val_losses", [])
                fold_best_val   = float(min(fold_val_losses)) if fold_val_losses else float("inf")

                # Unique key for this run × fold — used for all dict entries below
                entry_id = f"run{run_i+1}_fold{fold_i+1}"

                # ── Extract Hill KN parameters from trained model ─────────
                with torch.no_grad():
                    all_fold_log_n[entry_id] = (
                        fold_model.hill.log_n.detach().cpu().numpy().copy()
                    )
                    all_fold_log_K[entry_id] = (
                        fold_model.hill.log_K.detach().cpu().numpy().copy()
                    )

                # ── Extract mean edge_probs_lr over test loader ───────────
                fold_model.eval()
                _device = next(fold_model.parameters()).device
                ep_list: List[np.ndarray] = []
                ei_list: List[np.ndarray] = []
                with torch.no_grad():
                    for _batch in test_loader:
                        _batch = _batch.to(_device)
                        _, _, _, _ep, _ei = fold_model(
                            _batch.x, _batch.edge_index,
                            _batch.lr_features, _batch.spatial_coords,
                        )
                        ep_list.append(_ep.detach().cpu().numpy())
                        ei_list.append(_ei.detach().cpu().numpy())
                if ep_list:
                    all_fold_edge_probs[entry_id]   = np.concatenate(ep_list, axis=0)
                    all_fold_edge_indices[entry_id]  = np.concatenate(ei_list, axis=1)

                self.model        = fold_model
                self.train_loader = train_loader
                self.val_loader   = val_loader
                self.test_loader  = test_loader

                self.train_results = self.get_split_influence(train_loader)
                self.val_results   = self.get_split_influence(val_loader)
                (
                    self.test_results,
                    self.test_results_std,
                    self.test_results_ci_lo,
                    self.test_results_ci_hi,
                ) = self.get_split_influence_with_uncertainty(
                    test_loader, n_samples=self.cfg.MC_SAMPLES
                )

                validator = SVRNValidator(fold_model)
                val_suite = validator.compute_core_metrics(test_loader)

                _orig = self.cfg.OUTPUT_DIR
                self.cfg.OUTPUT_DIR = fold_output_dir
                self.save_results(full_data, val_suite)
                self.make_plots(full_data)
                self.cfg.OUTPUT_DIR = _orig

                # ── Collect this fold's communication-corridor statistics ─
                # (written by SVRNVisualizer.plot_corridor_summary to
                # <fold_output_dir>/plots/4b_corridor_summary.csv)
                fold_corridor_csv = os.path.join(fold_output_dir, "plots", "4b_corridor_summary.csv")
                if os.path.isfile(fold_corridor_csv):
                    try:
                        _fold_corr_df = pd.read_csv(fold_corridor_csv)
                        _fold_corr_df["run"]      = run_i + 1
                        _fold_corr_df["fold"]     = fold_i + 1
                        _fold_corr_df["entry_id"] = entry_id
                        all_fold_corridor_dfs.append(_fold_corr_df)
                        print(f"  ✓ Collected corridor summary for {entry_id}: "
                              f"{len(_fold_corr_df)} corridors")
                    except Exception as _e:
                        print(f"  ! Could not read {fold_corridor_csv} ({_e})")
                else:
                    print(f"  ! No corridor summary CSV found for {entry_id} "
                          f"(expected at {fold_corridor_csv})")

                # ── Collect UnifiedMetrics per split for fold summary ─────
                _metrics_calc = UnifiedMetrics(self.cfg)
                for _split_name, _idx, _scores in [
                    ("training_70",   train_idx, self.train_results),
                    ("validation_15", val_idx,   self.val_results),
                    ("testing_15",    test_idx,  self.test_results),
                ]:
                    _scores_norm = self._normalize_scores(_scores)
                    _adj   = full_data["adj_matrix"][_idx][:, _idx]
                    _ctypes = full_data["cell_types"][_idx]
                    _coords = full_data["spatial_coords"][_idx]
                    _um = _metrics_calc.compute_all(_scores_norm, _coords, _ctypes, _adj)
                    _um["split"] = _split_name
                    _um["run"]   = run_i + 1
                    _um["fold"]  = fold_i + 1
                    all_fold_unified_metrics.append(_um)

                fold_metrics = {
                    "run":   run_i + 1,
                    "fold":  fold_i + 1,
                    "model_seed": model_seed,
                    "n_train": len(train_idx),
                    "n_val":   len(val_idx),
                    "n_test":  len(test_idx),
                    "best_val_loss": fold_best_val,
                    **{f"test_{mk}": v
                       for mk, v in val_suite.items() if isinstance(v, float)},
                }
                run_fold_metrics.append(fold_metrics)
                all_runs_metrics.append(fold_metrics)

                all_fold_scores[entry_id]       = self.test_results.flatten()
                all_fold_test_indices[entry_id] = test_idx

                if fold_best_val < best_run_val_loss:
                    best_run_val_loss = fold_best_val
                    best_run_fold_idx = fold_i
                    if fold_best_val < best_global_val_loss:
                        best_global_val_loss = fold_best_val
                        self._best_model        = fold_model
                        self._best_train_loader = train_loader
                        self._best_val_loader   = val_loader
                        self._best_test_loader  = test_loader

                print(f"  Run {run_i+1} Fold {fold_i+1} best val loss: {fold_best_val:.6f}")

            # ── Per-run summary ───────────────────────────────────────────
            run_df = pd.DataFrame(run_fold_metrics)
            numeric_cols = [c for c in run_df.select_dtypes(include=[np.number]).columns
                            if c not in ("run", "fold", "model_seed",
                                         "n_train", "n_val", "n_test")]
            run_agg = {"run": run_i + 1, "fold": "mean±std"}
            for col in numeric_cols:
                run_agg[col] = f"{run_df[col].mean():.4f}±{run_df[col].std():.4f}"
            run_df = pd.concat([run_df, pd.DataFrame([run_agg])], ignore_index=True)
            run_summary_path = os.path.join(run_output_dir, "kfold_metrics_summary.csv")
            run_df.to_csv(run_summary_path, index=False)
            print(f"\n✓ Run {run_i+1} summary: {run_summary_path}")

        # ── Global summary across all runs × folds ────────────────────────
        global_df = pd.DataFrame(all_runs_metrics)
        numeric_cols = [c for c in global_df.select_dtypes(include=[np.number]).columns
                        if c not in ("run", "fold", "model_seed",
                                     "n_train", "n_val", "n_test")]
        global_agg = {"run": "ALL", "fold": "mean±std"}
        for col in numeric_cols:
            global_agg[col] = (f"{global_df[col].mean():.4f}"
                               f"±{global_df[col].std():.4f}")
        global_df = pd.concat([global_df, pd.DataFrame([global_agg])], ignore_index=True)
        global_summary_path = os.path.join(self.cfg.OUTPUT_DIR, "all_runs_summary.csv")
        global_df.to_csv(global_summary_path, index=False)
        print(f"\n✓ Global summary ({n_runs} runs × {k} folds): {global_summary_path}")

        unified_df = pd.DataFrame(all_fold_unified_metrics)
        metric_cols = [c for c in unified_df.columns
                       if c not in ("split", "run", "fold")]
        summary_rows = []
        for split_name in ["training_70", "validation_15", "testing_15"]:
            sub = unified_df[unified_df["split"] == split_name]
            if sub.empty:
                continue
            row: Dict[str, Any] = {"split": split_name, "n_folds": len(sub)}
            for col in metric_cols:
                vals = pd.to_numeric(sub[col], errors="coerce").dropna()
                if vals.empty:
                    row[f"{col}_mean"] = np.nan
                    row[f"{col}_std"]  = np.nan
                    row[f"{col}_cv_pct"] = np.nan
                else:
                    mu  = float(vals.mean())
                    std = float(vals.std())
                    cv  = abs(std / mu * 100) if mu != 0 else np.nan
                    row[f"{col}_mean"]   = round(mu,  6)
                    row[f"{col}_std"]    = round(std, 6)
                    row[f"{col}_cv_pct"] = round(cv,  2)
            summary_rows.append(row)

        # Also add a full per-fold detail sheet (all rows, all splits)
        unified_detail_path  = os.path.join(self.cfg.OUTPUT_DIR,
                                             "folds_unified_metrics_detail.csv")
        unified_summary_path = os.path.join(self.cfg.OUTPUT_DIR,
                                            "folds_unified_metrics_summary.csv")
        unified_df.to_csv(unified_detail_path, index=False)
        summary_unified_df = pd.DataFrame(summary_rows)
        summary_unified_df.to_csv(unified_summary_path, index=False)

        # Print the summary table to stdout
        print(f"\n{'='*70}")
        print("UNIFIED METRICS SUMMARY ACROSS ALL FOLDS")
        print(f"{'='*70}")
        for _, row in summary_unified_df.iterrows():
            print(f"\n  Split: {row['split']}  (n_folds={int(row['n_folds'])})")
            for col in metric_cols:
                mu  = row.get(f"{col}_mean",   np.nan)
                std = row.get(f"{col}_std",    np.nan)
                cv  = row.get(f"{col}_cv_pct", np.nan)
                print(f"    {col:<25} {mu:>10.6f} ± {std:.6f}   CV={cv:.1f}%")
        print(f"\n✓ Unified metrics detail : {unified_detail_path}")
        print(f"✓ Unified metrics summary: {unified_summary_path}")

        # ── Consensus influence across ALL runs × folds ───────────────────
        total_models = n_runs * k
        print(f"\n{'='*70}")
        print(f"COMPUTING CONSENSUS INFLUENCE  "
              f"(K={self.cfg.CONSENSUS_K}, {total_models} models: {n_runs}×{k})")
        print(f"{'='*70}")
        consensus_engine = ConsensusInfluence(K=self.cfg.CONSENSUS_K)
        consensus_df = consensus_engine.compute_from_full(
            fold_scores       = all_fold_scores,
            fold_test_indices = all_fold_test_indices,
            all_cell_types    = full_data["cell_types"],
        )
        consensus_engine.save(consensus_df, self.cfg.OUTPUT_DIR)

        # ── Consensus communication-corridor summary across ALL folds ─────
        # Aggregates the per-fold 4b_corridor_summary.csv files collected above
        # (one per run x fold) into:
        #   all_folds_corridor_summary.csv  - raw per-fold rows (long format)
        #   consensus_corridor_summary.csv  - mean ± std across folds, grouped
        #                                      by corridor_id (C1..C8)
        # The consensus table is handed to ConsensusPlotter for the C4 plot.
        consensus_corridor_df: Optional[pd.DataFrame] = None
        if all_fold_corridor_dfs:
            all_folds_corridor_df = pd.concat(all_fold_corridor_dfs, ignore_index=True)
            all_folds_corridor_path = os.path.join(
                self.cfg.OUTPUT_DIR, "all_folds_corridor_summary.csv"
            )
            all_folds_corridor_df.to_csv(all_folds_corridor_path, index=False)
            print(f"\n✓ All-folds corridor summary saved "
                  f"({len(all_folds_corridor_df)} rows from "
                  f"{all_folds_corridor_df['entry_id'].nunique()} folds): "
                  f"{all_folds_corridor_path}")

            def _mode_str(s: pd.Series) -> str:
                m = s.mode()
                return str(m.iloc[0]) if not m.empty else ""

            consensus_rows = []
            for corridor_id, grp in all_folds_corridor_df.groupby("corridor_id"):
                consensus_rows.append({
                    "corridor_id":        corridor_id,
                    "n_folds":            len(grp),
                    "mean_influence":     grp["mean_influence"].mean(),
                    "mean_influence_std": grp["mean_influence"].std(ddof=0),
                    "n_hops":             grp["n_hops"].mean(),
                    "n_hops_std":         grp["n_hops"].std(ddof=0),
                    "spatial_length":     grp["spatial_length"].mean(),
                    "spatial_length_std": grp["spatial_length"].std(ddof=0),
                    "dominant_type":      _mode_str(grp["dominant_type"]),
                    "src_type":           _mode_str(grp["src_type"]),
                    "dst_type":           _mode_str(grp["dst_type"]),
                })
            consensus_corridor_df = pd.DataFrame(consensus_rows)
            # Sort C1, C2, ... C10 in numeric order rather than lexicographic
            consensus_corridor_df["_sort_key"] = (
                consensus_corridor_df["corridor_id"]
                .str.extract(r"(\d+)").astype(int)[0]
            )
            consensus_corridor_df = (
                consensus_corridor_df.sort_values("_sort_key")
                .drop(columns="_sort_key")
                .reset_index(drop=True)
            )
            consensus_corridor_path = os.path.join(
                self.cfg.OUTPUT_DIR, "consensus_corridor_summary.csv"
            )
            consensus_corridor_df.to_csv(consensus_corridor_path, index=False)
            print(f"✓ Consensus corridor summary saved "
                  f"({len(consensus_corridor_df)} corridors, "
                  f"averaged across {all_folds_corridor_df['entry_id'].nunique()} folds): "
                  f"{consensus_corridor_path}")

            # ── 4c: Consensus RELAY table ───────────────────────────────
           
            if "relay_cell_ids_global" in all_folds_corridor_df.columns:
                relay_records = []
                all_ct = full_data["cell_types"]
                for _, row in all_folds_corridor_df.iterrows():
                    raw = row.get("relay_cell_ids_global", "")
                    if not isinstance(raw, str) or not raw:
                        continue
                    for cell_id_str in raw.split(","):
                        cell_id_str = cell_id_str.strip()
                        if not cell_id_str:
                            continue
                        cell_id = int(cell_id_str)
                        relay_records.append({
                            "corridor_id": row["corridor_id"],
                            "run":         row["run"],
                            "fold":        row["fold"],
                            "entry_id":    row["entry_id"],
                            "cell_id":     cell_id,
                            "cell_type":   all_ct[cell_id] if 0 <= cell_id < len(all_ct) else "",
                        })

                if relay_records:
                    relay_long_df = pd.DataFrame(relay_records)
                    n_folds_per_corridor = (
                        all_folds_corridor_df.groupby("corridor_id")["entry_id"].nunique()
                    )

                    consensus_relay_rows = []
                    for (corridor_id, cell_id), grp in relay_long_df.groupby(
                        ["corridor_id", "cell_id"]
                    ):
                        n_folds_total = int(n_folds_per_corridor.get(corridor_id, len(grp)))
                        n_as_relay = grp["entry_id"].nunique()
                        consensus_relay_rows.append({
                            "corridor_id":       corridor_id,
                            "cell_id":           cell_id,
                            "cell_type":         _mode_str(grp["cell_type"]),
                            "n_folds_as_relay":  n_as_relay,
                            "n_folds_total":     n_folds_total,
                            "relay_frequency":   n_as_relay / n_folds_total if n_folds_total else np.nan,
                        })

                    consensus_relay_df = pd.DataFrame(consensus_relay_rows)
                    consensus_relay_df["_sort_key"] = (
                        consensus_relay_df["corridor_id"].str.extract(r"(\d+)").astype(int)[0]
                    )
                    consensus_relay_df = (
                        consensus_relay_df
                        .sort_values(["_sort_key", "relay_frequency"], ascending=[True, False])
                        .drop(columns="_sort_key")
                        .reset_index(drop=True)
                    )
                    consensus_relay_path = os.path.join(
                        self.cfg.OUTPUT_DIR, "4c_consensus_corridor_relays.csv"
                    )
                    consensus_relay_df.to_csv(consensus_relay_path, index=False)
                    print(f"✓ Consensus corridor relay table saved "
                          f"({len(consensus_relay_df)} corridor–relay rows, "
                          f"{consensus_relay_df['cell_id'].nunique()} unique relay cells): "
                          f"{consensus_relay_path}")
                else:
                    print("\n! No relay cells found across folds "
                          "(all corridors had 0 intermediate hops); "
                          "skipping 4c_consensus_corridor_relays.csv.")
            else:
                print("\n! relay_cell_ids_global column not found in per-fold "
                      "corridor summaries (old-format 4b CSV?); "
                      "skipping 4c_consensus_corridor_relays.csv.")
        else:
            print("\n! No per-fold corridor summaries were collected; "
                  "C4 will be plotted without the corridor-statistics inset.")

        # ── Consensus plots (all 9 analysis types) ────────────────────────
        # Build LR pair labels from full_data
        _lr_df = full_data.get("lr_pairs", pd.DataFrame())
        if not _lr_df.empty:
            _lig_col = self.preprocessor.detect_lr_columns(_lr_df)[0]
            _rec_col = self.preprocessor.detect_lr_columns(_lr_df)[1]
            lr_names = [
                f"{l}__{r}" for l, r in zip(
                    _lr_df[_lig_col].astype(str), _lr_df[_rec_col].astype(str)
                )
            ]
        else:
            lr_names = []

        consensus_plotter = ConsensusPlotter(
            consensus_df          = consensus_df,
            fold_scores           = all_fold_scores,
            fold_test_indices     = all_fold_test_indices,
            all_cell_types        = full_data["cell_types"],
            all_spatial_coords    = full_data["spatial_coords"],
            adj_matrix            = full_data["adj_matrix"],
            output_dir            = os.path.join(self.cfg.OUTPUT_DIR, "consensus_plots"),
            fold_log_n            = all_fold_log_n,
            fold_log_K            = all_fold_log_K,
            fold_edge_probs       = all_fold_edge_probs,
            fold_edge_indices     = all_fold_edge_indices,
            lr_pair_names         = lr_names,
            consensus_corridor_df = consensus_corridor_df,
        )
        # Thread species from cfg so C9 mygene annotation uses the right organism
        consensus_plotter._species = getattr(self.cfg, "SPECIES", "mouse")
        consensus_plotter.run_all(sel_freq_threshold=0.5)


        # Expose best model
        self.model        = self._best_model
        self.train_loader = self._best_train_loader
        self.val_loader   = self._best_val_loader
        self.test_loader  = self._best_test_loader
        print(f"\n✓ Best global val loss: {best_global_val_loss:.6f}")

    @torch.no_grad()
    def get_split_influence(self, loader: PyGDataLoader) -> np.ndarray:
        """Single deterministic forward pass (used for train/val splits to save time)."""
        self.model.eval()
        device = next(self.model.parameters()).device
        all_influence = []

        for batch in tqdm(loader, desc="Extracting influence"):
            batch = batch.to(device)
            influence, _, _, _, _ = self.model(
                batch.x, batch.edge_index, batch.lr_features, batch.spatial_coords,
                getattr(batch, "cell_type_idx", None),
            )
            # Soft Winsorising: clamp only extreme outliers (q0.1 / q99.9)
            lower = torch.quantile(influence, 0.001)
            upper = torch.quantile(influence, 0.999)
            influence = torch.clamp(influence, lower, upper)
            all_influence.append(influence.detach().cpu().numpy())

        result = np.concatenate(all_influence, axis=0)
        print(f"✓ Influence extracted: {result.shape}")
        return result

    def get_split_influence_with_uncertainty(
        self, loader: PyGDataLoader, n_samples: int = 30
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """MC-dropout / VAE stochastic sampling for per-cell uncertainty estimates.

        Keeps the model in *train* mode so that both dropout and the VAE
        reparameterisation noise remain active, producing a posterior sample.

        Returns
        -------
        mean_scores : (n_cells,)
        std_scores  : (n_cells,)  — posterior standard deviation
        ci_lo       : (n_cells,)  — 2.5th percentile
        ci_hi       : (n_cells,)  — 97.5th percentile
        """
        set_seed(self.cfg.SEED)          # reproducible MC samples
        self.model.train()               # activate stochastic components
        device = next(self.model.parameters()).device
        all_samples: List[np.ndarray] = []

        print(f"  Running {n_samples} MC forward passes for uncertainty estimation...")
        for s in range(n_samples):
            run_scores: List[np.ndarray] = []
            with torch.no_grad():
                for batch in loader:
                    batch = batch.to(device)
                    influence, _, _, _, _ = self.model(
                        batch.x, batch.edge_index, batch.lr_features, batch.spatial_coords,
                        getattr(batch, "cell_type_idx", None),
                    )
                    lower = torch.quantile(influence, 0.001)
                    upper = torch.quantile(influence, 0.999)
                    influence = torch.clamp(influence, lower, upper)
                    run_scores.append(influence.detach().cpu().numpy())
            all_samples.append(np.concatenate(run_scores, axis=0))

        self.model.eval()                # restore eval mode
        samples = np.stack(all_samples)  # (n_samples, n_cells)
        mean_s = samples.mean(axis=0)
        std_s = samples.std(axis=0)
        ci_lo = np.percentile(samples, 2.5, axis=0)
        ci_hi = np.percentile(samples, 97.5, axis=0)
        print(f"✓ MC uncertainty computed over {n_samples} samples.")
        return mean_s, std_s, ci_lo, ci_hi


    @staticmethod
    def _normalize_scores(scores: np.ndarray) -> np.ndarray:
        scores = scores.flatten()
        return (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

    def analyze_cell_type_influence(self, scores: np.ndarray, cell_types: np.ndarray, split_name: str) -> pd.DataFrame:
        df = pd.DataFrame({"cell_type": cell_types, "influence": scores.flatten()})
        stats = df.groupby("cell_type")["influence"].agg(["mean", "std", "count"]).reset_index()
        stats = stats.sort_values("mean", ascending=False)

        path = os.path.join(self.cfg.OUTPUT_DIR, f"cell_type_influence_{split_name}.csv")
        stats.to_csv(path, index=False)

        print(f"\nTop signaling cell types for {split_name}:")
        print(stats.head(10))
        print(f"✓ Cell-type influence saved: {path}")
        return stats

    def save_split_csv(
        self,
        full_data: Dict[str, Any],
        split_name: str,
        indices: np.ndarray,
        scores: np.ndarray,
        scores_std: Optional[np.ndarray] = None,
        scores_ci_lo: Optional[np.ndarray] = None,
        scores_ci_hi: Optional[np.ndarray] = None,
    ) -> None:
        scores_norm = self._normalize_scores(scores)
        row = {
            "cell_id": [full_data["cell_names"][i] for i in indices],
            "cell_type": [full_data["cell_types"][i] for i in indices],
            "influence_score": scores.flatten(),
            "influence_score_norm": scores_norm,
        }
        if scores_std is not None:
            row["influence_std"] = scores_std.flatten()
        if scores_ci_lo is not None:
            row["influence_ci_lo_2.5"] = scores_ci_lo.flatten()
        if scores_ci_hi is not None:
            row["influence_ci_hi_97.5"] = scores_ci_hi.flatten()
        df = pd.DataFrame(row)
        path = os.path.join(self.cfg.OUTPUT_DIR, f"SVRN_{split_name}_results.csv")
        df.to_csv(path, index=False)
        print(f"✓ {split_name} results saved: {path}")


    def save_results(self, full_data: Dict[str, Any], val_suite_results: Dict[str, float]) -> None:
        metrics_calculator = UnifiedMetrics(self.cfg)
        all_metrics = []

        split_info = [
            ("training_70", self.train_idx, self.train_results, None, None, None),
            ("validation_15", self.val_idx, self.val_results, None, None, None),
            ("testing_15", self.test_idx, self.test_results,
             self.test_results_std, self.test_results_ci_lo, self.test_results_ci_hi),
        ]

        for split_name, idx, scores, s_std, s_lo, s_hi in split_info:
            scores_norm = self._normalize_scores(scores)
            adj = full_data["adj_matrix"][idx][:, idx]
            cell_types = full_data["cell_types"][idx]
            coords = full_data["spatial_coords"][idx]

            metrics = metrics_calculator.compute_all(scores_norm, coords, cell_types, adj)
            metrics["split"] = split_name
            all_metrics.append(metrics)

            self.save_split_csv(full_data, split_name, idx, scores, s_std, s_lo, s_hi)
            self.analyze_cell_type_influence(scores_norm, cell_types, split_name)

        metrics_df = pd.DataFrame(all_metrics)
        metrics_path = os.path.join(self.cfg.OUTPUT_DIR, "final_metrics.csv")
        metrics_df.to_csv(metrics_path, index=False)
        print(f"✓ Unified metrics saved: {metrics_path}")

        val_path = os.path.join(self.cfg.OUTPUT_DIR, "validation_summary.csv")
        pd.DataFrame([val_suite_results]).to_csv(val_path, index=False)
        print(f"✓ Validation summary saved: {val_path}")

        # ── Model evaluation visual panel ─────────────────────────────────
        SVRNValidator.plot_model_evaluation_panel(
            results=val_suite_results,
            output_dir=os.path.join(self.cfg.OUTPUT_DIR, "plots"),
            split_name="test",
        )

    def make_plots(self, full_data: Dict[str, Any]) -> None:
        test_idx = np.asarray(self.test_idx)
        scores = self._normalize_scores(self.test_results)
        adj = full_data["adj_matrix"][test_idx][:, test_idx]

        _mn = self.test_results.min()
        _mx = self.test_results.max() + 1e-8

        def _norm_like(arr):
            if arr is None:
                return None
            return np.clip((arr - _mn) / (_mx - _mn), 0.0, 1.0)

        visualizer = SVRNVisualizer(
            influence_scores=scores,
            spatial_coords=full_data["spatial_coords"][test_idx],
            cell_types=full_data["cell_types"][test_idx],
            adj_matrix=adj,
            output_dir=os.path.join(self.cfg.OUTPUT_DIR, "plots"),
            influence_std=_norm_like(self.test_results_std),
            influence_ci_lo=_norm_like(self.test_results_ci_lo),
            influence_ci_hi=_norm_like(self.test_results_ci_hi),
            global_cell_idx=test_idx,
        )

        train_losses = getattr(self.model, "train_losses", [])
        val_losses = getattr(self.model, "val_losses", [])
        val_epochs = getattr(self.model, "val_epochs", [])
        visualizer.run_all(train_losses, val_losses, val_epochs)

# =====================================================================
#  Main
# =====================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Stable SVRN for spatial transcriptomics.")

    parser.add_argument("--data_path", type=str, help="Path to .h5ad spatial transcriptomics file.")
    parser.add_argument("--lr_path", type=str, help="Path to ligand-receptor CSV file.")
    parser.add_argument("--output_dir", type=str, default="svrn_results", help="Output directory.")

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)  # reduced from 1e-3 to stabilise spike training
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--max_hops", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--min_cells", type=int, default=20)
    parser.add_argument("--min_genes", type=int, default=10)
    parser.add_argument("--n_hvgs", type=int, default=1000)
    parser.add_argument("--mc_samples", type=int, default=30,
                        help="Number of MC forward passes for uncertainty estimation.")

    parser.add_argument("--k_folds", type=int, default=5,
                        help="Number of folds for K-fold cross-validation (0 or 1 = disabled). "
                             "25 is recommended for stable consensus influence estimation.")

    parser.add_argument("--n_runs", type=int, default=5,
                        help="Independent training runs with different model-init seeds. "
                             "Literature recommendation: 20–25 for ±10%% selection-frequency "
                             "precision. Each run reuses the same data folds so fold variance "
                             "and model variance are cleanly separated.")

    parser.add_argument("--consensus_k", type=int, default=2,
                        help="Top-K cell types per fold counted as 'selected' in consensus "
                             "selection frequency (default: 2).")

    # Memory controls
    parser.add_argument("--n_neighbors", type=int, default=14,
                        help="KNN spatial graph degree. Lower = fewer edges = less VRAM.")
    parser.add_argument("--edge_chunk_size", type=int, default=512,
                        help="Edges per chunk in LR encoder. Lower = less peak VRAM.")
    parser.add_argument("--max_edges_per_step", type=int, default=4000,
                        help="Subsample edges per training step (0 = use all).")

    parser.add_argument("--run_example", action="store_true")

    parser.add_argument(
        "--split_path", type=str, default="",
        help=(
            "Path to a previously saved split_indices.npz file. "
            "If omitted, the script looks for <output_dir>/split_indices.npz "
            "and loads it automatically; if that file does not exist it "
            "computes a fresh split and saves it there."
        ),
    )

    parser.add_argument(
        "--kfold_split_path", type=str, default="",
        help=(
            "Path to a previously saved kfold_split_indices.npz file. "
            "If omitted, the script looks for "
            "<output_dir>/kfold_split_indices.npz and loads it automatically; "
            "if that file does not exist it computes fresh folds and saves "
            "them there. Reusing this file across runs is what guarantees "
            "identical fold membership/counts run-to-run — recomputing folds "
            "from scratch each time is only seeded, not bit-reproducible, "
            "since spatial KMeans clustering runs through multi-threaded "
            "floating-point reductions whose order can vary between runs."
        ),
    )

    args = parser.parse_args()

    if args.run_example:
        example_usage()
        return

    if not args.data_path or not args.lr_path:
        parser.error("--data_path and --lr_path are required unless --run_example is used.")

    cfg = Config(
        DATA_PATH=args.data_path,
        LR_PATH=args.lr_path,
        OUTPUT_DIR=args.output_dir,
        EPOCHS=args.epochs,
        BATCH_SIZE=args.batch_size,
        LR=args.lr,
        HIDDEN_DIM=args.hidden_dim,
        MULTI_HOP_STEPS=args.max_hops,
        DROPOUT=args.dropout,
        SEED=args.seed,
        MIN_CELLS=args.min_cells,
        MIN_GENES=args.min_genes,
        N_HVGS=args.n_hvgs,
        MC_SAMPLES=args.mc_samples,
        N_NEIGHBORS=args.n_neighbors,
        EDGE_CHUNK_SIZE=args.edge_chunk_size,
        MAX_EDGES_PER_STEP=args.max_edges_per_step,
        SPLIT_PATH=args.split_path,
        KFOLD_SPLIT_PATH=args.kfold_split_path,
        K_FOLDS=args.k_folds,
        N_RUNS=args.n_runs,
        CONSENSUS_K=args.consensus_k,
    )

    pipeline = SVRNPipeline(cfg)
    pipeline.run()


if __name__ == "__main__":
    main()
