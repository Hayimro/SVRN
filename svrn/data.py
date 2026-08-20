# -*- coding: utf-8 -*-
"""
svrn.data
=========
Data loading and preprocessing: reading the AnnData spatial
transcriptomics object and ligand-receptor table, QC filtering,
normalization, HVG selection, spatial KNN graph construction, and
train/val/test and K-fold splitting (:class:`ScalableDataPreprocessor`).

Split out of the original monolithic ``pipeline.py``; depends only on
:class:`svrn.utils.Config` for configuration.
"""

import os
import zipfile
import random
import warnings
from typing import Dict, Any, List, Optional, Tuple

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

import scanpy as sc
from anndata import AnnData
from scipy.sparse import csr_matrix, issparse
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans

from torch_geometric.data import Data

from .utils import Config, set_seed


class ScalableDataPreprocessor:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.adata: Optional[AnnData] = None

    @staticmethod
    def _dense(X):
        if issparse(X):
            return X.toarray()
        return np.asarray(X)

    @staticmethod
    def detect_lr_columns(lr_pairs: pd.DataFrame) -> Tuple[str, str]:
        # Priority: ENSEMBL columns first so they match the ENSMUSG panel var_names.
        # Gene-symbol columns are last-resort fallbacks only.
        candidates = [
            ("Ligand_Ensembl",  "Receptor_Ensembl"),   # corrected CSV (primary)
            ("Ligand_ENSEMBL",  "Receptor_ENSEMBL"),   # alternate capitalisation
            ("ligand_gene_id",  "receptor_gene_id"),
            ("ligand",          "receptor"),
            ("Ligand",          "Receptor"),
            ("Ligand Symbols",  "Receptor Symbols"),   # symbol fallback — last resort
            ("source",          "target"),
        ]
        for ligand_col, receptor_col in candidates:
            if ligand_col in lr_pairs.columns and receptor_col in lr_pairs.columns:
                if lr_pairs[ligand_col].notna().any() and lr_pairs[receptor_col].notna().any():
                    return ligand_col, receptor_col
        raise ValueError(
            "Could not detect ligand/receptor columns. "
            f"Found columns: {lr_pairs.columns.tolist()}"
        )

    @staticmethod
    def detect_cell_type_column(adata: AnnData) -> str:
        candidates = ["class", "cell_type", "celltype", "CellType", "annotation", "label", "cluster"]
        for col in candidates:
            if col in adata.obs.columns:
                return col
        print("⚠️ No cell type column found. Using 'Unknown'.")
        adata.obs["Unknown"] = "Unknown"
        return "Unknown"

    def get_hvg_features(self, adata: AnnData) -> torch.Tensor:
   
        print("  Using log-normalised HVG expression features.")
        return torch.tensor(self._dense(adata.X), dtype=torch.float32)

    def get_cell_type_prior(self, adata: AnnData, adj_matrix: csr_matrix):
        """Compute a row-normalised cell-type transition matrix P[ct_i, ct_j].

        Biological rationale
        --------------------
        Seminal spatial-CCC benchmarks (Cang & Nie 2020 Nat Methods; Armingol et al.
        2021 Nat Rev Genet; Palla et al. 2022 Nat Methods) consistently show that
        cell-type identity is the dominant structural constraint on which ligand-receptor
        axes are active.  Using graph-weighted co-occurrence counts as the prior (rather
        than a flat uniform prior) encodes the actual tissue microenvironment topology:
        abundant sender->receiver pairs in the KNN graph receive higher prior weight,
        biasing the BCE target toward biologically plausible interactions and away from
        spurious co-expression artefacts (cf. CellChat normalisation, Jin et al. 2021).

        Returns
        -------
        ct_prior   : torch.Tensor [n_types, n_types]  row-normalised transition probs
        type_to_idx: dict  str cell-type -> int index
        """
        cell_type_col = self.detect_cell_type_column(adata)
        labels = adata.obs[cell_type_col].astype(str).values
        unique_types = sorted(set(labels))
        type_to_idx = {t: i for i, t in enumerate(unique_types)}
        label_indices = np.array([type_to_idx[l] for l in labels])
        n_types = len(unique_types)

        ct_counts = np.zeros((n_types, n_types), dtype=np.float32)
        rows, cols = adj_matrix.nonzero()
        weights = np.asarray(adj_matrix.data, dtype=np.float32)
        for r, c, w in zip(rows, cols, weights):
            ct_counts[label_indices[r], label_indices[c]] += w

        # Row-normalise -> transition probabilities (Markov-style, cf. CellChat 2021)
        row_sums = ct_counts.sum(axis=1, keepdims=True)
        ct_prior = ct_counts / (row_sums + 1e-8)
        preview = unique_types[:5]
        suffix = "..." if n_types > 5 else ""
        print(f"  Cell-type prior shape {ct_prior.shape} | types: {preview}{suffix}")
        return torch.tensor(ct_prior, dtype=torch.float32), type_to_idx
    def construct_spatial_graph(self, spatial_coords: np.ndarray, n_neighbors: int = 6) -> csr_matrix:
       print("  Constructing Adaptive Capped Gaussian-weighted spatial KNN graph for MERFISH...")
       n_neighbors = min(n_neighbors, len(spatial_coords))
       knn = NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto").fit(spatial_coords)
       distances, indices = knn.kneighbors(spatial_coords)

       median_dist = np.median(distances[:, 1:]) if distances.shape[1] > 1 else np.median(distances)
       max_distance_cap = median_dist * 2.5  
       
       sigma = median_dist + 1e-8
       weights = np.exp(-(distances ** 2) / (2.0 * sigma**2))

       valid_mask = distances <= max_distance_cap
       
       rows, cols, filtered_weights = [], [], []
       for i in range(len(spatial_coords)):
           valid_idx = valid_mask[i]
           rows.extend([i] * np.sum(valid_idx))
           cols.extend(indices[i, valid_idx])
           filtered_weights.extend(weights[i, valid_idx])

       adj = csr_matrix((filtered_weights, (rows, cols)), shape=(len(spatial_coords), len(spatial_coords)))
       adj.setdiag(0.0)
       adj.eliminate_zeros()
       return adj

    @staticmethod
    def to_edge_index(adj_matrix: csr_matrix) -> torch.Tensor:
        coo = adj_matrix.tocoo()
        return torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)

    def get_lr_features(self, adata: AnnData, lr_pairs: pd.DataFrame) -> torch.Tensor:
        print("  Extracting ligand-receptor features...")
        ligand_col, receptor_col = self.detect_lr_columns(lr_pairs)

        all_features = []
        n_cells = adata.shape[0]

        var_names = set(adata.var_names)

        var_lower = {g.lower(): g for g in adata.var_names}

        all_ligands   = lr_pairs[ligand_col].dropna().unique().tolist()
        all_receptors = lr_pairs[receptor_col].dropna().unique().tolist()
        all_lr_genes  = set(all_ligands + all_receptors)

        exact_hits  = all_lr_genes & var_names
        case_hits   = {g for g in all_lr_genes if g.lower() in var_lower and g not in var_names}
        total_miss  = len(all_lr_genes) - len(exact_hits) - len(case_hits)

        print(f"\n  ── LR gene–panel alignment diagnostic ──")
        print(f"     LR CSV unique genes  : {len(all_lr_genes)}")
        print(f"     Exact matches in panel: {len(exact_hits)}")
        print(f"     Case-insensitive hits : {len(case_hits)}  (will be remapped)")
        print(f"     Unresolvable misses   : {total_miss}  (filled with zeros)")
        if total_miss == len(all_lr_genes):
            print(f"\n  ⚠️  WARNING: 0/{len(all_lr_genes)} LR genes found in adata.var_names!")
            print(f"     adata.var_names sample : {list(adata.var_names)[:8]}")
            print(f"     LR CSV gene sample     : {all_ligands[:4] + all_receptors[:4]}")
            print(f"     → Check gene name namespace (symbol vs ENSEMBL vs Entrez).\n")
        elif total_miss > 0:
            missed = [g for g in all_lr_genes
                      if g not in var_names and g.lower() not in var_lower]
            print(f"     Missed genes (first 10): {missed[:10]}")
        print(f"  ────────────────────────────────────────\n")

        n_matched_pairs = 0
        for idx, row in lr_pairs.iterrows():
            ligand_gene   = row[ligand_col]
            receptor_gene = row[receptor_col]

            # Resolve with case-insensitive fallback
            def _resolve(gene):
                if pd.isna(gene):
                    return None
                if gene in var_names:
                    return gene
                if gene.lower() in var_lower:
                    return var_lower[gene.lower()]
                return None

            lig_key = _resolve(ligand_gene)
            rec_key = _resolve(receptor_gene)

            ligand_expr   = (self._dense(adata[:, lig_key].X).reshape(-1).astype(np.float32)
                             if lig_key else np.zeros(n_cells, dtype=np.float32))
            receptor_expr = (self._dense(adata[:, rec_key].X).reshape(-1).astype(np.float32)
                             if rec_key else np.zeros(n_cells, dtype=np.float32))

            if lig_key and rec_key:
                n_matched_pairs += 1

            all_features.append(torch.tensor(ligand_expr, dtype=torch.float32).unsqueeze(1))
            all_features.append(torch.tensor(receptor_expr, dtype=torch.float32).unsqueeze(1))

            if (idx + 1) % 50 == 0:
                print(f"    Processed {idx + 1}/{len(lr_pairs)} LR pairs")

        print(f"  ✓ LR pairs with BOTH ligand AND receptor matched: {n_matched_pairs}/{len(lr_pairs)}")
        if n_matched_pairs == 0:
            print(f"\n  ⛔ CRITICAL: No LR pair could be matched to the expression panel.")
            print(f"     All lr_features will be zero → AUROC/AUPRC will be NaN.")
            print(f"     Solutions: (1) add gene_alias column to CSV, (2) convert gene")
            print(f"     names to match adata.var_names namespace before running.\n")

        if len(all_features) == 0:
            raise ValueError("No ligand-receptor features were created. Check LR CSV.")

        lr_features = torch.cat(all_features, dim=1)
        return lr_features

    def _compute_split(
        self,
        spatial_coords: np.ndarray,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute a fresh spatial KMeans split (deterministic given seed + coords)."""
        n_cells = spatial_coords.shape[0]
        n_clusters = max(500, int(1 / min(val_ratio, test_ratio)))
        n_clusters = min(n_clusters, n_cells)

        # Re-seed immediately before KMeans so the split is independent of
        # whatever RNG state upstream operations (HVG, graph build, …) left
        # behind.  This is necessary but not sufficient across library upgrades
        # or filter-induced cell-count changes; saving indices is the hard
        # guarantee (see spatial_train_val_test_split below).
        set_seed(self.cfg.SEED)
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.cfg.SEED, n_init="auto")
        labels = kmeans.fit_predict(spatial_coords)

        rng = np.random.default_rng(self.cfg.SEED)
        clusters = np.arange(n_clusters)
        rng.shuffle(clusters)

        n_test = max(1, int(round(n_clusters * test_ratio)))
        n_val  = max(1, int(round(n_clusters * val_ratio)))

        test_clusters  = clusters[:n_test]
        val_clusters   = clusters[n_test:n_test + n_val]
        train_clusters = clusters[n_test + n_val:]

        train_idx = np.where(np.isin(labels, train_clusters))[0]
        val_idx   = np.where(np.isin(labels, val_clusters))[0]
        test_idx  = np.where(np.isin(labels, test_clusters))[0]

        return train_idx, val_idx, test_idx

    def _compute_kfold_splits(
        self,
        spatial_coords: np.ndarray,
        k: int = 5,
        cell_types: Optional[np.ndarray] = None,
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        
        from sklearn.model_selection import StratifiedKFold

        n_cells = spatial_coords.shape[0]
        n_clusters = max(k * 20, int(1 / max(1 / n_cells, 1 / 500)))
        n_clusters = min(n_clusters, n_cells)

        # ── Step 1: spatial KMeans ────────────────────────────────────────
        # Seed is mixed with a large prime so KMeans partitioning is fully
        # independent of any upstream RNG state changes between runs.
        _kmeans_seed = int(self.cfg.SEED) ^ 0x6B43A9B5
        set_seed(_kmeans_seed)
        kmeans = KMeans(n_clusters=n_clusters, random_state=_kmeans_seed, n_init="auto")
        cell_cluster_labels = kmeans.fit_predict(spatial_coords)   # (N,) int

        # ── Step 2: dominant tissue-region label per cluster ──────────────
        if cell_types is not None:
            # Majority cell type inside each spatial cluster
            cluster_region_labels = np.empty(n_clusters, dtype=object)
            for c in range(n_clusters):
                mask = cell_cluster_labels == c
                if not mask.any():
                    cluster_region_labels[c] = "unknown"
                    continue
                types_in_cluster = cell_types[mask]
                unique, counts = np.unique(types_in_cluster, return_counts=True)
                cluster_region_labels[c] = unique[counts.argmax()]
            print(f"  Tissue-region stratification: {len(np.unique(cluster_region_labels))} "
                  f"unique regions across {n_clusters} spatial clusters.")
        else:
            # Fallback: label each cluster by its spatial quadrant (4 regions)
            centroids = kmeans.cluster_centers_          # (n_clusters, 2)
            x_mid = np.median(centroids[:, 0])
            y_mid = np.median(centroids[:, 1])
            cluster_region_labels = np.where(
                centroids[:, 0] >= x_mid,
                np.where(centroids[:, 1] >= y_mid, "NE", "SE"),
                np.where(centroids[:, 1] >= y_mid, "NW", "SW"),
            )
            print(f"  No cell_types provided — using spatial quadrant as region label.")

        # StratifiedKFold needs integer class codes
        unique_regions, region_codes = np.unique(cluster_region_labels, return_inverse=True)
        print(f"  Regions: {list(unique_regions)}")

        # ── Step 3: stratified K-fold split of clusters ───────────────────
        # BUG FIX 1: the outer StratifiedKFold must produce test ≈ TEST_RATIO,
        # NOT test = 1/K_FOLDS.  When K_FOLDS=25 but TEST_RATIO=0.15, using
        # n_splits=k gave test=1/25=4% instead of 15%.
        # Fix: derive n_splits_test from TEST_RATIO so 1/n_splits_test ≈ TEST_RATIO,
        # independent of K_FOLDS.
        _test_ratio = float(getattr(self.cfg, "TEST_RATIO", 0.15))
        _val_ratio  = float(getattr(self.cfg, "VAL_RATIO",  0.15))
        n_splits_test = max(2, round(1.0 / _test_ratio))  # e.g. 0.15 → 7 (14.3%)

        _skf_seed = int(self.cfg.SEED) ^ 0x9E3779B9
        clusters  = np.arange(n_clusters)
        # Use n_splits_test for the outer (test) split, iterate k times for k folds
        skf = StratifiedKFold(n_splits=n_splits_test, shuffle=True,
                              random_state=_skf_seed)
        # Collect only the first k splits so we get exactly K_FOLDS folds
        all_outer_splits = list(skf.split(clusters, region_codes))
        # If k > n_splits_test, wrap around (each outer split is used at most twice)
        outer_splits = [all_outer_splits[i % n_splits_test] for i in range(k)]

        # BUG FIX 2: val fraction was hardcoded to 10% of remaining.
        # With TEST_RATIO=0.15, remaining=85%, so val should be
        # VAL_RATIO / (1 - TEST_RATIO) = 0.15/0.85 ≈ 17.65% of remaining,
        # NOT 10%.  The fix uses cfg.VAL_RATIO and cfg.TEST_RATIO directly.
        _val_of_remaining = _val_ratio / (1.0 - _test_ratio)   # ≈ 0.1765

        folds: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for fold_i, (remaining_clusters, test_clusters) in enumerate(outer_splits):
            # ── Step 4: stratified val hold-out from remaining clusters ───
            # Reserve VAL_RATIO/(1-TEST_RATIO) of remaining clusters as val
            remaining_region_codes = region_codes[remaining_clusters]
            n_val_clusters = max(1, int(round(len(remaining_clusters)
                                             * _val_of_remaining)))

            # Use a second StratifiedKFold pass to pick val proportionally.
            # n_val_folds = round(1 / _val_of_remaining) ≈ 6 for 17.65%
            n_val_folds = max(2, round(1.0 / _val_of_remaining))
            # Derive val-split seed independently per fold using Knuth hash
            _val_seed = int(self.cfg.SEED) ^ (0x517CC1B727220A95 * (fold_i + 1) & 0xFFFFFFFF)
            inner_skf = StratifiedKFold(
                n_splits=n_val_folds, shuffle=True,
                random_state=_val_seed,
            )
            # Take the first split's "test" portion as our val set
            try:
                train_part, val_part = next(
                    inner_skf.split(remaining_clusters, remaining_region_codes)
                )
                train_clusters = remaining_clusters[train_part]
                val_clusters   = remaining_clusters[val_part]
                # Trim val to exactly n_val_clusters if the fold is oversized
                if len(val_clusters) > n_val_clusters * 2:
                    val_clusters = val_clusters[:n_val_clusters]
                    train_clusters = np.concatenate(
                        [train_clusters, val_clusters[n_val_clusters:]]
                    )
            except Exception:
                # Fallback: plain slice (rare edge case with very few clusters)
                train_clusters = remaining_clusters[n_val_clusters:]
                val_clusters   = remaining_clusters[:n_val_clusters]

            test_idx  = np.where(np.isin(cell_cluster_labels, test_clusters))[0]
            val_idx   = np.where(np.isin(cell_cluster_labels, val_clusters))[0]
            train_idx = np.where(np.isin(cell_cluster_labels, train_clusters))[0]

            # Diagnostic: region distribution in test fold
            if cell_types is not None:
                test_types, test_counts = np.unique(cell_types[test_idx], return_counts=True)
                dist_str = ", ".join(
                    f"{t}:{c}" for t, c in zip(test_types, test_counts)
                )
                print(f"  Fold {fold_i + 1} test region mix → {dist_str}")

            folds.append((train_idx, val_idx, test_idx))

        return folds

    def get_kfold_splits(
        self,
        spatial_coords: np.ndarray,
        k: int = 5,
        cell_types: Optional[np.ndarray] = None,
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Return k (train_idx, val_idx, test_idx) folds, loading from disk if
        a saved split exists; otherwise compute, save, and return fresh folds.

        _compute_kfold_splits() is seeded (KMeans random_state + StratifiedKFold
        random_state are all derived deterministically from cfg.SEED), but that
        is NOT a hard reproducibility guarantee: KMeans' Lloyd-iteration distance
        sums run through multi-threaded BLAS/OpenMP, whose floating-point
        reduction order can vary between separate process runs even with an
        identical random_state and identical input data. A handful of cells
        sitting near a cluster boundary can flip to a neighboring cluster on
        any given run, which then cascades through the StratifiedKFold step
        into different train/val/test cell counts and membership per fold —
        exactly the kind of small (tens-of-cells) fold-to-fold drift that
        produces different downstream metrics and consensus/selection-frequency
        results across otherwise-identical runs.

        Saving and reloading indices (same pattern as spatial_train_val_test_split
        above) is the only robust way to guarantee identical folds across runs,
        independent of any underlying library's floating-point determinism.

        Priority:
          1. cfg.KFOLD_SPLIT_PATH (explicit override) — load & verify.
          2. Default path <output_dir>/kfold_split_indices.npz — load if present.
          3. Compute fresh folds via _compute_kfold_splits, save, and return.
        """
        n_cells = spatial_coords.shape[0]

        explicit_path = self.cfg.KFOLD_SPLIT_PATH.strip() if self.cfg.KFOLD_SPLIT_PATH else ""
        default_path  = os.path.join(self.cfg.OUTPUT_DIR, "kfold_split_indices.npz")
        load_path     = explicit_path if explicit_path else (
            default_path if os.path.isfile(default_path) else ""
        )

        if load_path:
            try:
                saved = np.load(load_path)
                saved_k = int(saved["k"])
                if saved_k != k:
                    raise ValueError(
                        f"Saved split has k={saved_k} folds but k={k} was requested."
                    )
                folds: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
                for fold_i in range(k):
                    train_idx = saved[f"train_idx_{fold_i}"]
                    val_idx   = saved[f"val_idx_{fold_i}"]
                    test_idx  = saved[f"test_idx_{fold_i}"]

                    max_idx = max(train_idx.max(), val_idx.max(), test_idx.max())
                    if max_idx >= n_cells:
                        raise ValueError(
                            f"Saved fold {fold_i} has max index {max_idx} but "
                            f"current data has only {n_cells} cells. The split "
                            f"was saved from a different preprocessing run. "
                            f"Delete {load_path} to recompute."
                        )
                    total_saved = len(train_idx) + len(val_idx) + len(test_idx)
                    if total_saved != n_cells:
                        raise ValueError(
                            f"Saved fold {fold_i} covers {total_saved} cells but "
                            f"current data has {n_cells}. Delete {load_path} to recompute."
                        )
                    folds.append((train_idx, val_idx, test_idx))

                print(f"✓ Loaded saved k-fold split from: {load_path}")
                print(f"  {k} folds reused as-is (no recomputation).")
                return folds

            except (KeyError, ValueError, EOFError, OSError, zipfile.BadZipFile) as exc:
                # Broadened beyond (KeyError, ValueError): a truncated/corrupted
                # .npz (interrupted write, disk-full, partial download/transfer)
                # raises EOFError/OSError/zipfile.BadZipFile, not KeyError or
                # ValueError — those were previously uncaught here and would
                # crash the whole pipeline instead of falling through to the
                # "recompute" path below.
                if explicit_path:
                    raise RuntimeError(
                        f"Failed to load k-fold split from --kfold_split_path="
                        f"{explicit_path}: {exc}"
                    ) from exc
                print(f"⚠ Could not load k-fold split from {load_path}: {exc}")
                print("  Recomputing folds and overwriting the file…")

        print("  Computing spatial KMeans k-fold splits…")
        folds = self._compute_kfold_splits(spatial_coords, k=k, cell_types=cell_types)

        save_kwargs = {"k": k}
        for fold_i, (train_idx, val_idx, test_idx) in enumerate(folds):
            save_kwargs[f"train_idx_{fold_i}"] = train_idx
            save_kwargs[f"val_idx_{fold_i}"]   = val_idx
            save_kwargs[f"test_idx_{fold_i}"]  = test_idx
        np.savez(default_path, **save_kwargs)
        print(f"✓ K-fold split indices saved to: {default_path}")
        print("  (Delete this file only if you intentionally want new folds.)")

        return folds

    def spatial_train_val_test_split(
        self,
        spatial_coords: np.ndarray,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (train_idx, val_idx, test_idx), loading from disk if a saved
        split exists; otherwise compute, save, and return a fresh split.

        Priority:
          1. cfg.SPLIT_PATH (explicit --split_path argument) — load & verify.
          2. Default path <output_dir>/split_indices.npz — load if present.
          3. Compute a new split, save to <output_dir>/split_indices.npz.

        Saving and reloading indices is the only robust way to guarantee
        identical splits across runs when upstream preprocessing (HVG
        selection, graph construction) may shift the global RNG state.
        """
        n_cells = spatial_coords.shape[0]

        # ── Determine which file to try ──────────────────────────────────
        explicit_path = self.cfg.SPLIT_PATH.strip() if self.cfg.SPLIT_PATH else ""
        default_path  = os.path.join(self.cfg.OUTPUT_DIR, "split_indices.npz")
        load_path     = explicit_path if explicit_path else (
            default_path if os.path.isfile(default_path) else ""
        )

        # ── Try to load ───────────────────────────────────────────────────
        if load_path:
            try:
                saved = np.load(load_path)
                train_idx = saved["train_idx"]
                val_idx   = saved["val_idx"]
                test_idx  = saved["test_idx"]

                # Sanity: indices must be valid for the current cell count.
                max_idx = max(train_idx.max(), val_idx.max(), test_idx.max())
                if max_idx >= n_cells:
                    raise ValueError(
                        f"Saved split has max index {max_idx} but current "
                        f"data has only {n_cells} cells. The split was saved "
                        f"from a different preprocessing run. Delete "
                        f"{load_path} to recompute."
                    )
                total_saved = len(train_idx) + len(val_idx) + len(test_idx)
                if total_saved != n_cells:
                    raise ValueError(
                        f"Saved split covers {total_saved} cells but current "
                        f"data has {n_cells}. Delete {load_path} to recompute."
                    )

                print(f"✓ Loaded saved split from: {load_path}")
                print(f"  Train cells: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")
                return train_idx, val_idx, test_idx

            except (KeyError, ValueError, EOFError, OSError, zipfile.BadZipFile) as exc:
                # See matching comment in get_kfold_splits: broadened beyond
                # (KeyError, ValueError) to also catch corrupted/truncated
                # .npz files instead of letting them crash the pipeline.
                # If an explicit path was given and it fails, abort loudly.
                if explicit_path:
                    raise RuntimeError(
                        f"Failed to load split from --split_path={explicit_path}: {exc}"
                    ) from exc
                # Default-path failure: fall through and recompute.
                print(f"⚠ Could not load split from {load_path}: {exc}")
                print("  Recomputing split and overwriting the file…")

        # ── Compute a new split ───────────────────────────────────────────
        print("  Computing spatial KMeans split…")
        train_idx, val_idx, test_idx = self._compute_split(
            spatial_coords, val_ratio=val_ratio, test_ratio=test_ratio,
        )

        # ── Save so every future run reuses the same split ────────────────
        np.savez(default_path, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
        print(f"✓ Split indices saved to: {default_path}")
        print("  (Delete this file only if you intentionally want a new split.)")

        return train_idx, val_idx, test_idx

    def load_and_preprocess(self) -> Dict[str, Any]:
        print(f"  Loading .h5ad data from: {self.cfg.DATA_PATH}")
        try:
            self.adata = sc.read_h5ad(self.cfg.DATA_PATH)
        except Exception as exc:
            raise RuntimeError(
                f"Could not read DATA_PATH as an AnnData .h5ad file: "
                f"{self.cfg.DATA_PATH!r}. The path exists but its contents "
                f"could not be parsed ({exc}). Verify it is a valid .h5ad file."
            ) from exc
        try:
            lr_pairs = pd.read_csv(self.cfg.LR_PATH)
        except Exception as exc:
            raise RuntimeError(
                f"Could not read LR_PATH as a CSV file: {self.cfg.LR_PATH!r}. "
                f"The path exists but its contents could not be parsed "
                f"({exc}). Verify it is a valid CSV with ligand/receptor columns."
            ) from exc
        if lr_pairs.empty:
            raise ValueError(
                f"LR_PATH={self.cfg.LR_PATH!r} parsed to an empty table — no "
                f"ligand-receptor pairs to model."
            )

        if "spatial" not in self.adata.obsm:
            raise ValueError("Expected spatial coordinates in adata.obsm['spatial'].")

        print(f"  Raw data shape: {self.adata.shape}")

        print(f"  Filtering cells min_genes={self.cfg.MIN_GENES}, genes min_cells={self.cfg.MIN_CELLS}")
        sc.pp.filter_cells(self.adata, min_genes=self.cfg.MIN_GENES)
        sc.pp.filter_genes(self.adata, min_cells=self.cfg.MIN_CELLS)

        if "counts" not in self.adata.layers:
            self.adata.layers["counts"] = self.adata.X.copy()

        print("  Normalizing and log-transforming expression...")
        sc.pp.normalize_total(self.adata, target_sum=1e4)
        sc.pp.log1p(self.adata)

        print(f"  Selecting top {self.cfg.N_HVGS} highly variable genes...")
        n_hvgs = min(self.cfg.N_HVGS, self.adata.shape[1])
        # Re-seed here: seurat_v3 HVG selection uses randomness internally;
        # without this the selected gene set can differ between runs.
        set_seed(self.cfg.SEED)
        sc.pp.highly_variable_genes(
            self.adata, n_top_genes=n_hvgs, flavor="seurat_v3", check_values=False,
        )
        # Force LR genes into HVG set — prevents lr_features from being all-zero
        # (reuses the already-loaded/validated `lr_pairs` instead of re-reading
        # LR_PATH from disk a second time — same data, one fewer I/O failure point)
        ligand_col, receptor_col = self.detect_lr_columns(lr_pairs)
        lr_genes = set(lr_pairs[ligand_col].dropna().tolist() +
                       lr_pairs[receptor_col].dropna().tolist())
        lr_present = lr_genes & set(self.adata.var_names)
        if lr_present:
            self.adata.var.loc[list(lr_present), "highly_variable"] = True
            print(f"  Forced {len(lr_present)} LR genes into feature set")
        hvg_adata = self.adata[:, self.adata.var.highly_variable].copy()

        node_features = self.get_hvg_features(hvg_adata)

        self.cfg.N_GENES = node_features.shape[1]
        self.cfg.N_CELLS = self.adata.shape[0]
        self.cfg.N_LR = len(lr_pairs)

        spatial_coords = np.asarray(self.adata.obsm["spatial"], dtype=np.float32)
        adj_matrix = self.construct_spatial_graph(spatial_coords, n_neighbors=self.cfg.N_NEIGHBORS)
        edge_index = self.to_edge_index(adj_matrix)

        ct_prior, type_to_idx = self.get_cell_type_prior(self.adata, adj_matrix)
        self.cfg.CT_PRIOR = ct_prior
        self.cfg.CT_TYPE_TO_IDX = type_to_idx
        self.cfg.N_CT = len(type_to_idx)   # still used by CT_PRIOR / edge_ifw abundance correction
        print(f"✓ Cell-type prior stored in cfg.")

        lr_features = self.get_lr_features(self.adata, lr_pairs)

        cell_type_col = self.detect_cell_type_column(self.adata)
        cell_types = self.adata.obs[cell_type_col].astype(str).values

        print("✓ Final processed shapes:")
        print(f"  node_features: {tuple(node_features.shape)}")
        print(f"  lr_features: {tuple(lr_features.shape)}")
        print(f"  edge_index: {tuple(edge_index.shape)}")
        print(f"  spatial_coords: {tuple(spatial_coords.shape)}")
        print(f"  N_LR: {self.cfg.N_LR}")

        return {
            "node_features": node_features,
            "lr_features": lr_features,
            "edge_index": edge_index,
            "spatial_coords": spatial_coords,
            "cell_types": cell_types,
            "cell_names": self.adata.obs_names.tolist(),
            "adj_matrix": adj_matrix,
            "lr_pairs": lr_pairs,
        }

    def get_pyg_data(self) -> Tuple[Data, Dict[str, Any]]:
        full_data = self.load_and_preprocess()

        cell_type_labels = full_data["cell_types"]
        type_to_idx = self.cfg.CT_TYPE_TO_IDX or {
            t: i for i, t in enumerate(sorted(set(cell_type_labels)))
        }
        cell_type_indices = np.array([type_to_idx.get(t, 0) for t in cell_type_labels],
                                     dtype=np.int64)

        data = Data(
            x=full_data["node_features"],
            edge_index=full_data["edge_index"],
            lr_features=full_data["lr_features"],
            spatial_coords=torch.tensor(full_data["spatial_coords"], dtype=torch.float32),
            cell_type_idx=torch.tensor(cell_type_indices, dtype=torch.long),
        )
       
        data.cell_types = np.array(cell_type_labels)

        return data, full_data


# =====================================================================
# 7. Metrics
# =====================================================================