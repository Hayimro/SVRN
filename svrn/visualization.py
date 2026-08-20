# -*- coding: utf-8 -*-
"""
svrn.visualization
===================
Publication-quality plotting: per-run diagnostic plots (training/val
loss curves, influence-score maps, uncertainty panels — via
:class:`SVRNVisualizer`) and multi-run/K-fold consensus plots
(selection-frequency maps, ranked communication corridors — via
:class:`ConsensusPlotter`).

"""

import os
import warnings
from typing import Dict, Any, List, Optional, Tuple

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import scanpy as sc
from anndata import AnnData
from scipy.sparse import csr_matrix, issparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

# -----------------------------
# Publication plot style
# -----------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
})


class SVRNVisualizer:
    def __init__(
        self,
        influence_scores: np.ndarray,
        spatial_coords: np.ndarray,
        cell_types: np.ndarray,
        adj_matrix: csr_matrix,
        output_dir: str,
        influence_std: Optional[np.ndarray] = None,
        influence_ci_lo: Optional[np.ndarray] = None,
        influence_ci_hi: Optional[np.ndarray] = None,
        global_cell_idx: Optional[np.ndarray] = None,
    ):
        self.influence = np.asarray(influence_scores).flatten()
        self.coords = np.asarray(spatial_coords)
        self.cell_types = np.asarray(cell_types)
        self.adj = adj_matrix
        self.output_dir = output_dir
        # Uncertainty arrays (None when MC sampling was skipped)
        self.influence_std = np.asarray(influence_std).flatten() if influence_std is not None else None
        self.influence_ci_lo = np.asarray(influence_ci_lo).flatten() if influence_ci_lo is not None else None
        self.influence_ci_hi = np.asarray(influence_ci_hi).flatten() if influence_ci_hi is not None else None
        # Maps a LOCAL node index (0..len(coords)-1, i.e. position within this
        # fold's test subset) back to the GLOBAL cell index in the full
        # AnnData / adjacency matrix. Needed because corridor paths are
        # computed on the local (per-fold) subgraph, but relay-cell identity
        # only means something consistent across folds/runs in global terms.
        self.global_cell_idx = (
            np.asarray(global_cell_idx) if global_cell_idx is not None else None
        )
        os.makedirs(self.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Existing plots
    # ------------------------------------------------------------------
    def plot_loss_curve(self, train_losses: List[float], val_losses: List[float], val_epochs: Optional[List[int]] = None) -> None:
        if len(train_losses) == 0 and len(val_losses) == 0:
            print("! No loss data to plot.")
            return

        fig, ax = plt.subplots(figsize=(9, 5))

        train_x = np.arange(len(train_losses))
        if val_epochs and len(val_epochs) == len(val_losses):
            val_x = np.array(val_epochs)
        else:
            val_x = np.arange(len(val_losses))

        def _ema(values: List[float], alpha: float = 0.15) -> np.ndarray:
            out = np.empty(len(values))
            out[0] = values[0]
            for i in range(1, len(values)):
                out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
            return out

        if len(train_losses) > 0:
            raw = np.array(train_losses)
            smooth = _ema(train_losses, alpha=0.15)
            ax.plot(train_x, raw, color="#4C8CBF", alpha=0.25, linewidth=1.0, label="_nolegend_")
            ax.plot(train_x, smooth, color="#4C8CBF", linewidth=2.2, label="Training Loss (EMA)")

        if len(val_losses) > 0:
            ax.plot(val_x, val_losses, color="#E07B39", linewidth=2.2, marker="o",
                    markersize=3.5, label="Validation Loss")

        if len(val_losses) > 0:
            best_idx = int(np.argmin(val_losses))
            ax.axvline(val_x[best_idx], color="#888888", linestyle=":", linewidth=1.2,
                       label=f"Best val ({val_losses[best_idx]:.4f} @ ep {val_x[best_idx]})")

        ax.set_title("SVRN Training and Validation Loss", fontsize=16, pad=10)
        ax.set_xlabel("Epoch", fontsize=14)
        ax.set_ylabel("Loss", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_xlim(left=0)
        fig.tight_layout()
        save_path = os.path.join(self.output_dir, "loss_curve.png")
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f"✓ Loss curve saved: {save_path}")

    def plot_influence_distribution(self) -> None:
        plt.figure(figsize=(8, 6))
        sns.histplot(self.influence, kde=True, bins=50)
        plt.axvline(np.mean(self.influence), linestyle="--", label=f"Mean: {np.mean(self.influence):.3f}")
        plt.title("SVRN Influence Distribution")
        plt.xlabel("Influence Score")
        plt.ylabel("Cell Count")
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, "1_influence_distribution.png"), dpi=300)
        plt.close()

    def plot_spatial_influence(self) -> None:
        plt.figure(figsize=(10, 8))
        vmax = np.percentile(self.influence, 98)
        scatter = plt.scatter(
            self.coords[:, 0], self.coords[:, 1],
            c=self.influence, s=2.0, alpha=0.85, vmax=vmax,
        )
        plt.colorbar(scatter, label="Relay Influence Intensity")
        plt.title("Spatial Communication Influence")
        plt.axis("equal")
        plt.savefig(os.path.join(self.output_dir, "2_spatial_influence.png"), dpi=300)
        plt.close()

    def plot_cell_type_enrichment(self) -> None:
        df = pd.DataFrame({"Influence": self.influence, "CellType": self.cell_types})
        plt.figure(figsize=(12, 6))
        order = df.groupby("CellType")["Influence"].median().sort_values(ascending=False).index
        sns.boxplot(x="CellType", y="Influence", data=df, order=order)
        plt.xticks(rotation=45, ha="right")
        plt.title("Influence Enrichment by Cell Type")
        plt.savefig(os.path.join(self.output_dir, "3_cell_type_enrichment.png"), dpi=300)
        plt.close()

    # ------------------------------------------------------------------
    # NEW: Uncertainty plots
    # ------------------------------------------------------------------
    def plot_uncertainty_spatial(self) -> None:
        """Spatial map of per-cell influence uncertainty (posterior std from MC sampling)."""
        if self.influence_std is None:
            print("! Skipping uncertainty spatial plot: no MC samples available.")
            return

        fig, axes = plt.subplots(1, 2, figsize=(18, 7))

        # Panel A — mean influence
        vmax_mean = np.percentile(self.influence, 98)
        sc0 = axes[0].scatter(
            self.coords[:, 0], self.coords[:, 1],
            c=self.influence, s=2.0, alpha=0.85,
            cmap="viridis", vmax=vmax_mean,
        )
        plt.colorbar(sc0, ax=axes[0], label="Mean Influence Score")
        axes[0].set_title("Mean Influence (MC)", fontsize=14)
        axes[0].axis("equal")

        # Panel B — posterior std (epistemic uncertainty)
        vmax_std = np.percentile(self.influence_std, 98)
        sc1 = axes[1].scatter(
            self.coords[:, 0], self.coords[:, 1],
            c=self.influence_std, s=2.0, alpha=0.85,
            cmap="hot_r", vmax=vmax_std,
        )
        plt.colorbar(sc1, ax=axes[1], label="Posterior Std (Uncertainty)")
        axes[1].set_title("Influence Uncertainty (MC Std)", fontsize=14)
        axes[1].axis("equal")

        fig.suptitle("SVRN Influence — Mean vs Uncertainty", fontsize=16, y=1.01)
        fig.tight_layout()
        save_path = os.path.join(self.output_dir, "5_uncertainty_spatial.png")
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f"✓ Uncertainty spatial plot saved: {save_path}")

    def plot_uncertainty_by_cell_type(self) -> None:
        """Violin plot: per-cell-type distribution of uncertainty (std) from MC sampling."""
        if self.influence_std is None:
            print("! Skipping uncertainty cell-type plot: no MC samples available.")
            return

        df = pd.DataFrame({
            "CellType": self.cell_types,
            "Mean Influence": self.influence,
            "Uncertainty (Std)": self.influence_std,
        })
        order = df.groupby("CellType")["Mean Influence"].median().sort_values(ascending=False).index

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        sns.violinplot(
            data=df, x="CellType", y="Mean Influence",
            order=order, ax=axes[0], palette="muted", inner="box",
        )
        axes[0].set_title("Mean Influence by Cell Type", fontsize=13)
        axes[0].tick_params(axis="x", rotation=40)

        sns.violinplot(
            data=df, x="CellType", y="Uncertainty (Std)",
            order=order, ax=axes[1], palette="flare", inner="box",
        )
        axes[1].set_title("Influence Uncertainty (Std) by Cell Type", fontsize=13)
        axes[1].tick_params(axis="x", rotation=40)

        fig.suptitle("MC-Sampled Influence: Mean and Uncertainty per Cell Type", fontsize=15)
        fig.tight_layout()
        save_path = os.path.join(self.output_dir, "6_uncertainty_by_cell_type.png")
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f"✓ Uncertainty cell-type plot saved: {save_path}")

    def plot_confidence_intervals(self, top_n: int = 30) -> None:
        """95 % credible-interval error-bar chart for the top-N highest-influence cells."""
        if self.influence_ci_lo is None or self.influence_ci_hi is None:
            print("! Skipping CI plot: no MC samples available.")
            return

        order = np.argsort(self.influence)[::-1][:top_n]
        means = self.influence[order]
        lo = self.influence_ci_lo[order]
        hi = self.influence_ci_hi[order]
        labels = [f"{self.cell_types[i]}\n(#{i})" for i in order]
        # Clamp lo/hi so error bars are never negative (can happen after
        # independent normalisation of the CI bounds).
        lo = np.minimum(lo, means)
        hi = np.maximum(hi, means)
        errors = np.clip(np.array([means - lo, hi - means]), 0.0, None)

        fig, ax = plt.subplots(figsize=(max(10, top_n * 0.45), 6))
        x = np.arange(top_n)
        ax.bar(x, means, color="#4C8CBF", alpha=0.75, label="Mean influence")
        ax.errorbar(
            x, means, yerr=errors,
            fmt="none", ecolor="#E07B39", elinewidth=1.5, capsize=3,
            label="95 % credible interval",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
        ax.set_ylabel("Influence Score")
        ax.set_title(f"Top-{top_n} Cells: Mean Influence ± 95 % CI (MC Sampling)")
        ax.legend(fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        fig.tight_layout()
        save_path = os.path.join(self.output_dir, "7_confidence_intervals_top_cells.png")
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f"✓ Confidence interval plot saved: {save_path}")

    def plot_uncertainty_vs_influence(self) -> None:
        """Scatter: mean influence vs uncertainty — reveals high-signal / low-confidence cells."""
        if self.influence_std is None:
            print("! Skipping uncertainty scatter: no MC samples available.")
            return

        fig, ax = plt.subplots(figsize=(8, 6))
        unique_types = np.unique(self.cell_types)
        cmap = plt.get_cmap("tab10", len(unique_types))
        for k, ct in enumerate(unique_types):
            mask = self.cell_types == ct
            ax.scatter(
                self.influence[mask], self.influence_std[mask],
                s=4, alpha=0.6, color=cmap(k), label=ct,
            )
        ax.set_xlabel("Mean Influence Score")
        ax.set_ylabel("Uncertainty (Posterior Std)")
        ax.set_title("Influence vs Uncertainty by Cell Type")
        ax.legend(fontsize=7, markerscale=2, loc="upper left", ncol=2)
        ax.grid(linestyle="--", alpha=0.4)
        fig.tight_layout()
        save_path = os.path.join(self.output_dir, "8_influence_vs_uncertainty.png")
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f"✓ Influence-vs-uncertainty scatter saved: {save_path}")

    # ------------------------------------------------------------------
    # NEW: Communication corridor visualisation
    # ------------------------------------------------------------------
    def _extract_corridors(
        self,
        top_k_hubs: int = 15,
        min_hops: int = 3,
        max_hops: int = 25,
        max_corridors: int = 8,
    ) -> List[Dict]:
        """Extract weighted shortest-path corridors between high-influence hub cells.

        Edge weight = 1 - mean_influence(u, v) so that high-influence edges are
        traversed preferentially (lower effective distance).
        """
        G = nx.from_scipy_sparse_array(self.adj, create_using=nx.DiGraph())

        # Assign influence-aware edge weights
        for u, v in G.edges():
            w = max(1.0 - 0.5 * (self.influence[u] + self.influence[v]), 1e-3)
            G[u][v]["weight"] = w

        hub_idx = np.argsort(self.influence)[::-1][:top_k_hubs]
        corridors: List[Dict] = []

        for i, src in enumerate(hub_idx):
            for dst in hub_idx[i + 1:]:
                if len(corridors) >= max_corridors:
                    break
                try:
                    path = nx.shortest_path(G, int(src), int(dst), weight="weight")
                    n_hops = len(path) - 1
                    if not (min_hops <= n_hops <= max_hops):
                        continue
                    mean_inf = float(self.influence[path].mean())
                    spatial_len = float(
                        np.sum(np.linalg.norm(np.diff(self.coords[path], axis=0), axis=1))
                    )
                    # Dominant cell type along the corridor
                    types_along = self.cell_types[path]
                    unique, counts = np.unique(types_along, return_counts=True)
                    dominant_type = unique[np.argmax(counts)]
                    # Relay cells = every intermediate node on the path,
                    # i.e. excluding the two hub endpoints.
                    relay_path_local = path[1:-1]
                    if self.global_cell_idx is not None:
                        path_global = self.global_cell_idx[path].tolist()
                        relay_path_global = self.global_cell_idx[relay_path_local].tolist()
                    else:
                        # No global mapping available (e.g. called outside
                        # make_plots) — fall back to local indices.
                        path_global = list(path)
                        relay_path_global = list(relay_path_local)

                    corridors.append({
                        "path": path,
                        "n_hops": n_hops,
                        "mean_influence": mean_inf,
                        "spatial_length": spatial_len,
                        "dominant_type": dominant_type,
                        "src_type": self.cell_types[src],
                        "dst_type": self.cell_types[dst],
                        "relay_path_local": relay_path_local,
                        "relay_path_global": relay_path_global,
                        "path_global": path_global,
                    })
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
            if len(corridors) >= max_corridors:
                break

        return corridors

    def plot_communication_corridors(self) -> None:
        """Spatial map of top communication corridors with influence-weighted line width."""
        corridors = self._extract_corridors()

        fig, ax = plt.subplots(figsize=(11, 9))

        # Background: all cells coloured by influence
        vmax = np.percentile(self.influence, 98)
        sc = ax.scatter(
            self.coords[:, 0], self.coords[:, 1],
            c=self.influence, s=1.5, alpha=0.5,
            cmap="Greys", vmax=vmax, zorder=1,
        )
        plt.colorbar(sc, ax=ax, label="Influence Score", shrink=0.7)

        if not corridors:
            ax.set_title("Communication Corridors (none found)")
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "4_communication_corridors.png"), dpi=300)
            plt.close(fig)
            return

        cmap_lines = plt.get_cmap("tab10", len(corridors))
        inf_vals = np.array([c["mean_influence"] for c in corridors])
        inf_min, inf_max = inf_vals.min(), inf_vals.max() + 1e-8

        for k, corr in enumerate(corridors):
            path = corr["path"]
            path_coords = self.coords[path]

            # Line width proportional to mean influence (1–5 pt)
            lw = 1.0 + 4.0 * (corr["mean_influence"] - inf_min) / (inf_max - inf_min)
            color = cmap_lines(k)

            ax.plot(
                path_coords[:, 0], path_coords[:, 1],
                color=color, linewidth=lw, alpha=0.85, zorder=3,
                label=(
                    f"C{k+1}: {corr['src_type']} → {corr['dst_type']} "
                    f"({corr['n_hops']} hops, inf={corr['mean_influence']:.2f})"
                ),
            )
            # Mark endpoints
            ax.scatter(
                [path_coords[0, 0], path_coords[-1, 0]],
                [path_coords[0, 1], path_coords[-1, 1]],
                s=40, color=color, edgecolors="black", linewidths=0.5,
                zorder=4,
            )

        ax.legend(fontsize=7, loc="upper left", framealpha=0.85)
        ax.set_title(
            f"Top-{len(corridors)} Communication Corridors\n"
            "(line width ∝ mean influence; endpoints marked)",
            fontsize=14,
        )
        ax.axis("equal")
        fig.tight_layout()
        save_path = os.path.join(self.output_dir, "4_communication_corridors.png")
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f"✓ Communication corridors plot saved: {save_path}")

    def plot_corridor_summary(self) -> None:
        """Bar chart: corridor statistics — spatial length, hop count, and mean influence.

        Also exports corridor statistics to ``4b_corridor_summary.csv`` so that
        the same data can be collected across folds and aggregated into a
        consensus corridor summary for :meth:`ConsensusPlotter.plot_communication_corridors`
        (C4).
        """
        corridors = self._extract_corridors()
        if not corridors:
            print("! No corridors found; skipping corridor summary plot.")
            return

        labels = [f"C{i+1}" for i in range(len(corridors))]
        mean_infs = [c["mean_influence"] for c in corridors]
        n_hops = [c["n_hops"] for c in corridors]
        sp_lens = [c["spatial_length"] for c in corridors]

        # ── Export statistics to CSV ──────────────────────────────────────
        corridor_records = []
        for i, (label, corr) in enumerate(zip(labels, corridors)):
            corridor_records.append({
                "corridor_id":    label,
                "n_hops":         corr["n_hops"],
                "mean_influence": corr["mean_influence"],
                "spatial_length": corr["spatial_length"],
                "dominant_type":  corr["dominant_type"],
                "src_type":       corr["src_type"],
                "dst_type":       corr["dst_type"],
                "path_cell_ids":  ",".join(str(p) for p in corr["path"]),
                "n_relays":       len(corr["relay_path_global"]),
                "relay_cell_ids_local":  ",".join(str(p) for p in corr["relay_path_local"]),
                "relay_cell_ids_global": ",".join(str(p) for p in corr["relay_path_global"]),
                "path_cell_ids_global":  ",".join(str(p) for p in corr["path_global"]),
            })
        corridor_df = pd.DataFrame(corridor_records)
        csv_path = os.path.join(self.output_dir, "4b_corridor_summary.csv")
        corridor_df.to_csv(csv_path, index=False)
        print(f"✓ Corridor statistics CSV saved: {csv_path}")

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        colors = plt.get_cmap("tab10", len(corridors))(np.arange(len(corridors)))

        axes[0].bar(labels, mean_infs, color=colors)
        axes[0].set_ylabel("Mean Influence")
        axes[0].set_title("Mean Influence per Corridor")
        axes[0].tick_params(axis="x", rotation=30)

        axes[1].bar(labels, n_hops, color=colors)
        axes[1].set_ylabel("Number of Hops")
        axes[1].set_title("Relay Length (Hops)")
        axes[1].tick_params(axis="x", rotation=30)

        axes[2].bar(labels, sp_lens, color=colors)
        axes[2].set_ylabel("Spatial Length (µm)")
        axes[2].set_title("Physical Corridor Length")
        axes[2].tick_params(axis="x", rotation=30)

        # Annotate dominant cell type
        for ax_i, vals in zip(axes, [mean_infs, n_hops, sp_lens]):
            for j, (label, val, corr) in enumerate(zip(labels, vals, corridors)):
                ax_i.text(
                    j, val * 1.01,
                    corr["dominant_type"][:8],
                    ha="center", va="bottom", fontsize=6, rotation=30,
                )

        fig.suptitle("Communication Corridor Statistics", fontsize=15)
        fig.tight_layout()
        save_path = os.path.join(self.output_dir, "4b_corridor_summary.png")
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f"✓ Corridor summary plot saved: {save_path}")

    # ------------------------------------------------------------------
    # run_all
    # ------------------------------------------------------------------
    def run_all(self, train_losses: List[float], val_losses: List[float], val_epochs: Optional[List[int]] = None) -> None:
        self.plot_loss_curve(train_losses, val_losses, val_epochs)
        self.plot_influence_distribution()
        self.plot_spatial_influence()
        self.plot_cell_type_enrichment()
        # Communication corridors (replaces old spatial trajectories)
        self.plot_communication_corridors()
        self.plot_corridor_summary()
        # Uncertainty plots (only produce output when MC samples were passed in)
        self.plot_uncertainty_spatial()
        self.plot_uncertainty_by_cell_type()
        self.plot_confidence_intervals()
        self.plot_uncertainty_vs_influence()
        print(f"✓ All plots saved in: {self.output_dir}")


# =====================================================================
# Validator
# =====================================================================-e 


class ConsensusPlotter:

    def __init__(
        self,
        consensus_df:       pd.DataFrame,          # from ConsensusInfluence.save()
        fold_scores:        Dict[str, np.ndarray],  # {entry_id: (n_test,) scores}
        fold_test_indices:  Dict[str, np.ndarray],  # {entry_id: (n_test,) global idx}
        all_cell_types:     np.ndarray,             # (N_total,) global labels
        all_spatial_coords: np.ndarray,             # (N_total, 2) global XY
        adj_matrix:         np.ndarray,             # global adjacency (N×N sparse/dense)
        output_dir:         str,
        # Optional per-fold Hill parameters and LR edge probs
        fold_log_n:         Optional[Dict[str, np.ndarray]] = None,  # {entry_id: (n_lr,)}
        fold_log_K:         Optional[Dict[str, np.ndarray]] = None,  # {entry_id: (n_lr,)}
        fold_edge_probs:    Optional[Dict[str, np.ndarray]] = None,  # {entry_id: (n_edges, n_lr)}
        fold_edge_indices:  Optional[Dict[str, np.ndarray]] = None,  # {entry_id: (2, n_edges)}
        lr_pair_names:      Optional[List[str]] = None,              # length n_lr
        # Consensus communication-corridor statistics, aggregated across all
        # run x fold 4b_corridor_summary.csv files (mean ± std per corridor_id).
        # Produced by SVRNPipeline.run_kfold(); see C4 (plot_communication_corridors).
        consensus_corridor_df: Optional[pd.DataFrame] = None,
    ):
        self.consensus_df      = consensus_df
        self.fold_scores       = fold_scores
        self.fold_test_indices = fold_test_indices
        self.all_cell_types    = all_cell_types
        self.all_spatial_coords = all_spatial_coords
        self.adj_matrix        = adj_matrix
        self.output_dir        = output_dir
        self.fold_log_n        = fold_log_n  or {}
        self.fold_log_K        = fold_log_K  or {}
        self.fold_edge_probs   = fold_edge_probs  or {}
        self.fold_edge_indices = fold_edge_indices or {}
        self.lr_pair_names     = lr_pair_names or []
        self.consensus_corridor_df = consensus_corridor_df
        os.makedirs(output_dir, exist_ok=True)

        # Build global per-cell mean and std across folds
        N = len(all_cell_types)
        score_sums   = np.zeros(N, dtype=np.float64)
        score_counts = np.zeros(N, dtype=np.float64)
        score_sq     = np.zeros(N, dtype=np.float64)

        for entry_id, scores in fold_scores.items():
            idx = fold_test_indices[entry_id]
            score_sums[idx]   += scores.flatten()
            score_sq[idx]     += scores.flatten() ** 2
            score_counts[idx] += 1

        covered = score_counts > 0
        self.cell_mean = np.zeros(N, dtype=np.float64)
        self.cell_std  = np.zeros(N, dtype=np.float64)
        self.cell_mean[covered] = score_sums[covered] / score_counts[covered]
        variance = np.zeros(N, dtype=np.float64)
        variance[covered] = (
            score_sq[covered] / score_counts[covered]
            - self.cell_mean[covered] ** 2
        )
        self.cell_std[covered] = np.sqrt(np.clip(variance[covered], 0, None))
        self.covered_mask = covered

        # Normalise mean to [0, 1] for colour mapping
        mn, mx = self.cell_mean[covered].min(), self.cell_mean[covered].max()
        self.cell_mean_norm = np.zeros(N, dtype=np.float64)
        self.cell_mean_norm[covered] = (self.cell_mean[covered] - mn) / (mx - mn + 1e-8)

    # ------------------------------------------------------------------
    # Plot 1 — Influence distribution (histogram + KDE)
    # ------------------------------------------------------------------
    def plot_influence_distribution(self) -> None:
        """Histogram + KDE of consensus influence across all covered cells."""
        from scipy.stats import gaussian_kde

        scores = self.cell_mean_norm[self.covered_mask]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(scores, bins=60, density=True, alpha=0.45,
                color="steelblue", edgecolor="white", label="Histogram")

        xs = np.linspace(scores.min(), scores.max(), 400)
        kde = gaussian_kde(scores, bw_method="scott")
        ax.plot(xs, kde(xs), color="navy", lw=2, label="KDE")

        ax.set_xlabel("Consensus Influence Score (normalised)", fontsize=18)#12
        ax.set_ylabel("Density", fontsize=17)
        ax.set_title("Consensus Influence Distribution\n(mean across all folds)", fontsize=18)#13
        ax.legend(framealpha=0.10)
        fig.tight_layout()
        path = os.path.join(self.output_dir, "C1_influence_distribution.png")
        fig.savefig(path, dpi=300)
        plt.close(fig)
        print(f"✓ Plot saved: {path}")

    # ------------------------------------------------------------------
    # Plot 2 — Cell-type enrichment (bar + optional boxplot)
    # ------------------------------------------------------------------
    def plot_cell_type_enrichment(self) -> None:
        """Bar chart of consensus influence per cell type with fold-std error bars.

        Uses consensus_pct_rank from consensus_df for the bar height (stable
        cross-fold metric) and per-cell fold std to compute per-cell-type spread.
        """
        ct_order = self.consensus_df["cell_type"].tolist()
        consensus_vals = self.consensus_df["consensus_pct_rank"].values
        sel_freq_vals  = self.consensus_df["selection_frequency"].values

        # Per-cell-type std: mean of per-cell fold stds within that type
        ct_std = []
        for ct in ct_order:
            mask = (self.all_cell_types == ct) & self.covered_mask
            ct_std.append(self.cell_std[mask].mean() if mask.any() else 0.0)
        ct_std = np.array(ct_std)

        n_ct = len(ct_order)
        fig, ax = plt.subplots(figsize=(max(8, n_ct * 0.7), 6))

        # Bar colour encodes selection frequency (low → light, high → dark)
        norm_sf = (sel_freq_vals - sel_freq_vals.min()) / (
            sel_freq_vals.max() - sel_freq_vals.min() + 1e-8
        )
        cmap = plt.get_cmap("Blues")
        colors = [cmap(0.35 + 0.55 * v) for v in norm_sf]

        bars = ax.bar(ct_order, consensus_vals, color=colors,
                      edgecolor="grey", linewidth=0.6, yerr=ct_std,
                      capsize=4, error_kw={"elinewidth": 1.2, "ecolor": "black"})

        # Colour-bar for selection frequency
        sm = plt.cm.ScalarMappable(
            cmap=cmap,
            norm=plt.Normalize(vmin=sel_freq_vals.min(), vmax=sel_freq_vals.max()),
        )
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label("Selection Frequency", fontsize=17)#10

        ax.set_xlabel("Cell Type", fontsize=17)
        ax.set_ylabel("Consensus Influence (pct rank)", fontsize=18)#12
        ax.set_title("Cell-type Enrichment of Consensus Influence\n"
                     "(bar height = consensus rank; colour = selection frequency; "
                     "error bars = mean fold std)", fontsize=17)#11
        ax.tick_params(axis="x", rotation=40)
        fig.tight_layout()
        path = os.path.join(self.output_dir, "C2_celltype_enrichment.png")
        fig.savefig(path, dpi=300)
        plt.close(fig)
        print(f"✓ Plot saved: {path}")

    # ------------------------------------------------------------------
    # Plot 3 — Spatial influence map
    # ------------------------------------------------------------------
    def plot_spatial_influence(self) -> None:
        """Scatter plot of all cells coloured by consensus mean influence."""
        coords = self.all_spatial_coords
        scores = self.cell_mean_norm.copy()
        scores[~self.covered_mask] = np.nan   # uncovered cells greyed out

        fig, ax = plt.subplots(figsize=(8, 7))
        sc = ax.scatter(
            coords[:, 0], coords[:, 1],
            c=scores, cmap="magma", s=6, alpha=0.85,
            vmin=0.0, vmax=1.0, rasterized=True,
        )
        cbar = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("Consensus Influence (norm.)", fontsize=17)#10
        ax.set_xlabel("X (µm)", fontsize=18)#11
        ax.set_ylabel("Y (µm)", fontsize=18)#11
        ax.set_title("Spatial Consensus Influence Map\n"
                     "(mean across all folds)", fontsize=18)#13
        ax.set_aspect("equal")
        fig.tight_layout()
        path = os.path.join(self.output_dir, "C3_spatial_influence_map.png")
        fig.savefig(path, dpi=300)
        plt.close(fig)
        print(f"✓ Plot saved: {path}")

    # ------------------------------------------------------------------
    # Plot 4 — Communication corridors with consensus + sel_freq
    # ------------------------------------------------------------------
    def plot_communication_corridors(self, top_k: int = 10) -> None:
        """Corridor lines coloured and sized by consensus influence.

        Corridors are high-influence edges in the adjacency matrix.
        Line width ∝ consensus mean influence of the endpoint cells.
        Annotated with selection frequency when > 0.5 (appears in majority of folds).

        If ``self.consensus_corridor_df`` was provided (built by
        :meth:`SVRNPipeline.run_kfold` from each fold's
        ``4b_corridor_summary.csv``), a stats inset on the right shows, for
        each named communication corridor (C1, C2, ...), the mean ± std
        across folds of mean influence, hop count, and spatial length, plus
        the consensus dominant cell type. The aggregated table is also saved
        as ``C4_consensus_corridor_summary.csv``.
        """
        coords = self.all_spatial_coords
        scores = self.cell_mean_norm

        corridor_df = self.consensus_corridor_df

        # Build edge list from adjacency (sparse or dense)
        try:
            import scipy.sparse as sp_sparse
            if sp_sparse.issparse(self.adj_matrix):
                coo = self.adj_matrix.tocoo()
                src_arr, dst_arr = coo.row, coo.col
            else:
                src_arr, dst_arr = np.where(self.adj_matrix > 0)
        except Exception:
            src_arr, dst_arr = np.where(np.array(self.adj_matrix) > 0)

        # Edge weight = mean of endpoint consensus scores
        valid = (src_arr < len(scores)) & (dst_arr < len(scores))
        src_arr, dst_arr = src_arr[valid], dst_arr[valid]
        edge_weights = (scores[src_arr] + scores[dst_arr]) / 2.0

        # Select top-K edges
        top_idx = np.argsort(edge_weights)[-top_k:][::-1]
        top_src = src_arr[top_idx]
        top_dst = dst_arr[top_idx]
        top_w   = edge_weights[top_idx]

        # Cell-type lookup for endpoint labels
        def ct_label(i):
            return self.all_cell_types[i] if i < len(self.all_cell_types) else "?"

        # Build a per-cell-type selection-frequency lookup
        sf_lookup = dict(zip(
            self.consensus_df["cell_type"],
            self.consensus_df["selection_frequency"],
        ))

        # ── Figure layout: spatial map + optional corridor-stats inset ────
        have_stats = corridor_df is not None and not corridor_df.empty
        if have_stats:
            fig = plt.figure(figsize=(14, 8))
            ax       = fig.add_axes([0.04, 0.08, 0.54, 0.84])   # left: spatial map
            ax_stats = fig.add_axes([0.62, 0.08, 0.36, 0.84])   # right: corridor stats
        else:
            fig, ax = plt.subplots(figsize=(9, 8))
            ax_stats = None

        # Background: all cells (grey)
        ax.scatter(coords[:, 0], coords[:, 1], c="lightgrey", s=4,
                   alpha=0.4, rasterized=True, zorder=1)
        # Highlight corridor endpoints
        ep_idx = np.unique(np.concatenate([top_src, top_dst]))
        ax.scatter(coords[ep_idx, 0], coords[ep_idx, 1],
                   c=scores[ep_idx], cmap="magma", s=30,
                   edgecolors="black", linewidths=0.4, zorder=3, vmin=0, vmax=1)

        w_min, w_max = top_w.min(), top_w.max() + 1e-8
        cmap_line = plt.get_cmap("YlOrRd")

        for rank, (s, d, w) in enumerate(zip(top_src, top_dst, top_w)):
            lw     = 1.0 + 5.0 * (w - w_min) / (w_max - w_min)
            color  = cmap_line(0.3 + 0.7 * (w - w_min) / (w_max - w_min))
            ax.plot(
                [coords[s, 0], coords[d, 0]],
                [coords[s, 1], coords[d, 1]],
                color=color, linewidth=lw, alpha=0.85, zorder=2,
            )
            # Annotate with selection frequency if > 0.5
            src_ct = ct_label(s)
            sf_s   = sf_lookup.get(src_ct, 0.0)
            if sf_s > 0.5:
                mid_x = (coords[s, 0] + coords[d, 0]) / 2
                mid_y = (coords[s, 1] + coords[d, 1]) / 2
                ax.text(mid_x, mid_y, f"sf={sf_s:.2f}",
                        fontsize=5.5, ha="center", va="center",
                        color="navy", alpha=0.85)

        sm = plt.cm.ScalarMappable(cmap=cmap_line,
                                   norm=plt.Normalize(vmin=w_min, vmax=w_max))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("Corridor Consensus Influence", fontsize=16)

        ax.set_xlabel("X (µm)", fontsize=18)#11
        ax.set_ylabel("Y (µm)", fontsize=18)#11
        ax.set_title(f"Communication Corridors (top {top_k})\n"
                     "Line width & colour ∝ consensus influence; "
                     "sf annotation when > 0.5", fontsize=18)#11
        ax.set_aspect("equal")

        # ── Corridor-stats inset: mean ± std across folds ──────────────────
        if ax_stats is not None:
            cids  = corridor_df["corridor_id"].tolist()
            x     = np.arange(len(cids))
            bar_w = 0.25
            tab_colors = plt.get_cmap("tab10", len(cids))(np.arange(len(cids)))

            # Normalise each metric (and its std) to [0, 1] using the metric's
            # own range, so all three groups of bars share one y-axis.
            def _norm_with_err(val_col: str, std_col: str):
                vals = corridor_df[val_col].values.astype(float)
                errs = corridor_df[std_col].values.astype(float)
                rng  = vals.max() - vals.min() + 1e-8
                return (vals - vals.min()) / rng, errs / rng

            m_inf_n, m_inf_e = _norm_with_err("mean_influence", "mean_influence_std")
            n_h_n,   n_h_e   = _norm_with_err("n_hops", "n_hops_std")
            sp_l_n,  sp_l_e  = _norm_with_err("spatial_length", "spatial_length_std")

            ax_stats.bar(x - bar_w, m_inf_n, bar_w, yerr=m_inf_e, capsize=2,
                         label="Mean Inf (norm)", color=tab_colors, alpha=0.85)
            ax_stats.bar(x,          n_h_n,  bar_w, yerr=n_h_e, capsize=2,
                         label="Hops (norm)",     color=tab_colors, alpha=0.55,
                         hatch="//")
            ax_stats.bar(x + bar_w, sp_l_n,  bar_w, yerr=sp_l_e, capsize=2,
                         label="Spatial L (norm)", color=tab_colors, alpha=0.35,
                         hatch="xx")

            # Consensus dominant cell-type + fold-count label above each group
            top_env = np.maximum(np.maximum(m_inf_n + m_inf_e, n_h_n + n_h_e),
                                  sp_l_n + sp_l_e)
            # Cap the label anchor: when a metric's std is large relative to
            # its own range, errs/rng can exceed 1, pushing top_env well past
            # the axes' ylim. Combined with clip_on defaulting to False for
            # ax.text, that rendered labels entirely outside the plot.
            label_y = np.clip(top_env, 0.0, 1.15) + 0.04
            for i, row in corridor_df.iterrows():
                dom = str(row.get("dominant_type", ""))[:10]
                nfolds = int(row.get("n_folds", 0))
                ax_stats.text(
                    i, label_y[i],
                    f"{dom}\n(n={nfolds})",
                    ha="center", va="bottom", fontsize=14, rotation=35,
                    clip_on=True,
                )

            ax_stats.set_xticks(x)
            ax_stats.set_xticklabels(cids, fontsize=14, rotation=30)
            ax_stats.set_ylim(0, 1.40)
            ax_stats.set_ylabel("Normalised value (mean ± std across folds)", fontsize=15)#8.5
            ax_stats.set_title(
                "Consensus Corridor Statistics\n(aggregated across all folds)",
                fontsize=15)#9
            ax_stats.legend(fontsize=14, loc="upper right", framealpha=0.8)
            ax_stats.yaxis.set_tick_params(labelsize=9)

            c4_csv = os.path.join(self.output_dir, "C4_consensus_corridor_summary.csv")
            corridor_df.to_csv(c4_csv, index=False)
            print(f"✓ Consensus corridor statistics CSV (C4) saved: {c4_csv}")

        fig.tight_layout()
        path = os.path.join(self.output_dir, "C4_communication_corridors.png")
        fig.savefig(path, dpi=300)
        plt.close(fig)
        print(f"✓ Plot saved: {path}")

    # ------------------------------------------------------------------
    # Plot 5 — Uncertainty spatial (fold std per cell)
    # ------------------------------------------------------------------
    def plot_uncertainty_spatial(self) -> None:
        """Spatial map of per-cell uncertainty (std across folds)."""
        coords = self.all_spatial_coords
        std_vals = self.cell_std.copy()
        std_vals[~self.covered_mask] = np.nan

        fig, ax = plt.subplots(figsize=(8, 7))
        sc = ax.scatter(
            coords[:, 0], coords[:, 1],
            c=std_vals, cmap="cividis", s=6, alpha=0.85,
            vmin=0.0, rasterized=True,
        )
        cbar = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("Fold Std (influence uncertainty)", fontsize=17)#10
        ax.set_xlabel("X (µm)", fontsize=17)#11
        ax.set_ylabel("Y (µm)", fontsize=17)#11
        ax.set_title("Spatial Uncertainty Map\n"
                     "(standard deviation of influence across folds)", fontsize=18)#13
        ax.set_aspect("equal")
        fig.tight_layout()
        path = os.path.join(self.output_dir, "C5_uncertainty_spatial.png")
        fig.savefig(path, dpi=300)
        plt.close(fig)
        print(f"✓ Plot saved: {path}")

    # ------------------------------------------------------------------
    # Plot 6 — Uncertainty by cell type (boxplot of per-cell fold std)
    # ------------------------------------------------------------------
    def plot_uncertainty_by_cell_type(self) -> None:
        """Boxplot: distribution of per-cell fold std within each cell type."""
        ct_order = self.consensus_df["cell_type"].tolist()
        data_by_ct = []
        for ct in ct_order:
            mask = (self.all_cell_types == ct) & self.covered_mask
            data_by_ct.append(self.cell_std[mask] if mask.any() else np.array([0.0]))

        fig, ax = plt.subplots(figsize=(max(8, len(ct_order) * 0.8), 6))
        bp = ax.boxplot(
            data_by_ct, labels=ct_order, patch_artist=True,
            medianprops={"color": "red", "linewidth": 1.5},
            boxprops={"facecolor": "lightsteelblue", "alpha": 0.7},
            flierprops={"marker": ".", "markersize": 3, "alpha": 0.4},
        )
        ax.set_xlabel("Cell Type", fontsize=17)#12
        ax.set_ylabel("Fold Std (influence uncertainty)", fontsize=17)#12
        ax.set_title("Prediction Uncertainty by Cell Type\n"
                     "(fold std = epistemic uncertainty from spatial partitioning)", fontsize=16)#11
        ax.tick_params(axis="x", rotation=40)
        fig.tight_layout()
        path = os.path.join(self.output_dir, "C6_uncertainty_by_celltype.png")
        fig.savefig(path, dpi=300)
        plt.close(fig)
        print(f"✓ Plot saved: {path}")

    # ------------------------------------------------------------------
    # Plot 7 — Confidence intervals for top cells (mean ± std error bars)
    # ------------------------------------------------------------------
    def plot_confidence_intervals_top_cells(self, top_n: int = 30) -> None:
        """Horizontal error-bar chart for the top-N highest-influence cells.

        Shows mean ± std across folds for each cell.
        """
        covered_idx = np.where(self.covered_mask)[0]
        top_order   = np.argsort(self.cell_mean[covered_idx])[-top_n:][::-1]
        top_global  = covered_idx[top_order]

        means  = self.cell_mean[top_global]
        stds   = self.cell_std[top_global]
        labels = [
            f"{self.all_cell_types[i]} [{i}]" for i in top_global
        ]

        fig, ax = plt.subplots(figsize=(9, max(6, top_n * 0.32)))
        y_pos = np.arange(len(top_global))
        ax.barh(y_pos, means, xerr=stds, align="center",
                color="steelblue", alpha=0.75, capsize=3,
                error_kw={"elinewidth": 1.0, "ecolor": "black"})
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=15)#7
        ax.invert_yaxis()
        ax.set_xlabel("Mean Influence Score (± fold std)", fontsize=17)#12
        ax.set_title(f"Top-{top_n} Cells: Consensus Influence with Uncertainty\n"
                     "(error bars = std across folds)", fontsize=17)#11
        fig.tight_layout()
        path = os.path.join(self.output_dir, "C7_confidence_intervals_top_cells.png")
        fig.savefig(path, dpi=300)
        plt.close(fig)
        print(f"✓ Plot saved: {path}")

    # ------------------------------------------------------------------
    # Plot 8 — Hill KN affinity analysis
    # ------------------------------------------------------------------
    def plot_hill_kn_affinity(self) -> None:
        """Mean ± std of log_n and log_K per LR pair across folds.

        Cooperativity classification: n > 1.2 = positive cooperativity.
        Does NOT use consensus influence — uses per-fold model parameters directly.
        """
        if not self.fold_log_n:
            print("! No fold_log_n data; skipping Hill KN plot.")
            return

        log_n_stack = np.stack(list(self.fold_log_n.values()), axis=0)  # (n_folds, n_lr)
        log_K_stack = np.stack(list(self.fold_log_K.values()), axis=0)  # (n_folds, n_lr)

        n_vals = np.exp(log_n_stack)   # actual Hill coefficient n per fold per LR
        K_vals = np.exp(log_K_stack)   # actual affinity K per fold per LR

        mean_n = n_vals.mean(axis=0)   # (n_lr,)
        std_n  = n_vals.std(axis=0)
        mean_K = K_vals.mean(axis=0)
        std_K  = K_vals.std(axis=0)

        n_lr = len(mean_n)
        labels = self.lr_pair_names if len(self.lr_pair_names) == n_lr \
            else [f"LR{i}" for i in range(n_lr)]

        # Cooperativity: n > 1.2 = positive, n < 0.8 = negative, else neutral
        coop_colors = []
        for n in mean_n:
            if n > 1.2:
                coop_colors.append("tomato")       # positive cooperativity
            elif n < 0.8:
                coop_colors.append("cornflowerblue")  # negative
            else:
                coop_colors.append("lightgrey")    # neutral

        scatter_colors = [
            "#4c8cbf" if c == "lightgrey" else c for c in coop_colors
        ]

        # ── Figure 1: Bar charts (n and K per LR pair) — unchanged ──────
        fig, axes = plt.subplots(2, 1, figsize=(max(10, n_lr * 0.5), 9))

        # Top: Hill coefficient n
        axes[0].bar(labels, mean_n, yerr=std_n, color=coop_colors,
                    edgecolor="grey", linewidth=0.5, capsize=3,
                    error_kw={"elinewidth": 1.0, "ecolor": "black"})
        axes[0].axhline(1.2, color="tomato",         linestyle="--", lw=1.2,
                        label="n=1.2 (positive coop. threshold)")
        axes[0].axhline(0.8, color="cornflowerblue", linestyle="--", lw=1.2,
                        label="n=0.8 (negative coop. threshold)")
        axes[0].axhline(1.0, color="black",          linestyle=":",  lw=0.9)
        axes[0].set_ylabel("Hill coefficient n (mean ± std)", fontsize=17)
        axes[0].set_title("Hill KN Affinity — Cooperativity (n)\n"
                          "Red = positive (n>1.2), Blue = negative (n<0.8), "
                          "Grey = neutral", fontsize=17)#11
        axes[0].legend(fontsize=14, loc="upper right")
        axes[0].tick_params(axis="x", rotation=50)

        # Bottom: affinity K
        axes[1].bar(labels, mean_K, yerr=std_K, color="mediumpurple",
                    edgecolor="grey", linewidth=0.5, capsize=3,
                    error_kw={"elinewidth": 1.0, "ecolor": "black"})
        axes[1].set_ylabel("Affinity K (mean ± std)", fontsize=16)#11
        axes[1].set_title("Hill KN Affinity — Half-saturation Constant (K)", fontsize=16)#11
        axes[1].tick_params(axis="x", rotation=50)

        fig.suptitle("Hill KN Affinity Analysis (per-fold model parameters)",
                     fontsize=18, fontweight="bold")#13
        fig.tight_layout()
        path = os.path.join(self.output_dir, "C8_hill_kn_affinity.png")
        fig.savefig(path, dpi=300)
        plt.close(fig)

     
        fig2, ax_sc = plt.subplots(figsize=(8, 6))

        median_K = float(np.median(mean_K))
        median_n = float(np.median(mean_n))

        ax_sc.errorbar(
            mean_K, mean_n,
            xerr=std_K, yerr=std_n,
            fmt="none",
            alpha=0.18,
            elinewidth=0.6,
            capsize=2,
            ecolor="grey",
            zorder=2,
        )
        ax_sc.scatter(
            mean_K, mean_n,
            c=scatter_colors,
            s=26,
            alpha=0.9,
            edgecolors="white",
            linewidths=0.4,
            zorder=3,
        )

        # Quadrant guide lines (dashed grey, exactly like reference image)
        ax_sc.axvline(median_K, color="grey", linestyle="--", lw=1.0, alpha=0.75,
                      label=f"median K = {median_K:.4f}")
        ax_sc.axhline(median_n, color="grey", linestyle="--", lw=1.0, alpha=0.75,
                      label=f"median n = {median_n:.4f}")

        # Cooperativity threshold reference lines (lighter, dotted)
        ax_sc.axhline(1.2, color="tomato",         lw=0.8, ls=":", alpha=0.6,
                      label="n=1.2 (pos. coop.)")
        ax_sc.axhline(0.8, color="cornflowerblue", lw=0.8, ls=":", alpha=0.6,
                      label="n=0.8 (neg. coop.)")

        # Quadrant count annotations (number of LR pairs in each quadrant)
        q_labels = {
            "LL": (mean_K <= median_K) & (mean_n <= median_n),
            "LH": (mean_K <= median_K) & (mean_n >  median_n),
            "RL": (mean_K >  median_K) & (mean_n <= median_n),
            "RH": (mean_K >  median_K) & (mean_n >  median_n),
        }
        x_range = mean_K.max() - mean_K.min() + 1e-8
        y_range = mean_n.max() - mean_n.min() + 1e-8
        pad_x   = x_range * 0.04
        pad_y   = y_range * 0.04
        x_min, x_max = mean_K.min() - pad_x, mean_K.max() + pad_x
        y_min, y_max = mean_n.min() - pad_y, mean_n.max() + pad_y

        quadrant_positions = {
            "LL": (x_min + pad_x, y_min + pad_y, "left",  "bottom"),
            "LH": (x_min + pad_x, y_max - pad_y, "left",  "top"),
            "RL": (x_max - pad_x, y_min + pad_y, "right", "bottom"),
            "RH": (x_max - pad_x, y_max - pad_y, "right", "top"),
        }
        quadrant_desc = {
            "LL": "Low K\nLow n",
            "LH": "Low K\nHigh n",
            "RL": "High K\nLow n",
            "RH": "High K\nHigh n",
        }
        for qk, qmask in q_labels.items():
            xp, yp, ha_, va_ = quadrant_positions[qk]
            ax_sc.text(xp, yp,
                       f"{quadrant_desc[qk]}\nn={int(qmask.sum())}",
                       ha=ha_, va=va_, fontsize=14,
                       color="#555555",
                       bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                 ec="#cccccc", alpha=0.7))

        ax_sc.set_xlim(x_min, x_max)
        ax_sc.set_ylim(y_min, y_max)
        ax_sc.set_xlabel("K (affinity constant)", fontsize=17)#12
        ax_sc.set_ylabel("n (Hill coefficient)", fontsize=17)#12
        ax_sc.set_title("LR pair binding regimes", fontsize=18, fontweight="bold")

        # Legend: cooperativity colour key + quadrant lines
        from matplotlib.lines import Line2D
        legend_elems = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="tomato",
                   markersize=8, label="Positive cooperativity (n > 1.2)"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="cornflowerblue",
                   markersize=8, label="Negative cooperativity (n < 0.8)"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#4c8cbf",
                   markersize=8, label="Neutral (0.8 ≤ n ≤ 1.2)"),
            Line2D([0], [0], color="grey", lw=1, ls="--",
                   label=f"Median K = {median_K:.4f}"),
            Line2D([0], [0], color="grey", lw=1, ls="--",
                   label=f"Median n = {median_n:.4f}"),
        ]
   
        ax_sc.legend(handles=legend_elems, fontsize=14, loc="upper left",
                     bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0,
                     framealpha=0.9)#8
        ax_sc.spines[["top", "right"]].set_visible(False)
        fig2.tight_layout()
        path_sc = os.path.join(self.output_dir, "C8_hill_kn_scatter.png")
        fig2.savefig(path_sc, dpi=300, bbox_inches="tight")
        plt.close(fig2)
        print(f"✓ Hill KN scatter plot saved: {path_sc}")

        # ── Save numeric summary CSV ───────────────────────────────────────
        hill_df = pd.DataFrame({
            "lr_pair":   labels,
            "mean_n":    np.round(mean_n, 4),
            "std_n":     np.round(std_n,  4),
            "mean_K":    np.round(mean_K, 4),
            "std_K":     np.round(std_K,  4),
            "cooperativity": [
                "positive" if n > 1.2 else ("negative" if n < 0.8 else "neutral")
                for n in mean_n
            ],
            "quadrant": [
                f"{'High' if k > median_K else 'Low'}K_{'High' if n > median_n else 'Low'}n"
                for k, n in zip(mean_K, mean_n)
            ],
        })
        csv_path = os.path.join(self.output_dir, "C8_hill_kn_summary.csv")
        hill_df.to_csv(csv_path, index=False)
        print(f"✓ Plot saved: {path}")
        print(f"✓ Hill KN summary saved: {csv_path}")

  
    @staticmethod
    def _assign_pathway_category(pathway_name: str) -> str:
        """Return a broad biological category for a KEGG/Reactome pathway name.

        Categories are checked in priority order so that a pathway that matches
        multiple patterns (e.g. 'GPCR signalling in axon guidance') gets the
        most specific label.
        """
        name_l = pathway_name.lower()

        # --- Signalling axes ---
        if any(t in name_l for t in ("gpcr", "g protein", "g-protein",
                                      "adenylate cyclase", "camp", "camp-mediated",
                                      "adrenergic", "chemokine receptor",
                                      "olfactory", "rhodopsin")):
            return "GPCR / G-protein"

        if any(t in name_l for t in ("ecm", "extracellular matrix",
                                      "integrin", "collagen", "fibronectin",
                                      "laminin", "elastic fibre", "elastic fiber",
                                      "fibril", "proteoglycan", "heparan")):
            return "ECM / Integrin"

        if any(t in name_l for t in ("wnt", "notch", "hedgehog", "frizzled",
                                      "beta-catenin", "β-catenin")):
            return "Wnt / Notch / Hedgehog"

        if any(t in name_l for t in ("tgf", "bmp", "smad", "activin")):
            return "TGF-β / BMP"

        if any(t in name_l for t in ("vegf", "fgf", "egf", "pdgf", "igf",
                                      "nrg", "neuregulin", "erbb",
                                      "receptor tyrosine kinase", "rtk")):
            return "Growth Factor / RTK"

        if any(t in name_l for t in ("pi3k", "akt", "mtor", "pten",
                                      "pi3k-akt", "pi3k/akt")):
            return "PI3K / AKT / mTOR"

        if any(t in name_l for t in ("mapk", "ras", "erk", "mek", "raf",
                                      "rap1", "ras signaling", "mapk signaling")):
            return "MAPK / RAS"

        if any(t in name_l for t in ("jak", "stat", "interferon", "cytokine",
                                      "il-", "interleukin", "tnf", "nf-kb",
                                      "nfkb", "innate immune", "toll-like")):
            return "Cytokine / JAK-STAT / Immune"

        if any(t in name_l for t in ("complement", "mhc", "t cell", "b cell",
                                      "adaptive immune", "lymphocyte",
                                      "natural killer", "antigen")):
            return "Adaptive / Innate Immunity"

        # --- Neuronal / axon ---
        if any(t in name_l for t in ("axon", "ephrin", "eph-ephrin",
                                      "semaphorin", "netrin", "slit",
                                      "neurotrophin", "ntrk", "ngf",
                                      "neuron", "synap", "axon guidance",
                                      "nervous system", "neural",
                                      "neuropeptide", "glutamate", "gaba",
                                      "dopamine", "serotonin", "acetylcholine")):
            return "Neuronal / Axon Guidance"

        if any(t in name_l for t in ("cell adhesion", "cadherin", "nectin",
                                      "tight junction", "gap junction",
                                      "cell-cell", "nectins")):
            return "Cell Adhesion / Junction"

        # --- Proliferation / apoptosis ---
        if any(t in name_l for t in ("cell cycle", "apoptosis", "p53",
                                      "senescence", "dna damage", "dna repair",
                                      "autophagy", "ubiquitin")):
            return "Cell Cycle / Apoptosis"

        if any(t in name_l for t in ("angiogenesis", "vegf", "endothel",
                                      "blood vessel")):
            return "Angiogenesis / Endothelial"

        if any(t in name_l for t in ("hippo", "yap", "taz")):
            return "Hippo / YAP"

        # --- Metabolism ---
        if any(t in name_l for t in ("metabol", "glycolysis", "oxidative phosph",
                                      "fatty acid", "amino acid", "purine",
                                      "pyrimidine", "lipid", "cholesterol",
                                      "mitochondri")):
            return "Metabolism"

        # --- Development ---
        if any(t in name_l for t in ("development", "developmental",
                                      "morphogenesis", "differentiation",
                                      "embryo", "organogenesis")):
            return "Development / Morphogenesis"

        # --- Infection / disease ---
        if any(t in name_l for t in ("cancer", "carcinoma", "tumor",
                                      "oncogene", "micro", "virus",
                                      "bacterial", "leishmani", "pathog")):
            return "Disease / Infection"

        return "Other / Miscellaneous"

    # Plot 9 — Top LR pairs + KEGG/Reactome pathway enrichment (2-panel)
    # ------------------------------------------------------------------
    def plot_top_lr_pairs(
        self,
        top_n: int = 50,
        species: str = "mouse",
        top_pathways: int = 15,
    ) -> None:
        
        if not self.fold_edge_probs:
            print("! No fold_edge_probs data; skipping LR pair plot.")
            return

        # ── 1. Compute mean ± std edge probability per LR pair ────────────
        fold_mean_probs = []
        for ep in self.fold_edge_probs.values():
            fold_mean_probs.append(ep.mean(axis=0))   # (n_lr,)
        all_means = np.stack(fold_mean_probs, axis=0)  # (n_folds, n_lr)
        lr_mean   = all_means.mean(axis=0)
        lr_std    = all_means.std(axis=0)
        n_lr      = len(lr_mean)

        labels = (self.lr_pair_names
                  if len(self.lr_pair_names) == n_lr
                  else [f"LR{i}" for i in range(n_lr)])

        order     = np.argsort(lr_mean)[::-1]
        top_idx   = order[:top_n]

        # ── 2. Hill K / n means per LR pair (from fold_log_K / fold_log_n) ─
        has_hill = bool(self.fold_log_n and self.fold_log_K)
        if has_hill:
            log_n_stack = np.stack(list(self.fold_log_n.values()), axis=0)
            log_K_stack = np.stack(list(self.fold_log_K.values()), axis=0)
            mean_n = np.exp(log_n_stack).mean(axis=0)   # (n_lr,)
            mean_K = np.exp(log_K_stack).mean(axis=0)   # (n_lr,)
        else:
            mean_n = np.ones(n_lr)
            mean_K = np.ones(n_lr)

        # ── 3. Save ranked LR CSV ──────────────────────────────────────────
        lr_df = pd.DataFrame({
            "lr_pair":        [labels[i] for i in order],
            "mean_edge_prob":  np.round(lr_mean[order], 6),
            "std_edge_prob":   np.round(lr_std[order],  6),
            "mean_hill_n":     np.round(mean_n[order],  4),
            "mean_hill_K":     np.round(mean_K[order],  4),
            "rank":            np.arange(1, n_lr + 1),
        })
        csv_path = os.path.join(self.output_dir, "C9_top_lr_pairs_ranked.csv")
        lr_df.to_csv(csv_path, index=False)

        # ── 4. mygene annotation ───────────────────────────────────────────
        # Extract all unique gene IDs from lr_pair_names.
        # Supports formats: "GeneA__GeneB", "GeneA_GeneB", "GeneA-GeneB"
        import re

        def _split_pair(name: str) -> List[str]:
            for sep in ("__", ":", "_", "-"):
                parts = name.split(sep, 1)
                if len(parts) == 2 and parts[1]:
                    return parts
            return [name, name]

        all_lr_genes_raw: List[str] = []
        pair_to_genes: Dict[str, Tuple[str, str]] = {}
        for lbl in labels:
            lig, rec = _split_pair(lbl)
            pair_to_genes[lbl] = (lig, rec)
            all_lr_genes_raw.extend([lig, rec])
        all_lr_genes = list(dict.fromkeys(all_lr_genes_raw))   # dedup, order preserved

        gene_to_pathways: Dict[str, List[str]] = {}   # gene → list of pathway names
        gene_to_symbol:   Dict[str, str]       = {}   # ensembl/id → symbol

        try:
            import mygene
            mg = mygene.MyGeneInfo()

            gene_list = list(all_lr_genes)   # guarantee it's a list

          
            hits = mg.querymany(
                gene_list,
                scopes="ensembl.gene,symbol,ensembl.transcript",
                fields="symbol,name,entrezgene,pathway.kegg,pathway.reactome",
                species=species,
                as_dataframe=False,   # get list of dicts so 'query' key is preserved
                verbose=False,
            )

            for hit in hits:
                if hit.get("notfound"):
                    continue
                # 'query' is the original string we passed (e.g. 'ENSMUSG00000032796')
                orig_query = str(hit.get("query", ""))
                if not orig_query:
                    continue

                sym = hit.get("symbol", orig_query)
                if not isinstance(sym, str) or not sym:
                    sym = orig_query
                gene_to_symbol[orig_query] = sym

                pw_list: List[str] = []
                # KEGG — can be dict or list of dicts
                kegg = hit.get("pathway", {}).get("kegg")
                if isinstance(kegg, list):
                    pw_list += [f"KEGG: {p['name']}" for p in kegg
                                if isinstance(p, dict) and p.get("name")]
                elif isinstance(kegg, dict) and kegg.get("name"):
                    pw_list.append(f"KEGG: {kegg['name']}")

                # Reactome — same structure
                react = hit.get("pathway", {}).get("reactome")
                if isinstance(react, list):
                    pw_list += [f"Reactome: {p['name']}" for p in react
                                if isinstance(p, dict) and p.get("name")]
                elif isinstance(react, dict) and react.get("name"):
                    pw_list.append(f"Reactome: {react['name']}")

                # Store under original query — only update if we got more pathways
                if pw_list or orig_query not in gene_to_pathways:
                    gene_to_pathways[orig_query] = pw_list

            n_with_pw = sum(1 for v in gene_to_pathways.values() if v)
            print(f"  mygene: annotated {len(gene_to_pathways)}/{len(all_lr_genes)} genes"
                  f" ({n_with_pw} with pathway info)")

        except Exception as exc:
            print(f"  mygene annotation failed ({exc}); will show Hill scatter only.")
            gene_to_pathways = {}
            gene_to_symbol   = {}

        from collections import defaultdict
        pair_pathways: Dict[str, List[str]] = {}
        for lbl in labels:
            lig, rec = pair_to_genes[lbl]
            lig_pw = set(gene_to_pathways.get(lig, []))
            rec_pw = set(gene_to_pathways.get(rec, []))
            # Union: pathway relevant if either gene is a member
            pair_pathways[lbl] = list(lig_pw | rec_pw)

        # Collect pathways that cover at least 1 LR pair (threshold=1 for mouse
        # panels where LR databases are sparse; raise to 2 for large human panels)
        pathway_members: Dict[str, List[int]] = defaultdict(list)
        for i, lbl in enumerate(labels):
            for pw in pair_pathways[lbl]:
                pathway_members[pw].append(i)
        pathway_members = {pw: idxs for pw, idxs in pathway_members.items()
                           if len(idxs) >= 1}

        has_pathways = bool(pathway_members)

        # ── 6. Pathway-level statistics ────────────────────────────────────
        from scipy.stats import fisher_exact

        pathway_stats: List[Dict] = []
        n_total_lr = n_lr
        for pw, member_idx in pathway_members.items():
            mem_arr      = np.array(member_idx)
            pw_mean_prob = lr_mean[mem_arr].mean()
            pw_mean_n    = mean_n[mem_arr].mean()
            pw_mean_K    = mean_K[mem_arr].mean()
            n_mem        = len(mem_arr)

            # Fisher's exact: pathway members vs non-members in top-N
            in_top    = set(top_idx.tolist())
            in_pw     = set(mem_arr.tolist())
            a = len(in_pw & in_top)           # in pathway AND top-N
            b = len(in_top) - a               # top-N but not pathway
            c = len(in_pw) - a               # pathway but not top-N
            d = n_total_lr - a - b - c        # neither
            _, p_val = fisher_exact([[a, b], [c, d]], alternative="greater")
            active_frac = a / max(n_mem, 1)

            # Collect gene symbol labels for member LR pairs
            member_gene_labels = []
            for idx_m in mem_arr:
                lbl_m = labels[idx_m]
                lig_m, rec_m = pair_to_genes[lbl_m]
                ls = gene_to_symbol.get(lig_m, lig_m[-6:] if len(lig_m) > 6 else lig_m)
                rs = gene_to_symbol.get(rec_m, rec_m[-6:] if len(rec_m) > 6 else rec_m)
                member_gene_labels.append(f"{ls}__{rs}")

            pathway_stats.append({
                "pathway":             pw,
                "category":            self._assign_pathway_category(pw),
                "n_lr_members":        n_mem,
                "mean_edge_prob":      round(float(pw_mean_prob), 6),
                "mean_hill_n":         round(float(pw_mean_n),    4),
                "mean_hill_K":         round(float(pw_mean_K),    4),
                "active_frac":         round(float(active_frac),  4),
                "p_value":             float(p_val),
                "member_indices":      mem_arr,
                "member_gene_labels":  member_gene_labels,
            })

        if pathway_stats:
            # Benjamini-Hochberg FDR correction
            p_vals = np.array([s["p_value"] for s in pathway_stats])
            n_tests = len(p_vals)
            sorted_idx = np.argsort(p_vals)
            adj_p = p_vals.copy()
            for rank_i, orig_i in enumerate(sorted_idx):
                adj_p[orig_i] = min(1.0, p_vals[orig_i] * n_tests / (rank_i + 1))
            # Enforce monotonicity (standard B-H step-up)
            for i in range(n_tests - 2, -1, -1):
                adj_p[sorted_idx[i]] = min(adj_p[sorted_idx[i]],
                                           adj_p[sorted_idx[i + 1]])
            for i, s in enumerate(pathway_stats):
                s["adj_p_value"] = float(adj_p[i])
                s["neg_log10_adj_p"] = float(-np.log10(adj_p[i] + 1e-300))

            # Sort by adj p-value
            pathway_stats.sort(key=lambda x: x["adj_p_value"])

            # Save enrichment CSV (exclude non-serialisable list columns)
            _exclude = {"member_indices", "member_gene_labels"}
            pw_df = pd.DataFrame([{k: v for k, v in s.items()
                                    if k not in _exclude}
                                   for s in pathway_stats])
            pw_csv = os.path.join(self.output_dir, "C9_pathway_enrichment.csv")
            pw_df.to_csv(pw_csv, index=False)
            print(f"✓ Pathway enrichment saved: {pw_csv}")

        # ── 7. Build the figure ────────────────────────────────────────────
        if not has_pathways:
            # ── Fallback: bar chart + Hill n vs K scatter (2 panels) ──────
            fig, (ax_bar, ax_hill) = plt.subplots(
                1, 2, figsize=(18, max(6, min(top_n * 0.28, 14))),
                gridspec_kw={"width_ratios": [1.4, 1]},
            )

            # Panel A: horizontal bar of top-N LR pairs
            top_labels = [labels[i] for i in top_idx]
            # Use gene symbols where available, fall back to short Ensembl suffix
            def _short(lbl):
                lig, rec = _split_pair(lbl)
                ls = gene_to_symbol.get(lig, lig[-8:] if len(lig) > 8 else lig)
                rs = gene_to_symbol.get(rec, rec[-8:] if len(rec) > 8 else rec)
                return f"{ls}__{rs}"

            short_labels = [_short(lbl) for lbl in top_labels]
            ax_bar.barh(short_labels[::-1], lr_mean[top_idx][::-1],
                        xerr=lr_std[top_idx][::-1],
                        color="mediumseagreen", edgecolor="grey",
                        linewidth=0.4, capsize=2,
                        error_kw={"elinewidth": 0.8, "ecolor": "black"})
            ax_bar.set_xlabel("Mean SVRN LR probability", fontsize=20)
            ax_bar.set_title(f"Top-{top_n} LR pairs by mean edge probability",
                             fontsize=20, fontweight="bold")
            ax_bar.tick_params(axis="y", labelsize=14)
            ax_bar.spines[["top", "right"]].set_visible(False)

        # ── Three-panel figure (matching reference format) ────────────────
        top_pw   = pathway_stats[:top_pathways]   # already sorted by adj-p
        pw_names = [s["pathway"] for s in top_pw]
        # Truncate long pathway names for y-axis labels
        pw_short = [n[:52] + "…" if len(n) > 53 else n for n in pw_names]

        # ── Biological category colour palette ────────────────────────────
        _CAT_PALETTE = {
            "GPCR / G-protein":            "#e63946",
            "ECM / Integrin":              "#457b9d",
            "Neuronal / Axon Guidance":    "#2a9d8f",
            "MAPK / RAS":                  "#e9c46a",
            "PI3K / AKT / mTOR":           "#f4a261",
            "TGF-β / BMP":                 "#264653",
            "Wnt / Notch / Hedgehog":      "#8338ec",
            "Growth Factor / RTK":         "#fb5607",
            "Cytokine / JAK-STAT / Immune":"#3a86ff",
            "Adaptive / Innate Immunity":  "#06d6a0",
            "Cell Adhesion / Junction":    "#ffbe0b",
            "Cell Cycle / Apoptosis":      "#ff006e",
            "Angiogenesis / Endothelial":  "#8ecae6",
            "Hippo / YAP":                 "#a8dadc",
            "Metabolism":                  "#6d6875",
            "Development / Morphogenesis": "#b5838d",
            "Disease / Infection":         "#adb5bd",
            "Other / Miscellaneous":       "#ced4da",
        }

        def _cat_color(cat):
            return _CAT_PALETTE.get(cat, "#999999")

        pw_categories = [s.get("category", "Other / Miscellaneous") for s in top_pw]
        pw_colors     = [_cat_color(c) for c in pw_categories]

        # ── C9: Top KEGG/Reactome pathways (independent standalone figure) ─
        fig9 = plt.figure(figsize=(14, max(7, len(top_pw) * 0.5)))
        ax1  = fig9.add_subplot(111)

        pw_probs = [s["mean_edge_prob"] for s in top_pw]
        y_pos    = np.arange(len(top_pw))
        bars = ax1.barh(y_pos, pw_probs, color=pw_colors,
                        edgecolor="white", linewidth=0.3, height=0.72)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(pw_short, fontsize=17)
        ax1.invert_yaxis()
        ax1.set_xlabel("Mean SVRN LR probability", fontsize=17)
        ax1.set_title("Top KEGG/Reactome pathways\n(local = LR pairs matched)",
                      fontsize=19, fontweight="bold")
        max_prob = max(pw_probs) if pw_probs else 1.0
        for bar, s in zip(bars, top_pw):
            bw = bar.get_width()
            by = bar.get_y() + bar.get_height() / 2
            # n= count clipped inside bar tip (like reference)
            ax1.text(max(bw - max_prob * 0.005, 0), by,
                     f"n={s['n_lr_members']}",
                     va="center", ha="right", fontsize=16,
                     color="white", fontweight="bold")
            # Gene symbol pairs to the right of each bar
            gene_lbls = s.get("member_gene_labels", [])
            if gene_lbls:
                shown = ", ".join(gene_lbls[:3])
                if len(gene_lbls) > 3:
                    shown += f" +{len(gene_lbls)-3}"
                ax1.text(bw + max_prob * 0.008, by,
                         shown,
                         va="center", ha="left", fontsize=16,
                         color="#333333", style="italic")
        # Extend x-axis right margin for gene labels
        ax1.set_xlim(right=max_prob * 1.7)
        ax1.spines[["top", "right"]].set_visible(False)

        fig9.suptitle("Top LR pairs — KEGG/Reactome Pathway Enrichment",
                      fontsize=19, fontweight="bold", y=1.02)

        path9 = os.path.join(self.output_dir, "C9_top_lr_pairs.png")
        fig9.savefig(path9, dpi=300, bbox_inches="tight")
        plt.close(fig9)
        print(f"✓ Plot saved: {path9}")

        # ── C10: Pathway significance (independent standalone figure) ─────
        if not top_pw:
            # No pathway annotations matched (e.g. gene symbols not found in
            # the KEGG/Reactome mapping, or gene-set enrichment turned up
            # nothing significant) — nothing meaningful to plot, so skip C10
            # rather than crashing on empty-array reductions below.
            print("  No enriched pathways found — skipping C10 pathway-significance plot.")
            print(f"✓ Ranked LR pairs saved: {csv_path}")
            return

        fig10 = plt.figure(figsize=(13, max(7, len(top_pw) * 0.45)))
        ax3   = fig10.add_subplot(111)

        active_fracs  = np.array([s["active_frac"]         for s in top_pw])
        neg_log_p     = np.array([s["neg_log10_adj_p"]     for s in top_pw])
        n_members_arr = np.array([s["n_lr_members"]        for s in top_pw])
        # Scale bubble size generously to match reference visual weight
        bubble_sizes  = (n_members_arr / max(n_members_arr.max(), 1)) * 1800 + 300

        sc3 = ax3.scatter(active_fracs, neg_log_p,
                          s=bubble_sizes, c=pw_colors, alpha=0.80,
                          edgecolors="grey", linewidth=0.4, zorder=3)
        # Significance threshold (dashed line)
        ax3.axhline(-np.log10(0.05), color="#e63946", lw=0.9,
                    ls="--", alpha=0.8, label="adj-p = 0.05")
        ax3.set_xlabel("Mean active fraction of LR pairs", fontsize=17)
        ax3.set_ylabel("−log10(adjusted p-value)", fontsize=17)
        ax3.set_title("Pathway significance", fontsize=19, fontweight="bold")

        # Right-side legend listing ALL pathway names with colour swatches
        # (matching the reference layout where names float right of the bubbles)
        legend_elems3 = []
        from matplotlib.patches import Patch
        for i, (s, short) in enumerate(zip(top_pw, pw_short)):
            legend_elems3.append(
                Patch(facecolor=pw_colors[i], edgecolor="grey",
                      linewidth=0.4, label=short)
            )
        # Size legend entries
        import matplotlib.lines as mlines
        for sz_n in [2, 5, 10]:
            sz_pt = (sz_n / max(n_members_arr.max(), 1)) * 1800 + 300
            legend_elems3.append(
                mlines.Line2D([], [], marker="o", color="w",
                              markerfacecolor="grey", markersize=(sz_pt**0.5) * 0.45,
                              label=f"n={sz_n}", alpha=0.6)
            )
        ax3.legend(handles=legend_elems3, fontsize=13,
                   loc="upper left", bbox_to_anchor=(1.02, 1.0),
                   framealpha=0.5, borderpad=0.4, handlelength=1.0,
                   title="Pathway  |  size=n", title_fontsize=14)
        ax3.spines[["top", "right"]].set_visible(False)

        fig10.suptitle("Pathway Significance — KEGG/Reactome Enrichment",
                       fontsize=19, fontweight="bold", y=1.02)

        path10 = os.path.join(self.output_dir, "C10_pathway_significance.png")
        fig10.savefig(path10, dpi=300, bbox_inches="tight")
        plt.close(fig10)
        print(f"✓ Plot saved: {path10}")
        print(f"✓ Ranked LR pairs saved: {csv_path}")


    SEL_FREQ_THRESHOLD: float = 0.5

    def _build_downstream_masks(self, sel_freq_threshold: float = 0.5):
        """Compute per-cell masks and metadata for downstream plots."""
        freq_map = dict(zip(
            self.consensus_df["cell_type"],
            self.consensus_df["selection_frequency"],
        ))
        rank_map = dict(zip(
            self.consensus_df["cell_type"],
            self.consensus_df["consensus_pct_rank"],
        ))
        reliable_types = self.consensus_df.loc[
            self.consensus_df["selection_frequency"] >= sel_freq_threshold,
            "cell_type",
        ].values
        cell_freq = np.array([freq_map.get(ct, 0.0) for ct in self.all_cell_types])
        cell_rank = np.array([rank_map.get(ct, 0.0) for ct in self.all_cell_types])
        reliable_mask = np.isin(self.all_cell_types, reliable_types)
        return reliable_types, reliable_mask, cell_freq, cell_rank, freq_map, rank_map

    def plot_ds_influence_distribution(
        self, sel_freq_threshold: float = 0.5
    ) -> None:
        """Spatial scatter coloured by consensus influence; reliable cell types
        annotated with centroid labels (white stroke for readability)."""
        import matplotlib.patheffects as pe

        reliable_types, reliable_mask, *_ = self._build_downstream_masks(sel_freq_threshold)

        fig, ax = plt.subplots(figsize=(9, 8))
        sc = ax.scatter(
            self.all_spatial_coords[:, 0], self.all_spatial_coords[:, 1],
            c=self.cell_mean_norm, cmap="RdYlBu_r",
            s=8, linewidths=0, alpha=0.85, rasterized=True,
            vmin=0.0, vmax=1.0,
        )
        cbar = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("Consensus Influence Score (norm.)", fontsize=17)

        # Draw a convex-hull outline per reliable type for visual grouping
        for ct in reliable_types:
            mask = self.all_cell_types == ct
            if mask.sum() < 3:
                continue
            cx = self.all_spatial_coords[mask, 0].mean()
            cy = self.all_spatial_coords[mask, 1].mean()
            txt = ax.text(cx, cy, ct, fontsize=10, fontweight="bold",
                          ha="center", color="black")
            txt.set_path_effects([
                pe.Stroke(linewidth=2.5, foreground="white"), pe.Normal()
            ])

        ax.set_title(
            "Spatial Influence Distribution\n"
            f"(labels = reliable types, sel_freq ≥ {sel_freq_threshold})",
            fontsize=18,
        )
        ax.set_xlabel("X (µm)"); ax.set_ylabel("Y (µm)")
        ax.set_aspect("equal")
        fig.tight_layout()
        path = os.path.join(self.output_dir, "D1_influence_distribution_annotated.png")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"✓ Plot saved: {path}")

    def plot_ds_celltype_enrichment(
        self, sel_freq_threshold: float = 0.5
    ) -> None:
        """Horizontal bar chart of consensus pct-rank per cell type.
        Bar colour = selection frequency; hatching marks reliable types;
        error bars = mean fold std per cell type."""
        import matplotlib.cm as cm, matplotlib.colors as mcolors

        reliable_types, *_ = self._build_downstream_masks(sel_freq_threshold)
        df = self.consensus_df.sort_values("consensus_pct_rank", ascending=True)
        cell_types = df["cell_type"].values
        ranks      = df["consensus_pct_rank"].values
        freqs      = df["selection_frequency"].values

        # Per-cell-type std
        ct_std = np.array([
            self.cell_std[(self.all_cell_types == ct) & self.covered_mask].mean()
            if ((self.all_cell_types == ct) & self.covered_mask).any() else 0.0
            for ct in cell_types
        ])

        cmap  = cm.get_cmap("YlOrRd")
        norm  = mcolors.Normalize(vmin=0, vmax=1)
        colors = [cmap(norm(f)) for f in freqs]

        fig, ax = plt.subplots(figsize=(9, max(4, len(cell_types) * 0.48)))
        bars = ax.barh(cell_types, ranks, color=colors, edgecolor="grey",
                       linewidth=0.5, xerr=ct_std, capsize=3,
                       error_kw={"elinewidth": 1.0, "ecolor": "black"})

        for bar, ct in zip(bars, cell_types):
            if ct in reliable_types:
                bar.set_hatch("//")
                bar.set_edgecolor("black")

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.6)
        cbar.set_label("Selection Frequency", fontsize=17)

        ax.set_xlabel("Consensus Percentile Rank", fontsize=18)
        ax.set_title(
            f"Cell-type Enrichment (hatched = primary, sel_freq ≥ {sel_freq_threshold})\n"
            "Error bars = mean fold std",
            fontsize=18,
        )
        ax.axvline(50, color="grey", linestyle="--", linewidth=0.8, alpha=0.6,
                   label="Median rank")
        ax.legend(fontsize=15, loc="lower right")
        fig.tight_layout()
        path = os.path.join(self.output_dir, "D2_celltype_enrichment_gated.png")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"✓ Plot saved: {path}")

    def plot_ds_communication_corridors(
        self, sel_freq_threshold: float = 0.5
    ) -> None:
        """Spatial plot: edges between reliable-type cells only.
        Edge alpha ∝ geometric mean of endpoint consensus influence.
        Unreliable cells rendered at low opacity for context."""
        import matplotlib.collections as mc

        reliable_types, reliable_mask, _, cell_rank, *_ = self._build_downstream_masks(sel_freq_threshold)

        try:
            import scipy.sparse as _sp
            if _sp.issparse(self.adj_matrix):
                coo = self.adj_matrix.tocoo()
                rows, cols = coo.row, coo.col
            else:
                rows, cols = np.where(np.array(self.adj_matrix) > 0)
        except Exception:
            rows, cols = np.where(np.array(self.adj_matrix) > 0)

        keep = reliable_mask[rows] & reliable_mask[cols] & (rows < cols)
        rows, cols = rows[keep], cols[keep]

        norm_s = self.cell_mean_norm
        edge_w = np.sqrt(np.clip(norm_s[rows] * norm_s[cols], 0, 1))
        segments = [[self.all_spatial_coords[r], self.all_spatial_coords[c]]
                    for r, c in zip(rows, cols)]

        fig, ax = plt.subplots(figsize=(9, 8))

        # Background — unreliable cells
        unrel = ~reliable_mask
        if unrel.any():
            ax.scatter(self.all_spatial_coords[unrel, 0],
                       self.all_spatial_coords[unrel, 1],
                       c="lightgrey", s=4, alpha=0.20, rasterized=True, zorder=1)

        if segments:
            lc = mc.LineCollection(
                segments, linewidths=0.6,
                alpha=np.clip(edge_w * 0.85, 0.05, 0.65).tolist(),
                color="steelblue", zorder=2,
            )
            ax.add_collection(lc)

        sc = ax.scatter(
            self.all_spatial_coords[:, 0], self.all_spatial_coords[:, 1],
            c=cell_rank, cmap="plasma",
            s=np.where(reliable_mask, 10, 4),
            alpha=np.where(reliable_mask, 0.90, 0.15),
            linewidths=0, rasterized=True, zorder=3,
        )
        cbar = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("Consensus Pct Rank", fontsize=18)
        ax.set_title(
            f"Communication Corridors (edges between reliable types, n={len(segments)})\n"
            f"sel_freq ≥ {sel_freq_threshold}; edge alpha ∝ √(endpoint influence)",
            fontsize=18,
        )
        ax.set_xlabel("X (µm)"); ax.set_ylabel("Y (µm)")
        ax.set_aspect("equal"); ax.autoscale()
        fig.tight_layout()
        path = os.path.join(self.output_dir, "D3_corridors_reliable_types.png")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"✓ Plot saved: {path}")

    def plot_ds_uncertainty_map(
        self, sel_freq_threshold: float = 0.5
    ) -> None:
        """Dual-panel spatial map: left = mean influence, right = fold std.
        Unreliable-type cells are rendered at 10 % opacity in both panels."""
        _, reliable_mask, *_ = self._build_downstream_masks(sel_freq_threshold)

        std_vals = self.cell_std.copy()
        std_vals[~self.covered_mask] = np.nan
        mean_vals = self.cell_mean_norm.copy()
        mean_vals[~self.covered_mask] = np.nan

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        alpha_arr = np.where(reliable_mask, 0.90, 0.10)

        for ax, c_arr, cmap, label, title in [
            (axes[0], mean_vals, "RdYlBu_r",
             "Consensus Influence (norm.)", "Mean Influence"),
            (axes[1], std_vals, "cividis",
             "Fold Std (uncertainty)", "Fold Std (Uncertainty)"),
        ]:
            # Unreliable behind
            unrel = ~reliable_mask & self.covered_mask
            if unrel.any():
                ax.scatter(self.all_spatial_coords[unrel, 0],
                           self.all_spatial_coords[unrel, 1],
                           c=c_arr[unrel], cmap=cmap,
                           s=5, linewidths=0, alpha=0.10, rasterized=True,
                           vmin=np.nanmin(c_arr), vmax=np.nanmax(c_arr))
            # Reliable on top
            rel = reliable_mask & self.covered_mask
            if rel.any():
                sc = ax.scatter(
                    self.all_spatial_coords[rel, 0],
                    self.all_spatial_coords[rel, 1],
                    c=c_arr[rel], cmap=cmap,
                    s=8, linewidths=0, alpha=0.90, rasterized=True,
                    vmin=np.nanmin(c_arr), vmax=np.nanmax(c_arr),
                )
                fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02).set_label(label, fontsize=15)
            ax.set_title(f"{title}\n(low-freq types = 10% opacity)", fontsize=17)
            ax.set_xlabel("X (µm)"); ax.set_ylabel("Y (µm)")
            ax.set_aspect("equal")

        fig.suptitle(
            f"Spatial Uncertainty Map  (sel_freq ≥ {sel_freq_threshold})",
            fontsize=18, fontweight="bold",
        )
        fig.tight_layout()
        path = os.path.join(self.output_dir, "D4_uncertainty_map_gated.png")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"✓ Plot saved: {path}")

    def plot_ds_ci_top_cells(
        self, top_n: int = 30, sel_freq_threshold: float = 0.5
    ) -> None:
        """Horizontal bar chart of top-N cells from reliable types only.
        Each bar = mean influence; error bar = fold std; colour per cell type."""
        from matplotlib.patches import Patch

        _, reliable_mask, *_ = self._build_downstream_masks(sel_freq_threshold)
        rel_cov_mask = reliable_mask & self.covered_mask

        if not rel_cov_mask.any():
            print("  ⚠ No reliable-type covered cells — skipping D5.")
            return

        idx_rel     = np.where(rel_cov_mask)[0]
        scores_rel  = self.cell_mean[rel_cov_mask]
        top_local   = np.argsort(scores_rel)[::-1][:top_n]
        top_global  = idx_rel[top_local]

        top_scores  = self.cell_mean[top_global]
        top_stds    = self.cell_std[top_global]
        top_cts     = self.all_cell_types[top_global]

        unique_cts  = np.unique(top_cts)
        ct_cmap     = plt.get_cmap("tab20", max(len(unique_cts), 1))
        ct_colors   = {ct: ct_cmap(i / max(len(unique_cts), 1)) for i, ct in enumerate(unique_cts)}
        colors      = [ct_colors[ct] for ct in top_cts]

        fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.30)))
        y_pos = np.arange(top_n)
        ax.barh(y_pos, top_scores, xerr=top_stds, color=colors,
                edgecolor="none", height=0.72, capsize=3,
                error_kw={"elinewidth": 1.0, "ecolor": "black"})
        ax.set_yticks(y_pos)
        ax.set_yticklabels(
            [f"{ct}  [cell {top_global[i]}]" for i, ct in enumerate(top_cts)],
            fontsize=16,
        )
        ax.invert_yaxis()
        ax.set_xlabel("Consensus Influence Score (mean ± fold std)", fontsize=18)
        ax.set_title(
            f"Top-{top_n} Cells by Consensus Influence\n"
            f"(reliable types only, sel_freq ≥ {sel_freq_threshold})",
            fontsize=18,
        )
        fig.tight_layout()
        path = os.path.join(self.output_dir, "D5_top_cells_ci_reliable.png")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"✓ Plot saved: {path}")

    def plot_ds_influence_vs_uncertainty(
        self, sel_freq_threshold: float = 0.5
    ) -> None:
        """Scatter of mean influence vs fold std per cell, coloured by cell type.
        Reliable types are plotted at full opacity; unreliable types at 20%.
        Highlights high-influence / high-uncertainty cells (top-right quadrant)."""
        _, reliable_mask, *_ = self._build_downstream_masks(sel_freq_threshold)

        covered = self.covered_mask
        cts     = self.all_cell_types[covered]
        means   = self.cell_mean_norm[covered]
        stds    = self.cell_std[covered]
        rel     = reliable_mask[covered]

        unique_cts = np.unique(cts)
        ct_cmap    = plt.get_cmap("tab20", max(len(unique_cts), 1))
        ct_colors  = {ct: ct_cmap(i / max(len(unique_cts), 1)) for i, ct in enumerate(unique_cts)}

        fig, ax = plt.subplots(figsize=(9, 7))
        for ct in unique_cts:
            mask = cts == ct
            alpha = 0.75 if ct in (reliable_mask & covered) else 0.20
            # Use reliable_mask per cell
            is_reliable_arr = reliable_mask[covered]
            alpha_arr = np.where(is_reliable_arr[mask], 0.75, 0.20)
            ax.scatter(
                means[mask], stds[mask],
                s=8, color=ct_colors[ct],
                alpha=float(alpha_arr.mean()),
                label=ct, linewidths=0, rasterized=True,
            )

        # Quadrant lines at medians
        med_m = np.median(means)
        med_s = np.median(stds)
        ax.axvline(med_m, color="grey", lw=0.8, ls="--", alpha=0.6)
        ax.axhline(med_s, color="grey", lw=0.8, ls="--", alpha=0.6)
        ax.text(med_m * 1.01, stds.max() * 0.97,
                "High inf.\nHigh uncert.", fontsize=14, color="tomato", ha="left")

        ax.set_xlabel("Consensus Influence (norm. mean)", fontsize=17)
        ax.set_ylabel("Fold Std (epistemic uncertainty)", fontsize=17)
        ax.set_title(
            "Influence vs Uncertainty per Cell\n"
            f"(reliable types = full opacity, sel_freq ≥ {sel_freq_threshold})",
            fontsize=17,
        )
        ax.legend(fontsize=14, markerscale=2, ncol=1, loc='upper right', bbox_to_anchor=(0.98, 0.98),
                  framealpha=0.7, title="Cell Type", title_fontsize=14)
        
        ax.grid(linestyle="--", alpha=0.30)
        fig.tight_layout()
        path = os.path.join(self.output_dir, "D6_influence_vs_uncertainty.png")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"✓ Plot saved: {path}")
    def plot_consensus_kn_scatter(self) -> None:
       
        csv_path = os.path.join(self.output_dir, "C8_hill_kn_summary.csv")
        if not os.path.exists(csv_path):
            print("  ⚠ C8_hill_kn_summary.csv not found — "
                  "run plot_hill_kn_affinity() first. Skipping.")
            return

        df = pd.read_csv(csv_path)
        if "mean_K" not in df.columns or "mean_n" not in df.columns:
            print("  ⚠ C8_hill_kn_summary.csv missing mean_K/mean_n columns. Skipping.")
            return

        mean_K = df["mean_K"].to_numpy(dtype=float)
        mean_n = df["mean_n"].to_numpy(dtype=float)

        # ── Font: Times New Roman for all plot text (incl. mathtext) ───────
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Times New Roman"]
        plt.rcParams["mathtext.fontset"] = "custom"
        plt.rcParams["mathtext.rm"] = "Times New Roman"
        plt.rcParams["mathtext.it"] = "Times New Roman:italic"
        plt.rcParams["mathtext.bf"] = "Times New Roman:bold"

        # Cooperativity colour per dot (matches reference blue-dominant palette)
        dot_colors = []
        for n in mean_n:
            if n > 1.2:
                dot_colors.append("tomato")
            elif n < 0.8:
                dot_colors.append("#06d6a0")
            else:
                dot_colors.append("#4c8cbf")   # reference image steel-blue

        median_K = float(np.median(mean_K))
        median_n = float(np.median(mean_n))

        fig, ax = plt.subplots(figsize=(8, 6))

        # ── Scatter — clean dots, no error bars (reference style) ─────────
        ax.scatter(mean_K, mean_n, c=dot_colors,
                   s=45, alpha=0.80, edgecolors="white", linewidths=0.6,
                   zorder=3)

        # ── Dashed median crosshairs ───────────────────────────────────────
        ax.axvline(median_K, color="grey", linestyle="--", lw=1.1, alpha=0.75,
                   label=f"median $K$ = {median_K:.4f}")
        ax.axhline(median_n, color="grey", linestyle="--", lw=1.1, alpha=0.75,
                   label=f"median $n$ = {median_n:.4f}")

        # ── Axis limits with small padding (tight, like reference) ─────────
        x_pad = (mean_K.max() - mean_K.min()) * 0.06 + 1e-8
        y_pad = (mean_n.max() - mean_n.min()) * 0.08 + 1e-8
        x_min, x_max = mean_K.min() - x_pad, mean_K.max() + x_pad
        y_min, y_max = mean_n.min() - y_pad, mean_n.max() + y_pad
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        # ── Quadrant count annotations ─────────────────────────────────────
        q_masks = {
            "LL": (mean_K <= median_K) & (mean_n <= median_n),
            "LH": (mean_K <= median_K) & (mean_n >  median_n),
            "RL": (mean_K >  median_K) & (mean_n <= median_n),
            "RH": (mean_K >  median_K) & (mean_n >  median_n),
        }
        q_pos = {
            "LL": (x_min + x_pad * 0.6, y_min + y_pad * 0.7, "left",  "bottom"),
            "LH": (x_min + x_pad * 0.6, y_max - y_pad * 0.7, "left",  "top"),
            "RL": (x_max - x_pad * 0.6, y_min + y_pad * 3.1, "right", "bottom"),
            "RH": (x_max - x_pad * 0.6, y_max - y_pad * 0.7, "right", "top"),
        }
        q_desc = {
            "LL": "Low $K$\nLow $n$",
            "LH": "Low $K$\nHigh $n$",
            "RL": "High $K$\nLow $n$",
            "RH": "High $K$\nHigh $n$",
        }
        for qk, qmask in q_masks.items():
            xp, yp, ha_, va_ = q_pos[qk]
            ax.text(xp, yp,
                    f"{q_desc[qk]}\nn={int(qmask.sum())}",
                    ha=ha_, va=va_, fontsize=14, color="#555555",
                    fontname="Times New Roman",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white",
                              ec="#cccccc", alpha=0.75),
                    zorder=1)

        # ── Axes labels and title (matching reference image) ───────────────
        ax.set_xlabel("$K$ (affinity constant)", fontsize=17, fontname="Times New Roman")
        ax.set_ylabel("$n$ (Hill coefficient)", fontsize=17, fontname="Times New Roman")
        ax.set_title("LR pair binding regimes", fontsize=18, fontweight="bold",
                     fontname="Times New Roman")

        # ── Tick label font ──────────────────────────────────────────────
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontname("Times New Roman")

        # ── Legend ────────────────────────────────────────────────────────
        from matplotlib.lines import Line2D
        legend_elems = [
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="tomato", markersize=12,
                   label="Positive cooperativity ($n$ > 1.2)"),
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="#06d6a0", markersize=12,
                   label="Negative cooperativity ($n$ < 0.8)"),
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="#4c8cbf", markersize=12,
                   label="Neutral (0.8 ≤ $n$ ≤ 1.2)"),
            Line2D([0], [0], color="grey", lw=1.1, ls="--",
                   label=f"Median $K$ = {median_K:.4f}"),
            Line2D([0], [0], color="grey", lw=1.1, ls="--",
                   label=f"Median $n$ = {median_n:.4f}"),
        ]
        # Legend goes outside the axes so it can't collide with the
        # quadrant-count annotation in the bottom-right corner.
        legend = ax.legend(handles=legend_elems, fontsize=15, loc="upper left",
                  bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0,
                  framealpha=0.9)
        for text in legend.get_texts():
            text.set_fontname("Times New Roman")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()

        out_path = os.path.join(self.output_dir,
                                "C8_hill_kn_consensus_scatter.png")
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"✓ Consensus K–n scatter saved: {out_path}")
    
    def run_all(self, sel_freq_threshold: float = 0.5) -> None:
        print(f"\n{'='*70}")
        print(f"CONSENSUS PLOTS  (sel_freq threshold={sel_freq_threshold})")
        print(f"{'='*70}")
        # ── Core consensus plots (C1–C9) ──────────────────────────────────
        self.plot_influence_distribution()
        self.plot_cell_type_enrichment()
        self.plot_spatial_influence()
        self.plot_communication_corridors()
        self.plot_uncertainty_spatial()
        self.plot_uncertainty_by_cell_type()
        self.plot_confidence_intervals_top_cells()
        self.plot_hill_kn_affinity()
        self.plot_consensus_kn_scatter()          # C8b — reference-style K–n scatter
        self.plot_top_lr_pairs(species=getattr(self, "_species", "mouse"))
        # ── Downstream sel_freq-gated plots (D1–D6) ───────────────────────
        print(f"\n── Downstream (sel_freq-gated) plots ──")
        self.plot_ds_influence_distribution(sel_freq_threshold)
        self.plot_ds_celltype_enrichment(sel_freq_threshold)
        self.plot_ds_communication_corridors(sel_freq_threshold)
        self.plot_ds_uncertainty_map(sel_freq_threshold)
        self.plot_ds_ci_top_cells(top_n=30, sel_freq_threshold=sel_freq_threshold)
        self.plot_ds_influence_vs_uncertainty(sel_freq_threshold)
        print(f"\n✓ All consensus + downstream plots saved in: {self.output_dir}")
