# -*- coding: utf-8 -*-
"""
svrn.model
==========
The SVRN neural-network architecture: Hill-kinetics ligand–receptor
interaction scoring, LR-aware gated attention, the stochastic
variational relay that propagates communication signal over the
spatial graph, edge-aware graph diffusion smoothing, and the top-level
``SVRN`` LightningModule that composes them into a trainable model.

Split out of the original monolithic ``pipeline.py``; this module has
no dependency on data loading, metrics, or plotting code, only on
:class:`svrn.utils.Config` for typing/config access.
"""

import warnings
from typing import Dict, Any, List, Optional, Tuple

warnings.filterwarnings("ignore")

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

from torch_geometric.data import Data

from .utils import Config


class HillInteraction(nn.Module):
    def __init__(self, n_lr: int):
        super().__init__()
        self.n_lr = n_lr
        self.log_K = nn.Parameter(torch.zeros(n_lr))
        self.log_n = nn.Parameter(torch.zeros(n_lr))

    def forward(self, L_src: torch.Tensor, R_dst: torch.Tensor, spatial_dist: torch.Tensor) -> torch.Tensor:
        prod = L_src * R_dst + 1e-8
        n = F.softplus(self.log_n) + 0.1

        logits = n * (torch.log(prod) - self.log_K)
        logits = torch.clamp(logits, min=-10.0, max=10.0)

        hill = torch.sigmoid(logits)

        spatial_dist = spatial_dist / (torch.median(spatial_dist) + 1e-8)
        spatial_dist = torch.clamp(spatial_dist, min=0.0, max=5.0)
        
        # MERFISH Optimization: Sharpen spatial decay to track localized single-cell boundaries
        spatial_weight = torch.exp(-2.0 * torch.pow(spatial_dist, 2)).unsqueeze(1)

        return hill * spatial_weight

    def regularization_loss(self) -> torch.Tensor:
        n = F.softplus(self.log_n) + 0.1
        return torch.mean((n - 1.0).pow(2)) + torch.mean(self.log_K.pow(2))


# =====================================================================
# 2. LR-Aware Gated Attention
# =====================================================================
class LRPAwareGatedAttention(nn.Module):
    def __init__(self, in_channels: int, n_lr: int, heads: int = 2,
                 dropout: float = 0.1, edge_chunk_size: int = 512):
        super().__init__()
        self.n_lr = n_lr
        self._edge_chunk_size = edge_chunk_size

        self.node_enc = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.LayerNorm(64),
        )

        edge_in = 64 + 64 + n_lr + 1
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_in, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, n_lr),
        )

        self.residual_gate = nn.Sequential(
            nn.Linear(n_lr * 2, n_lr),
            nn.Sigmoid(),
        )

        # Projection layers for source and destination node embeddings
        self.W_send = nn.Linear(64, 64)
        self.W_recv = nn.Linear(64, 64)

    def forward(
        self,
        x: torch.Tensor,           
        edge_index: torch.Tensor,  
        hill_out: torch.Tensor,    
        spatial_dist: torch.Tensor,
        L_src: torch.Tensor,       
        R_dst: torch.Tensor,       
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        src, dst = edge_index

        h = self.node_enc(x)                                         

        spatial_scalar = torch.exp(-spatial_dist).unsqueeze(1) 
        
        # Call the pre-defined layers safely on the GPU
        edge_feat = torch.cat([
            self.W_send(h[src]), 
            self.W_recv(h[dst]), 
            hill_out.detach(), 
            spatial_scalar
        ], dim=1)

        logits_mlp = self.edge_mlp(edge_feat)         

        hill_centered = hill_out.clamp(1e-6, 1 - 1e-6)
        hill_mean = hill_centered.mean(dim=0, keepdim=True).clamp(1e-6, 1 - 1e-6)
        hill_logit = torch.logit(hill_centered) - torch.logit(hill_mean)  

        gate = self.residual_gate(
            torch.cat([logits_mlp, hill_out.detach()], dim=1)
        )                                                                             
        blended_logits = gate * logits_mlp + (1 - gate) * hill_logit

        edge_probs_lr  = torch.sigmoid(blended_logits).clamp(1e-6, 1 - 1e-6)  
        edge_logits_lr = blended_logits                                         
        edge_scores_global = edge_probs_lr.mean(dim=1)                         

        return edge_logits_lr, edge_probs_lr, edge_scores_global


# =====================================================================
# 3. Stochastic Variational Relay
# =====================================================================
class StochasticVariationalRelay(nn.Module):
    def __init__(self, steps: int = 3, latent_dim: int = 2):
        super().__init__()
        self.steps = steps
        self.latent_dim = latent_dim

        self.encoder_mu = nn.Linear(1, latent_dim)
        self.encoder_logvar = nn.Linear(1, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, max(latent_dim, 4)),
            nn.ReLU(),
            nn.Linear(max(latent_dim, 4), 1),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)

        nn.init.constant_(self.encoder_logvar.bias, -0.5)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # KL is minimized at logvar=0 and grows in EITHER direction away
        # from 0 (see forward()). A symmetric clamp avoids systematically
        # biasing the encoder toward one side of the convex KL bowl.
        logvar = torch.clamp(logvar, min=-3.0, max=3.0)

        if not self.training:
         
            return mu

        std = torch.exp(0.5 * logvar) + 1e-8
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        probs: torch.Tensor,
        n_nodes: int,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor, torch.Tensor]:
        device = probs.device
        original_dtype = probs.dtype

        probs_f32 = probs.float().view(-1, 1).clamp(0.0, 1.0)

        mu     = self.encoder_mu(probs_f32)
        logvar = self.encoder_logvar(probs_f32).clamp(min=-3.0, max=3.0)

        self.mu_mean     = mu.mean().detach().item()
        self.logvar_mean = logvar.mean().detach().item()

        z      = self.reparameterize(mu, logvar)
        logits = self.decoder(z).squeeze(-1)           

        variational_probs = torch.sigmoid(logits).float() + 1e-4  

        with torch.no_grad():
            row_sum = torch.zeros(n_nodes, device=device, dtype=torch.float32)
            row_sum.scatter_add_(0, dst, variational_probs.detach())
            row_sum.clamp_(min=1e-8)

        norm_values = variational_probs / row_sum[dst]   

        indices = torch.stack([dst, src], dim=0)
        P_T = torch.sparse_coo_tensor(
            indices,
            norm_values.detach().float(),   
            (n_nodes, n_nodes),
            device=device,
        ).coalesce()

        h = torch.ones((n_nodes, 1), device=device, dtype=torch.float32) / max(n_nodes, 1)
        for _ in range(self.steps):
            h = torch.sparse.mm(P_T, h)   

        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        kl_div     = kl_per_dim.sum(dim=1).mean()
        self.kl_loss = kl_div
        self.mu_for_loss = mu  # retained with grad for an explicit mu^2 penalty

        return h.to(original_dtype), variational_probs.to(original_dtype), kl_div

# =====================================================================
# 4. Graph Diffusion Smoother
# =====================================================================
class GraphDiffusionSmoother(nn.Module):
  
    def __init__(self, n_steps: int = 3, alpha: float = 0.4, kernel_scale: float = 0.75):
        super().__init__()
        self.n_steps = max(1, n_steps)
        self.alpha = alpha
        self.kernel_scale = kernel_scale

    def _diffuse_once(self, influence: torch.Tensor, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        src, dst = edge_index
        diff = (influence[src] - influence[dst]).pow(2)
        edge_weight = torch.exp(-torch.clamp(diff * self.kernel_scale, 0.0, 10.0))

        h_new = torch.zeros_like(influence)
        deg = torch.zeros(num_nodes, 1, device=influence.device, dtype=influence.dtype)

        h_new.scatter_add_(0, dst.unsqueeze(-1), influence[src] * edge_weight)
        deg.scatter_add_(0, dst.unsqueeze(-1), edge_weight)

        smoothed = h_new / (deg + 1e-8)
    
        has_deg = deg > 1e-6
        return torch.where(has_deg, smoothed, influence)

    def forward(self, influence: torch.Tensor, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        out = influence
        for _ in range(self.n_steps):
            smoothed = self._diffuse_once(out, edge_index, num_nodes)
            out = self.alpha * influence + (1 - self.alpha) * smoothed
        return out


# =====================================================================
# 5. SVRN Core Model
# =====================================================================
class SVRN(pl.LightningModule):
    def __init__(self, cfg: Config):
        super().__init__()
        self.save_hyperparameters(ignore=["cfg"])
        self.cfg = cfg

        self.train_losses = []
        self.val_losses = []
        self.val_epochs = []   # epoch index at which each val checkpoint was recorded
        self.kl_beta = 0.0
        
        self.node_encoder = nn.Sequential(
            nn.Linear(cfg.N_GENES, cfg.HIDDEN_DIM),
            nn.LayerNorm(cfg.HIDDEN_DIM),
            nn.ReLU(),
            nn.LayerNorm(cfg.HIDDEN_DIM),  
        )

        self.hill = HillInteraction(cfg.N_LR)
        self.attention = LRPAwareGatedAttention(
            cfg.HIDDEN_DIM, cfg.N_LR, dropout=cfg.DROPOUT,
            edge_chunk_size=cfg.EDGE_CHUNK_SIZE,
        )
        self.relay = StochasticVariationalRelay(steps=cfg.MULTI_HOP_STEPS)
     
        self.diffusion = GraphDiffusionSmoother(n_steps=6, alpha=0.15, kernel_scale=0.40)

        # Influence head now takes: z (HIDDEN_DIM) + relay scalar (1) — no ct_emb
        self.influence_head = nn.Sequential(
            nn.Linear(cfg.HIDDEN_DIM + 1, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        if cfg.CT_PRIOR is not None:
            self.register_buffer("ct_prior", cfg.CT_PRIOR)
        else:
            self.ct_prior = None

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        lr_features: torch.Tensor,
        spatial_coords: torch.Tensor,
        cell_type_idx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_nodes = x.size(0)
        z = self.node_encoder(x)

        lr_reshaped = lr_features.view(num_nodes, self.cfg.N_LR, 2)
        L = lr_reshaped[:, :, 0]
        R = lr_reshaped[:, :, 1]

        src, dst = edge_index

        max_e = self.cfg.MAX_EDGES_PER_STEP
        n_edges = src.size(0)

        # max_e == 0 means "use all edges" (no subsampling).
        # Guard: also fall through to full-graph path if the graph is empty.
        if max_e > 0 and n_edges > max_e:
            if self.training:
                # Random subset — unbiased stochastic gradient estimator.
                perm = torch.randperm(n_edges, device=src.device)[:max_e]
            else:
                # Deterministic uniform stride — reproducible, covers the full
                # graph without replacement bias.
                step = max(1, n_edges // max_e)
                perm = torch.arange(0, n_edges, step, device=src.device)[:max_e]
            src_u, dst_u = src[perm], dst[perm]
            edge_index_used = torch.stack([src_u, dst_u], dim=0)
        else:
            # max_e == 0 or graph fits within budget: use every edge.
            src_u, dst_u, edge_index_used = src, dst, edge_index

        spatial_dist = torch.norm(
            spatial_coords[src_u].float() - spatial_coords[dst_u].float(), dim=1,
        )
        spatial_dist = spatial_dist / (torch.median(spatial_dist) + 1e-8)
        spatial_dist = torch.clamp(spatial_dist, min=0.0, max=5.0)

        hill_out = self.hill(L[src_u], R[dst_u], spatial_dist)

        edge_logits_lr, edge_probs_lr, edge_scores_global = self.attention(
            z, edge_index_used, hill_out, spatial_dist, L[src_u], R[dst_u],
        )

        h, _, kl_div = self.relay(src_u, dst_u, edge_scores_global, num_nodes)

        relay_scalar = self.diffusion(h, edge_index, num_nodes)

        influence_input = torch.cat([z, relay_scalar], dim=1)
        final_influence_raw = self.influence_head(influence_input).squeeze(-1)

        final_influence = self.diffusion(
            final_influence_raw.unsqueeze(-1), edge_index, num_nodes
        ).squeeze(-1)

        return final_influence, kl_div, edge_logits_lr, edge_probs_lr, edge_index_used

    def _compute_loss(
        self,
        influence: torch.Tensor,
        kl_div: torch.Tensor,
        edge_logits_lr: torch.Tensor,
        edge_probs_lr: torch.Tensor,
        edge_index: torch.Tensor,
        batch,
    ):
        N_LR = edge_probs_lr.size(1)
        num_nodes = batch.x.size(0)
        lr_reshaped = batch.lr_features.view(num_nodes, N_LR, 2)
        L = lr_reshaped[:, :, 0]
        R = lr_reshaped[:, :, 1]
        src, dst = edge_index
        E = src.size(0)

        chunk = getattr(self.hparams, "edge_chunk_size", 256)
        num_chunks = (E + chunk - 1) // chunk

        lr_fidelity  = torch.zeros((), device=L.device)
        f1_fidelity  = torch.zeros((), device=L.device)
        auc_rank     = torch.zeros((), device=L.device)
        f1_rank      = torch.zeros((), device=L.device)

        # ── Pre-compute target matrix with continuous logit scaling ──
        with torch.no_grad():
            raw_inter_full = L[src] * R[dst]          

            col_min = raw_inter_full.min(dim=0, keepdim=True).values
            col_max = raw_inter_full.max(dim=0, keepdim=True).values
            range_mask = (col_max - col_min) > 1e-8
            
            target_full = torch.zeros_like(raw_inter_full)
            target_full[:, range_mask.squeeze(0)] = (raw_inter_full[:, range_mask.squeeze(0)] - col_min[:, range_mask.squeeze(0)]) / \
                                                    (col_max[:, range_mask.squeeze(0)] - col_min[:, range_mask.squeeze(0)] + 1e-8)
            
            target_full = torch.pow(target_full, 2)

            target_raw_full = target_full.clone()

            if self.ct_prior is not None and hasattr(batch, "cell_type_idx"):
                src_types_full = batch.cell_type_idx[src]
                dst_types_full = batch.cell_type_idx[dst]
                type_prob_full = self.ct_prior[src_types_full, dst_types_full].unsqueeze(1)  

                raw_d = torch.norm(
                    batch.spatial_coords[src].float() - batch.spatial_coords[dst].float(), dim=1
                )
                spatial_dist_full = (raw_d / (raw_d.median() + 1e-8)).clamp(0.0, 5.0)
                spatial_weight_full = torch.exp(-spatial_dist_full).unsqueeze(1)  

                target_full = target_full * spatial_weight_full * type_prob_full  

            # FIX: Build binary labels dynamically using thresholds computed solely among active pairs
            labels_binary_full = torch.zeros_like(target_full)
            for j in range(N_LR):
                col_targ = target_full[:, j]
                active_mask = col_targ > 1e-5
                if active_mask.sum() < 5:
                    continue
                nonzero_vals = col_targ[active_mask]
                thresh_j = torch.quantile(nonzero_vals, 0.75)
                labels_binary_full[:, j] = ((col_targ >= thresh_j) & active_mask).float()

            labels_raw_median_full = torch.zeros_like(raw_inter_full)
            for j in range(N_LR):
                col_raw = raw_inter_full[:, j]
                active_mask = col_raw > 1e-9
                if active_mask.sum() < 5:
                    continue
                thresh_j = torch.median(col_raw[active_mask])
                labels_raw_median_full[:, j] = ((col_raw >= thresh_j) & active_mask).float()



            if hasattr(batch, "cell_type_idx"):
                with torch.no_grad():
                    n_types = int(batch.cell_type_idx.max().item()) + 1
                    # Per-type frequency in THIS batch (stable to 0-count types)
                    type_counts = torch.bincount(
                        batch.cell_type_idx, minlength=n_types
                    ).float().clamp(min=1.0)
                    type_freq = type_counts / type_counts.sum()          # (n_types,)
                    # Edge weight = 1 / (freq_src * freq_dst), normalised to mean=1
                    edge_ifw_full = 1.0 / (
                        type_freq[batch.cell_type_idx[src]]
                        * type_freq[batch.cell_type_idx[dst]]
                        + 1e-8
                    )
                    edge_ifw_full = edge_ifw_full / (edge_ifw_full.mean() + 1e-8)
            else:
                edge_ifw_full = torch.ones(E, device=L.device)

        # ── Loss accumulation loop ────────────────────────────────────────────────
        for i in range(0, E, chunk):
            sl = slice(i, min(i + chunk, E))
            probs_c = edge_probs_lr[sl]                    
            logits_c = edge_logits_lr[sl]                  
            target = target_full[sl]                 
            target_raw = target_raw_full[sl]
            labels_bin = labels_binary_full[sl]
            labels_raw_med = labels_raw_median_full[sl]
            # Per-edge abundance-correction weights for this chunk
            ifw_c = edge_ifw_full[sl].unsqueeze(1)         # (chunk, 1) → broadcasts over LR dim

            # ① Focal BCE Loss  (abundance-corrected)
            FOCAL_GAMMA = 2.0
            FOCAL_ALPHA = 0.75  
            bce_raw = F.binary_cross_entropy(probs_c, target.clamp(0.0, 1.0), reduction="none")
            p_t = torch.where(target > 0.5, probs_c, 1.0 - probs_c)
            alpha_t = torch.where(target > 0.5,
                                  torch.full_like(probs_c, FOCAL_ALPHA),
                                  torch.full_like(probs_c, 1.0 - FOCAL_ALPHA))
            focal_weight = alpha_t * (1.0 - p_t).pow(FOCAL_GAMMA)
            fidelity_chunk = (ifw_c * focal_weight * bce_raw).sum() / (ifw_c.sum() + 1e-8)

            bce_raw_f1 = F.binary_cross_entropy(probs_c, target_raw.clamp(0.0, 1.0), reduction="none")
            p_t_f1 = torch.where(target_raw > 0.5, probs_c, 1.0 - probs_c)
            alpha_t_f1 = torch.where(target_raw > 0.5,
                                  torch.full_like(probs_c, FOCAL_ALPHA),
                                  torch.full_like(probs_c, 1.0 - FOCAL_ALPHA))
            focal_weight_f1 = alpha_t_f1 * (1.0 - p_t_f1).pow(FOCAL_GAMMA)
            f1_fidelity_chunk = (ifw_c * focal_weight_f1 * bce_raw_f1).sum() / (ifw_c.sum() + 1e-8)

            # ② Unconstrained Logit-Space List Rank Engine  (abundance-corrected)
            rank_bce = F.binary_cross_entropy_with_logits(
                logits_c, labels_bin, reduction="none"
            )
            
            with torch.no_grad():
                ranks = torch.argsort(torch.argsort(-probs_c, dim=0), dim=0)
                pos_weights = 1.0 / torch.log2(ranks.float() + 2.0)
            
            auc_chunk = (ifw_c * pos_weights * rank_bce).sum() / (ifw_c.sum() + 1e-8)

            margin_terms = []
            for j in range(N_LR):
                col_logits = logits_c[:, j]
                col_labels = labels_bin[:, j]
                col_ifw   = ifw_c[:, 0]
                pos_mask = col_labels > 0.5
                neg_mask = col_labels <= 0.5
                if pos_mask.sum() > 0 and neg_mask.sum() > 0:
                    # Weighted mean logits so rare-type edges count more
                    mean_pos_logit = (col_logits[pos_mask] * col_ifw[pos_mask]).sum() / (col_ifw[pos_mask].sum() + 1e-8)
                    mean_neg_logit = (col_logits[neg_mask] * col_ifw[neg_mask]).sum() / (col_ifw[neg_mask].sum() + 1e-8)
                    # Penalize if the separation distance is less than 3.0 nats
                    margin_terms.append(F.relu(3.0 - (mean_pos_logit - mean_neg_logit)))
            
            if len(margin_terms) >= 3:  # only fire if enough LR channels have both pos and neg
                margin_loss = torch.stack(margin_terms).mean()
            else:
                margin_loss = torch.zeros((), device=L.device)

            f1_rank_bce = F.binary_cross_entropy_with_logits(
                logits_c, labels_raw_med, reduction="none"
            )
            f1_rank_chunk = (ifw_c * f1_rank_bce).sum() / (ifw_c.sum() + 1e-8)

            f1_margin_terms = []
            for j in range(N_LR):
                col_logits = logits_c[:, j]
                col_labels = labels_raw_med[:, j]
                col_ifw   = ifw_c[:, 0]
                pos_mask = col_labels > 0.5
                neg_mask = col_labels <= 0.5
                if pos_mask.sum() > 0 and neg_mask.sum() > 0:
                    mean_pos_logit = (col_logits[pos_mask] * col_ifw[pos_mask]).sum() / (col_ifw[pos_mask].sum() + 1e-8)
                    mean_neg_logit = (col_logits[neg_mask] * col_ifw[neg_mask]).sum() / (col_ifw[neg_mask].sum() + 1e-8)
                    f1_margin_terms.append(F.relu(3.0 - (mean_pos_logit - mean_neg_logit)))

            if len(f1_margin_terms) >= 3:
                f1_margin_loss = torch.stack(f1_margin_terms).mean()
            else:
                f1_margin_loss = torch.zeros((), device=L.device)

            lr_fidelity = lr_fidelity + fidelity_chunk / num_chunks
            f1_fidelity = f1_fidelity + f1_fidelity_chunk / num_chunks
            auc_rank    = auc_rank    + (auc_chunk + 0.5 * margin_loss) / num_chunks
            f1_rank     = f1_rank     + (f1_rank_chunk + 0.5 * f1_margin_loss) / num_chunks
        

        edge_std = edge_probs_lr.mean(dim=1).std(unbiased=False).clamp(min=1e-8)

        col_stds = edge_probs_lr.std(dim=0, unbiased=False)            
        spread_penalty = F.relu(0.15 - col_stds).mean()                

        # kl_term = min(0.003, self.current_epoch / max(self.cfg.EPOCHS * 0.6, 1)) * torch.clamp(kl_div, max=0.15)
        if self.training:

            kl_beta = min(2.0, self.current_epoch / max(self.cfg.EPOCHS * 0.3, 1) * 2.0)
            kl_term = kl_beta * kl_div
     
            kl_ceil = 10.0 * F.relu(kl_div - 0.35).pow(2)
     
            mu_for_loss = getattr(self.relay, "mu_for_loss", None)
            if mu_for_loss is not None:
                mu_l2 = 0.5 * kl_beta * mu_for_loss.pow(2).mean()
            else:
                mu_l2 = torch.zeros((), device=kl_div.device)
            kl_term = kl_term + kl_ceil + mu_l2
        else:
            kl_term = torch.zeros((), device=kl_div.device)

        lr_active = (L[src] * R[dst]).abs().mean().item() > 1e-6
       
        lr_weight = 2.0 if lr_active else 0.05

        loss = (lr_weight * lr_fidelity
                + lr_weight * f1_fidelity     # F1-aligned fidelity (raw L*R ranking)
                + 1.5 * auc_rank
                + 3.0 * f1_rank                # F1-aligned ranking (raw L*R median split)
                + 0.60 * spread_penalty
                + kl_term
                + 0.01 * self.hill.regularization_loss())
        
        metrics = {
            "loss":        loss.detach(),
            "kl":          kl_div.detach(),
            "lr_fidelity": lr_fidelity.detach(),
            "f1_fidelity": f1_fidelity.detach(),
            "auc_rank":    auc_rank.detach(),
            "f1_rank":     f1_rank.detach(),
            "inf_std":     edge_std.detach(),
            "col_spread":  col_stds.mean().detach(),
        }
        if self.trainer.global_step % 50 == 0:
            print({k: round(v.item(), 4) for k, v in metrics.items()})

        return loss, metrics
       
    def training_step(self, batch: Data, batch_idx: int) -> torch.Tensor:
        influence, kl_div, edge_logits_lr, edge_probs_lr, edge_index_used = self(
            batch.x, batch.edge_index, batch.lr_features, batch.spatial_coords,
            getattr(batch, "cell_type_idx", None),
        )
        loss, metrics = self._compute_loss(
            influence, kl_div, edge_logits_lr, edge_probs_lr, edge_index_used, batch
        )
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: Data, batch_idx: int) -> torch.Tensor:
        influence, kl_div, edge_logits_lr, edge_probs_lr, edge_index_used = self(
            batch.x, batch.edge_index, batch.lr_features, batch.spatial_coords,
            getattr(batch, "cell_type_idx", None),
        )
        loss, metrics = self._compute_loss(
            influence, kl_div, edge_logits_lr, edge_probs_lr, edge_index_used, batch
        )
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def on_train_epoch_start(self) -> None:
        
        self.kl_beta = min(0.003, self.current_epoch / max(self.cfg.EPOCHS * 0.6, 1))
       
    def on_train_epoch_end(self) -> None:
        metric = self.trainer.callback_metrics.get("train_loss")
        if metric is not None:
            self.train_losses.append(float(metric.detach().cpu()))

    def on_validation_epoch_end(self) -> None:
        metric = self.trainer.callback_metrics.get("val_loss")
        if metric is not None:
            self.val_losses.append(float(metric.detach().cpu()))
            self.val_epochs.append(self.current_epoch)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.LR, weight_decay=1e-5)
        
        warmup_epochs = 5
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
        )
        # Option A — Best for your case (recommended)
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.cfg.EPOCHS*2,      
            eta_min=1e-6
        )
        
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_sched, cosine_sched],
            milestones=[warmup_epochs],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
# =====================================================================
# 6. Data Preprocessor
# =====================================================================