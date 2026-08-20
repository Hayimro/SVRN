"""
Lightweight unit tests for the core SVRN building blocks.

These tests are intentionally CPU-only, use tiny tensor sizes, and avoid
any file I/O so they run quickly in CI without GPU access or example data.
For a full end-to-end smoke test against a synthetic AnnData object, see
`example_usage()` in `svrn.pipeline` (also run separately in CI via
`python -m svrn.pipeline --run_example`).
"""
import torch

from svrn import (
    HillInteraction,
    LRPAwareGatedAttention,
    StochasticVariationalRelay,
    GraphDiffusionSmoother,
    set_seed,
    get_device,
)


def test_set_seed_is_deterministic():
    set_seed(0)
    a = torch.rand(5)
    set_seed(0)
    b = torch.rand(5)
    assert torch.allclose(a, b)


def test_get_device_returns_torch_device():
    device = get_device()
    assert isinstance(device, torch.device)


def test_hill_interaction_output_shape_and_range():
    n_lr = 4
    n_edges = 10
    module = HillInteraction(n_lr=n_lr)

    L_src = torch.rand(n_edges, n_lr) + 0.1
    R_dst = torch.rand(n_edges, n_lr) + 0.1
    spatial_dist = torch.rand(n_edges).abs() + 0.1

    out = module(L_src, R_dst, spatial_dist)

    assert out.shape == (n_edges, n_lr)
    assert torch.all(out >= 0.0) and torch.all(out <= 1.0)
    assert torch.all(torch.isfinite(out))


def test_hill_interaction_regularization_loss_is_scalar_and_finite():
    module = HillInteraction(n_lr=4)
    loss = module.regularization_loss()
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_lrp_aware_gated_attention_output_shapes():
    n_nodes, n_lr, in_channels = 12, 4, 8
    n_edges = 20
    module = LRPAwareGatedAttention(in_channels=in_channels, n_lr=n_lr)

    x = torch.rand(n_nodes, in_channels)
    src = torch.randint(0, n_nodes, (n_edges,))
    dst = torch.randint(0, n_nodes, (n_edges,))
    edge_index = torch.stack([src, dst], dim=0)
    hill_out = torch.rand(n_edges, n_lr).clamp(1e-4, 1 - 1e-4)
    spatial_dist = torch.rand(n_edges).abs() + 0.1
    L_src = torch.rand(n_edges, n_lr)
    R_dst = torch.rand(n_edges, n_lr)

    edge_logits_lr, edge_probs_lr, edge_scores_global = module(
        x, edge_index, hill_out, spatial_dist, L_src, R_dst
    )

    assert edge_logits_lr.shape == (n_edges, n_lr)
    assert edge_probs_lr.shape == (n_edges, n_lr)
    assert edge_scores_global.shape == (n_edges,)
    assert torch.all(edge_probs_lr >= 0.0) and torch.all(edge_probs_lr <= 1.0)


def test_graph_diffusion_smoother_preserves_shape():
    n_nodes = 8
    n_edges = 20
    smoother = GraphDiffusionSmoother(n_steps=3, alpha=0.2, kernel_scale=0.4)

    # GraphDiffusionSmoother expects a column vector (num_nodes, 1) and the
    # explicit node count, since it scatters edge messages by destination.
    signal = torch.rand(n_nodes, 1)
    src = torch.randint(0, n_nodes, (n_edges,))
    dst = torch.randint(0, n_nodes, (n_edges,))
    edge_index = torch.stack([src, dst], dim=0)

    out = smoother(signal, edge_index, n_nodes)
    assert out.shape == signal.shape
    assert torch.all(torch.isfinite(out))


def test_stochastic_variational_relay_train_vs_eval():
    n_nodes = 10
    n_edges = 16
    module = StochasticVariationalRelay(steps=3, latent_dim=4)

    src = torch.randint(0, n_nodes, (n_edges,))
    dst = torch.randint(0, n_nodes, (n_edges,))
    probs = torch.rand(n_edges).clamp(1e-4, 1 - 1e-4)

    module.train()
    h_train, variational_probs_train, kl_train = module(src, dst, probs, n_nodes)
    assert h_train.shape == (n_nodes, 1)
    assert variational_probs_train.shape == (n_edges,)
    assert torch.isfinite(kl_train)
    assert kl_train >= 0.0

    module.eval()
    with torch.no_grad():
        h_eval, variational_probs_eval, kl_eval = module(src, dst, probs, n_nodes)
    assert h_eval.shape == (n_nodes, 1)
    assert variational_probs_eval.shape == (n_edges,)
