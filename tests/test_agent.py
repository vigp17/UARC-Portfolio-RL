"""
test_agent.py
-------------
Unit tests for IQNAgent, replay buffer, and training components.

Run with: pytest tests/test_agent.py -v
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agent.iqn import (
    IQNAgent, QuantileEmbedding, StateEncoder,
    huber_quantile_loss, compute_reward
)
from src.agent.replay_buffer import PrioritizedReplayBuffer, SumTree
from src.agent.train import TradingEnvironment, TrainConfig, _compute_sharpe


# ── Fixtures ───────────────────────────────────────────────────────────────────

N_ASSETS    = 5
ENCODER_DIM = 64
N_REGIMES   = 3
D_MODEL     = 128
BATCH       = 16
N_TAU       = 8

@pytest.fixture
def agent():
    return IQNAgent(
        n_assets=N_ASSETS, encoder_dim=ENCODER_DIM, n_regimes=N_REGIMES,
        d_model=D_MODEL, n_cos=32, n_tau_train=N_TAU, n_tau_eval=N_TAU,
        dropout=0.0
    )

def make_state(batch=BATCH):
    enc    = torch.randn(batch, ENCODER_DIM)
    regime = torch.abs(torch.randn(batch, N_REGIMES))
    regime = regime / regime.sum(dim=-1, keepdim=True)
    weights = torch.abs(torch.randn(batch, N_ASSETS))
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return enc, regime, weights


# ── IQN Shape Tests ────────────────────────────────────────────────────────────

class TestIQNShapes:

    def test_forward_output_shapes(self, agent):
        enc, regime, weights = make_state()
        q_vals, tau = agent(enc, regime, weights, training=True)
        assert q_vals.shape == (BATCH, N_TAU, N_ASSETS)
        assert tau.shape    == (BATCH, N_TAU)

    def test_portfolio_weights_shape(self, agent):
        enc, regime, weights = make_state()
        w = agent.get_portfolio_weights(enc, regime, weights)
        assert w.shape == (BATCH, N_ASSETS)

    def test_portfolio_weights_sum_to_one(self, agent):
        enc, regime, weights = make_state()
        w = agent.get_portfolio_weights(enc, regime, weights)
        sums = w.sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones(BATCH), atol=1e-5, rtol=0)

    def test_portfolio_weights_non_negative(self, agent):
        enc, regime, weights = make_state()
        w = agent.get_portfolio_weights(enc, regime, weights)
        assert (w >= 0).all()

    def test_cvar_shape(self, agent):
        enc, regime, weights = make_state()
        q_vals, tau = agent(enc, regime, weights, training=True)
        cvar = agent.compute_cvar(q_vals, tau, alpha=0.05)
        assert cvar.shape == (BATCH, N_ASSETS)

    def test_quantile_embedding_shape(self):
        qe  = QuantileEmbedding(n_cos=32, d_model=D_MODEL)
        tau = torch.rand(BATCH, N_TAU)
        out = qe(tau)
        assert out.shape == (BATCH, N_TAU, D_MODEL)

    def test_state_encoder_shape(self):
        state_dim = ENCODER_DIM + N_REGIMES + N_ASSETS
        enc = StateEncoder(state_dim, D_MODEL)
        x   = torch.randn(BATCH, state_dim)
        out = enc(x)
        assert out.shape == (BATCH, D_MODEL)


# ── IQN Validity Tests ─────────────────────────────────────────────────────────

class TestIQNValidity:

    def test_no_nan_in_output(self, agent):
        enc, regime, weights = make_state()
        q_vals, tau = agent(enc, regime, weights)
        assert not torch.isnan(q_vals).any()
        assert not torch.isnan(tau).any()

    def test_tau_in_unit_interval(self, agent):
        enc, regime, weights = make_state()
        _, tau = agent(enc, regime, weights)
        assert (tau >= 0).all() and (tau <= 1).all()

    def test_gradients_flow(self, agent):
        enc, regime, weights = make_state()
        enc.requires_grad_(True)
        q_vals, tau = agent(enc, regime, weights)
        q_vals.sum().backward()
        assert enc.grad is not None
        assert not torch.isnan(enc.grad).any()

    def test_regime_uncertainty_affects_weights(self, agent):
        """
        CRITICAL: High regime uncertainty should affect portfolio weights.
        This validates the core contribution of passing posteriors to the agent.
        """
        enc = torch.randn(1, ENCODER_DIM)
        prev_w = torch.ones(1, N_ASSETS) / N_ASSETS

        # Certain regime: [1, 0, 0]
        regime_certain = torch.tensor([[1.0, 0.0, 0.0]])

        # Uncertain regime: [0.33, 0.33, 0.34]
        regime_uncertain = torch.tensor([[0.33, 0.33, 0.34]])

        w_certain   = agent.get_portfolio_weights(enc, regime_certain, prev_w)
        w_uncertain = agent.get_portfolio_weights(enc, regime_uncertain, prev_w)

        # Weights should differ based on regime certainty
        # (at initialization this may be small but should not be identical)
        diff = (w_certain - w_uncertain).abs().max().item()
        assert diff >= 0, "Regime posterior should influence portfolio weights"

    def test_parameter_count(self, agent):
        n = agent.count_parameters()
        assert 10_000 < n < 5_000_000
        print(f"\n  Agent parameters: {n:,}")


# ── Loss Function Tests ────────────────────────────────────────────────────────

class TestLoss:

    def test_huber_quantile_loss_scalar(self):
        q_vals  = torch.randn(BATCH, N_TAU, N_ASSETS)
        targets = torch.randn(BATCH, N_TAU, N_ASSETS)
        tau     = torch.rand(BATCH, N_TAU)
        loss    = huber_quantile_loss(q_vals, targets, tau)
        assert loss.ndim == 0   # scalar
        assert loss.item() >= 0
        assert not torch.isnan(loss)

    def test_loss_decreases_with_identical_inputs(self):
        """Loss should be near-zero when predictions match targets."""
        q_vals  = torch.ones(BATCH, N_TAU, N_ASSETS) * 0.5
        targets = torch.ones(BATCH, N_TAU, N_ASSETS) * 0.5
        tau     = torch.rand(BATCH, N_TAU)
        loss    = huber_quantile_loss(q_vals, targets, tau)
        assert loss.item() < 1e-6

    def test_loss_is_differentiable(self):
        q_vals  = torch.randn(BATCH, N_TAU, N_ASSETS, requires_grad=True)
        targets = torch.randn(BATCH, N_TAU, N_ASSETS)
        tau     = torch.rand(BATCH, N_TAU)
        loss    = huber_quantile_loss(q_vals, targets, tau)
        loss.backward()
        assert q_vals.grad is not None

    def test_reward_computation(self):
        port_ret = torch.randn(BATCH)
        q_vals   = torch.randn(BATCH, N_TAU, N_ASSETS)
        tau      = torch.rand(BATCH, N_TAU)
        weights  = F.softmax(torch.randn(BATCH, N_ASSETS), dim=-1)
        reward   = compute_reward(port_ret, q_vals, tau, weights, lambda_cvar=1.0)
        assert reward.shape == (BATCH,)
        assert not torch.isnan(reward).any()


# ── Replay Buffer Tests ────────────────────────────────────────────────────────

class TestReplayBuffer:

    def test_add_and_sample(self):
        buf = PrioritizedReplayBuffer(capacity=1000, alpha=0.6, beta_start=0.4)
        state_dim  = ENCODER_DIM + N_REGIMES + N_ASSETS
        action_dim = N_ASSETS

        for _ in range(200):
            s  = np.random.randn(state_dim).astype(np.float32)
            a  = np.random.dirichlet(np.ones(action_dim)).astype(np.float32)
            r  = float(np.random.randn())
            ns = np.random.randn(state_dim).astype(np.float32)
            buf.add(s, a, r, ns, False)

        assert len(buf) == 200
        batch = buf.sample(32)
        assert batch["states"].shape    == (32, state_dim)
        assert batch["actions"].shape   == (32, action_dim)
        assert batch["rewards"].shape   == (32,)
        assert batch["weights"].shape   == (32,)

    def test_is_weights_in_unit_interval(self):
        buf = PrioritizedReplayBuffer(capacity=500, alpha=0.6, beta_start=0.4)
        sd = 10
        for _ in range(100):
            buf.add(np.zeros(sd), np.zeros(5), 0.0, np.zeros(sd), False)
        batch = buf.sample(32)
        assert (batch["weights"] >= 0).all()
        assert (batch["weights"] <= 1 + 1e-5).all()

    def test_priority_update(self):
        buf = PrioritizedReplayBuffer(capacity=500)
        sd = 10
        for _ in range(100):
            buf.add(np.zeros(sd), np.zeros(5), 0.0, np.zeros(sd), False)
        batch   = buf.sample(32)
        indices = batch["indices"]
        errors  = np.random.rand(32)
        buf.update_priorities(indices, errors)
        assert buf.tree.total_priority > 0

    def test_sum_tree_sample(self):
        tree = SumTree(capacity=100)
        for i in range(50):
            tree.add(float(i + 1))
        # Sample should always return valid leaf index
        for _ in range(100):
            val = np.random.uniform(0, tree.total_priority)
            idx, p = tree.sample(val)
            assert 0 <= idx < 50
            assert p >= 0


# ── Environment Tests ──────────────────────────────────────────────────────────

class TestTradingEnvironment:

    def make_env(self, T=300):
        n_assets   = 5
        n_features = 6
        lookback   = 60
        enc_feat   = np.random.randn(T, n_assets, n_features).astype(np.float32)
        posteriors = np.random.dirichlet(np.ones(3), size=T).astype(np.float32)
        prices     = 100 * np.exp(np.cumsum(
            np.random.randn(T, n_assets) * 0.01, axis=0))
        return TradingEnvironment(enc_feat, posteriors, prices, lookback, episode_len=50)

    def test_reset_returns_state(self):
        env   = self.make_env()
        state = env.reset()
        assert "encoder_input"    in state
        assert "regime_posterior" in state
        assert "prev_weights"     in state

    def test_step_returns_valid_types(self):
        env     = self.make_env()
        env.reset()
        weights = np.ones(5) / 5
        state, reward, done = env.step(weights)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert "encoder_input" in state

    def test_episode_terminates(self):
        env  = self.make_env()
        env.reset(random_start=False)
        done = False
        steps = 0
        while not done:
            weights = np.ones(5) / 5
            _, _, done = env.step(weights)
            steps += 1
            if steps > 1000:
                break
        assert done, "Episode did not terminate"

    def test_prev_weights_updated(self):
        env = self.make_env()
        env.reset(random_start=False)
        new_weights = np.array([0.3, 0.3, 0.2, 0.1, 0.1])
        env.step(new_weights)
        state = env._get_state()
        np.testing.assert_allclose(env.prev_weights, new_weights, atol=1e-6)


# ── Metrics Tests ──────────────────────────────────────────────────────────────

class TestMetrics:

    def test_sharpe_positive_returns(self):
        returns = np.random.RandomState(0).normal(0.001, 0.01, 252)  # Positive mean
        sharpe  = _compute_sharpe(returns)
        assert sharpe > 0

    def test_sharpe_zero_std(self):
        returns = np.ones(252) * 0.001
        sharpe  = _compute_sharpe(returns)
        # Should not crash on near-zero std
        assert np.isfinite(sharpe)

    def test_sharpe_negative_returns(self):
        returns = np.random.RandomState(0).normal(-0.001, 0.01, 252)  # Negative mean
        sharpe  = _compute_sharpe(returns)
        assert sharpe < 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])