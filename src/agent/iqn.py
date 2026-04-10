"""
iqn.py
------
Implicit Quantile Network (IQN) agent for portfolio management.

The agent takes:
    - iTransformer embedding    : (batch, 64)
    - HMM regime posterior      : (batch, K)
    - Previous portfolio weights: (batch, N)

Regime conditioning uses FiLM (Feature-wise Linear Modulation):
    enc_conditioned = gamma(regime) * enc + beta(regime)

This ensures the regime posterior actively modulates the encoder
representation rather than being drowned out by concatenation.

Reference: Dabney et al. (2018) IQN — ICML 2018
           Perez et al. (2018) FiLM — AAAI 2018
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation.
    Conditions encoder output on regime posterior via scale+shift.

    gamma(regime) * encoder_output + beta(regime)
    """

    def __init__(self, n_regimes: int, encoder_dim: int):
        super().__init__()
        self.gamma = nn.Linear(n_regimes, encoder_dim)
        self.beta  = nn.Linear(n_regimes, encoder_dim)
        nn.init.ones_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, enc: torch.Tensor, regime: torch.Tensor) -> torch.Tensor:
        return self.gamma(regime) * enc + self.beta(regime)


class QuantileEmbedding(nn.Module):
    def __init__(self, n_cos: int = 64, d_model: int = 256):
        super().__init__()
        self.n_cos   = n_cos
        self.d_model = d_model
        self.linear  = nn.Linear(n_cos, d_model)
        i_values = torch.arange(1, n_cos + 1, dtype=torch.float32) * math.pi
        self.register_buffer("i_values", i_values)

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        tau_expanded = tau.unsqueeze(-1)
        cos_features = torch.cos(tau_expanded * self.i_values)
        return F.relu(self.linear(cos_features))


class StateEncoder(nn.Module):
    def __init__(self, state_dim: int, d_model: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class IQNAgent(nn.Module):
    """
    IQN agent with FiLM regime conditioning.

    Architecture:
        1. FiLM: regime posterior modulates encoder output
           enc_cond = gamma(regime) * enc + beta(regime)
        2. StateEncoder: [enc_cond | prev_weights] -> d_model
        3. QuantileEmbedding: tau -> (batch, n_tau, d_model)
        4. Elementwise product + output head -> (batch, n_tau, n_assets)
    """

    def __init__(
        self,
        n_assets:    int   = 5,
        encoder_dim: int   = 64,
        n_regimes:   int   = 3,
        d_model:     int   = 256,
        n_cos:       int   = 64,
        n_tau_train: int   = 32,
        n_tau_eval:  int   = 64,
        dropout:     float = 0.1,
    ):
        super().__init__()

        self.n_assets    = n_assets
        self.encoder_dim = encoder_dim
        self.n_regimes   = n_regimes
        self.d_model     = d_model
        self.n_tau_train = n_tau_train
        self.n_tau_eval  = n_tau_eval

        # FiLM: regime conditions the encoder output
        self.film = FiLM(n_regimes, encoder_dim)

        # State = [film_conditioned_enc | prev_weights]
        self.state_dim = encoder_dim + n_assets

        self.state_encoder      = StateEncoder(self.state_dim, d_model)
        self.quantile_embedding = QuantileEmbedding(n_cos, d_model)
        self.dropout            = nn.Dropout(dropout)

        self.output_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_assets),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Re-init FiLM to identity (gamma=1, beta=0) after xavier pass
        nn.init.ones_(self.film.gamma.weight)
        nn.init.zeros_(self.film.gamma.bias)
        nn.init.zeros_(self.film.beta.weight)
        nn.init.zeros_(self.film.beta.bias)

    def forward(
        self,
        encoder_output:   torch.Tensor,
        regime_posterior: torch.Tensor,
        prev_weights:     torch.Tensor,
        tau:              Optional[torch.Tensor] = None,
        training:         bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = encoder_output.shape[0]
        n_tau = self.n_tau_train if training else self.n_tau_eval

        # FiLM conditioning: regime modulates encoder output
        enc_cond = self.film(encoder_output, regime_posterior)  # (batch, encoder_dim)

        # State: conditioned encoder + prev weights (regime already baked in)
        state = torch.cat([enc_cond, prev_weights], dim=-1)     # (batch, state_dim)

        state_emb = self.state_encoder(state)                   # (batch, d_model)

        if tau is None:
            tau = torch.rand(batch, n_tau, device=encoder_output.device)

        tau_emb  = self.quantile_embedding(tau)                 # (batch, n_tau, d_model)
        combined = state_emb.unsqueeze(1) * tau_emb             # (batch, n_tau, d_model)
        combined = self.dropout(combined)

        quantile_values = self.output_head(combined)            # (batch, n_tau, n_assets)

        return quantile_values, tau

    def get_portfolio_weights(
        self,
        encoder_output:   torch.Tensor,
        regime_posterior: torch.Tensor,
        prev_weights:     torch.Tensor,
        risk_aversion:    float = 0.0,
    ) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            quantile_values, tau = self.forward(
                encoder_output, regime_posterior, prev_weights,
                training=False
            )
            expected_return = quantile_values.mean(dim=1)

            if risk_aversion > 0:
                cvar = self.compute_cvar(quantile_values, tau, alpha=0.05)
                score    = expected_return - risk_aversion * cvar.abs()
            else:
                score = expected_return

            return F.softmax(score, dim=-1)

    def compute_cvar(
        self,
        quantile_values: torch.Tensor,
        tau: torch.Tensor,
        alpha: float = 0.05,
    ) -> torch.Tensor:
        """
        Estimate per-asset CVaR from sampled quantiles.

        Quantiles are first ordered by their corresponding tau values so we can
        average the lower alpha-tail consistently across each batch element.
        """
        n_assets = quantile_values.shape[-1]
        sorted_tau, order = tau.sort(dim=1)
        gather_idx = order.unsqueeze(-1).expand(-1, -1, n_assets)
        sorted_q = quantile_values.gather(1, gather_idx)

        tail_mask = (sorted_tau <= alpha).unsqueeze(-1)
        no_tail = tail_mask.sum(dim=1, keepdim=True) == 0
        tail_mask[:, :1, :] |= no_tail

        tail_count = tail_mask.sum(dim=1).clamp(min=1)
        tail_sum = (sorted_q * tail_mask).sum(dim=1)
        return tail_sum / tail_count

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Loss and reward
# ---------------------------------------------------------------------------

def huber_quantile_loss(
    quantile_values: torch.Tensor,
    target_values:   torch.Tensor,
    tau:             torch.Tensor,
    kappa:           float = 1.0,
) -> torch.Tensor:
    batch, n_tau, n_assets = quantile_values.shape

    preds   = quantile_values.unsqueeze(2)
    targets = target_values.unsqueeze(1)
    u       = targets - preds

    huber = torch.where(
        u.abs() <= kappa,
        0.5 * u.pow(2),
        kappa * (u.abs() - 0.5 * kappa)
    )

    tau_expanded = tau.unsqueeze(2).unsqueeze(3)
    indicator    = (u.detach() < 0).float()
    weight       = (tau_expanded - indicator).abs()

    return (weight * huber).mean(dim=2).mean()


def compute_reward(
    portfolio_return: torch.Tensor,
    quantile_values:  torch.Tensor,
    tau:              torch.Tensor,
    weights:          torch.Tensor,
    lambda_cvar:      float = 0.1,
    alpha_cvar:       float = 0.05,
    transaction_cost: float = 0.001,
    prev_weights:     Optional[torch.Tensor] = None,
) -> torch.Tensor:
    reward = portfolio_return.clone()

    mask       = (tau <= alpha_cvar).float()
    mask_exp   = mask.unsqueeze(-1)
    tail_vals  = (quantile_values * mask_exp).sum(dim=1)
    tail_count = mask_exp.sum(dim=1).clamp(min=1)
    asset_cvar = tail_vals / tail_count
    port_cvar  = (weights * asset_cvar).sum(dim=-1)

    reward = reward - lambda_cvar * port_cvar.abs()

    if prev_weights is not None:
        turnover = (weights - prev_weights).abs().sum(dim=-1)
        reward   = reward - transaction_cost * turnover

    return reward
