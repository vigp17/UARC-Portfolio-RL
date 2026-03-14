"""
backtest.py
-----------
Backtesting engine for the UARC ablation study.

All five systems are evaluated on the holdout test set (2021-2024).
The four learned systems each use their own trained agent checkpoint.
No model parameters are updated during backtesting.
"""

import logging
import numpy as np
import pandas as pd
import torch
from typing import List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    name:            str
    daily_returns:   np.ndarray
    weights_history: np.ndarray  # (T, n_assets)

    @property
    def cumulative_returns(self) -> np.ndarray:
        return np.exp(np.cumsum(self.daily_returns)) - 1

    @property
    def total_return(self) -> float:
        return float(np.exp(self.daily_returns.sum()) - 1)

    @property
    def annualized_return(self) -> float:
        T = len(self.daily_returns)
        return float(np.exp(self.daily_returns.sum() * 252 / T) - 1)

    @property
    def sharpe_ratio(self) -> float:
        if self.daily_returns.std() < 1e-10:
            return 0.0
        return float(self.daily_returns.mean() / self.daily_returns.std() * np.sqrt(252))

    @property
    def max_drawdown(self) -> float:
        cum  = np.exp(np.cumsum(self.daily_returns))
        peak = np.maximum.accumulate(cum)
        return float(((cum - peak) / peak).min())

    @property
    def cvar_5(self) -> float:
        n = max(1, int(0.05 * len(self.daily_returns)))
        return float(np.sort(self.daily_returns)[:n].mean())

    @property
    def calmar_ratio(self) -> float:
        mdd = abs(self.max_drawdown)
        return float(self.annualized_return / mdd) if mdd > 1e-10 else 0.0

    @property
    def annualized_volatility(self) -> float:
        return float(self.daily_returns.std() * np.sqrt(252))

    @property
    def avg_turnover(self) -> float:
        if len(self.weights_history) < 2:
            return 0.0
        return float(np.abs(np.diff(self.weights_history, axis=0)).sum(axis=1).mean())

    def to_dict(self) -> dict:
        return {
            "System":          self.name,
            "Total Return":    f"{self.total_return:+.1%}",
            "Ann. Return":     f"{self.annualized_return:+.1%}",
            "Sharpe Ratio":    f"{self.sharpe_ratio:.3f}",
            "Ann. Volatility": f"{self.annualized_volatility:.1%}",
            "Max Drawdown":    f"{self.max_drawdown:.1%}",
            "CVaR (5%)":       f"{self.cvar_5:.4f}",
            "Calmar Ratio":    f"{self.calmar_ratio:.3f}",
            "Avg Turnover":    f"{self.avg_turnover:.3f}",
        }


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _run_agent_backtest(
    name:        str,
    agent,
    embeddings:  np.ndarray,   # (T, encoder_dim) — pre-computed, frozen encoder
    posteriors:  np.ndarray,   # (T, K)
    prices:      np.ndarray,   # (T, n_assets)
    device:      torch.device,
    regime_mode: str,          # "uniform" | "hard" | "posterior"
    lookback:    int = 60,
) -> BacktestResult:
    """
    Generic agent backtest loop used by all four learned systems.

    Args:
        embeddings:  Pre-computed encoder outputs for each test timestep.
        posteriors:  HMM posteriors — transformed per regime_mode.
        regime_mode: How to present the posterior to the agent.
    """
    T        = len(embeddings)
    n_assets = prices.shape[1]
    K        = posteriors.shape[1]

    log_rets = np.zeros_like(prices)
    log_rets[1:] = np.log(prices[1:] / (prices[:-1] + 1e-10))

    returns  = []
    wt_hist  = []
    prev_w   = np.ones(n_assets, dtype=np.float32) / n_assets

    agent.eval()
    with torch.no_grad():
        for t in range(lookback, T - 1):
            enc = torch.from_numpy(embeddings[t]).unsqueeze(0).to(device)
            reg = torch.from_numpy(
                _transform_regime(posteriors[t], regime_mode, K)
            ).unsqueeze(0).to(device)
            wts = torch.from_numpy(prev_w).unsqueeze(0).to(device)

            w      = agent.get_portfolio_weights(enc, reg, wts, risk_aversion=0.0).squeeze(0).cpu().numpy()
            r      = float(np.dot(w, log_rets[t + 1]))
            prev_w = w

            returns.append(r)
            wt_hist.append(w)

    return BacktestResult(
        name=name,
        daily_returns=np.array(returns),
        weights_history=np.array(wt_hist),
    )


def _transform_regime(posterior: np.ndarray, mode: str, K: int) -> np.ndarray:
    if mode == "uniform":
        return np.full(K, 1.0 / K, dtype=np.float32)
    elif mode == "hard":
        v = np.zeros(K, dtype=np.float32)
        v[int(np.argmax(posterior))] = 1.0
        return v
    elif mode == "posterior":
        return posterior.astype(np.float32)
    raise ValueError(f"Unknown regime_mode: {mode}")


# ---------------------------------------------------------------------------
# Public backtest functions
# ---------------------------------------------------------------------------

def backtest_buy_and_hold(
    prices:   np.ndarray,
    n_assets: int = 5,
) -> BacktestResult:
    T        = len(prices)
    weights  = np.ones((T, n_assets)) / n_assets
    log_rets = np.zeros_like(prices)
    log_rets[1:] = np.log(prices[1:] / (prices[:-1] + 1e-10))
    returns  = np.array([np.dot(weights[t-1], log_rets[t]) for t in range(1, T)])
    return BacktestResult(
        name="Buy & Hold (Equal Weight)",
        daily_returns=returns,
        weights_history=weights[1:],
    )


def backtest_hmm_hard_label_dqn(
    encoder,                   # kept for API consistency, not used (embeddings pre-computed)
    agent,
    embeddings:  np.ndarray,
    posteriors:  np.ndarray,
    prices:      np.ndarray,
    device:      torch.device,
    lookback:    int = 60,
    n_features:  int = 6,
    regime_mode: str = "hard",
) -> BacktestResult:
    return _run_agent_backtest(
        "HMM Hard Label + DQN", agent, embeddings, posteriors,
        prices, device, regime_mode="hard",
    )


def backtest_no_regime_iqn(
    encoder,
    agent,
    embeddings:  np.ndarray,
    prices:      np.ndarray,
    device:      torch.device,
    lookback:    int = 60,
    n_features:  int = 6,
    n_regimes:   int = 3,
) -> BacktestResult:
    # Uniform posterior — no regime information
    posteriors = np.full((len(embeddings), n_regimes), 1.0 / n_regimes, dtype=np.float32)
    return _run_agent_backtest(
        "No Regime + IQN", agent, embeddings, posteriors,
        prices, device, regime_mode="uniform",
    )


def backtest_hmm_posterior_dqn(
    encoder,
    agent,
    embeddings:  np.ndarray,
    posteriors:  np.ndarray,
    prices:      np.ndarray,
    device:      torch.device,
    lookback:    int = 60,
    n_features:  int = 6,
    regime_mode: str = "posterior",
) -> BacktestResult:
    return _run_agent_backtest(
        "HMM Posterior + DQN", agent, embeddings, posteriors,
        prices, device, regime_mode="posterior",
    )


def backtest_uarc_full(
    encoder,
    agent,
    embeddings:  np.ndarray,
    posteriors:  np.ndarray,
    prices:      np.ndarray,
    device:      torch.device,
    lookback:    int = 60,
    n_features:  int = 6,
) -> BacktestResult:
    return _run_agent_backtest(
        "UARC Full System (Ours)", agent, embeddings, posteriors,
        prices, device, regime_mode="posterior",
    )


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------

def build_results_table(results: List[BacktestResult]) -> pd.DataFrame:
    return pd.DataFrame([r.to_dict() for r in results]).set_index("System")


def print_results_table(results: List[BacktestResult]):
    df = build_results_table(results)
    print("\n" + "=" * 90)
    print("BACKTEST RESULTS — Test Set (2021-2024)")
    print("=" * 90)
    print(df.to_string())
    print("=" * 90 + "\n")