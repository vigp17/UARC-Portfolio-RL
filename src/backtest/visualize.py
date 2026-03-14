"""
visualize.py
------------
Paper-ready visualizations for Stage 4 backtest results.

Generates:
    1. Cumulative returns plot (all 5 systems)
    2. Drawdown comparison plot
    3. Portfolio weight evolution (UARC system)
    4. Regime overlay on UARC returns
    5. Rolling Sharpe ratio comparison
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import List, Dict, Optional

from src.backtest.backtest import BacktestResult

# Color scheme: clean, publication-friendly
COLORS = {
    "Buy & Hold (Equal Weight)":       "#AAAAAA",
    "HMM Hard Label + DQN":            "#E07B54",
    "No Regime + IQN":                 "#5B9BD5",
    "HMM Posterior + DQN":             "#70AD47",
    "UARC Full System (Ours)":         "#1F3864",
}
LINEWIDTHS = {
    "Buy & Hold (Equal Weight)":       1.2,
    "HMM Hard Label + DQN":            1.2,
    "No Regime + IQN":                 1.2,
    "HMM Posterior + DQN":             1.2,
    "UARC Full System (Ours)":         2.5,
}
REGIME_COLORS = ["#2196F3", "#4CAF50", "#F44336"]   # Bull, Sideways, Bear
REGIME_LABELS = ["Bull", "Sideways", "Bear"]

ASSETS = ["SPY", "QQQ", "TLT", "GLD", "SHY"]


def plot_cumulative_returns(
    results: List[BacktestResult],
    test_dates: Optional[pd.DatetimeIndex] = None,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Main comparison figure: cumulative returns of all 5 systems.
    This is Figure 1 in the paper.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    for result in results:
        cumret = (1 + result.cumulative_returns) * 100   # Index to 100
        # Align to shortest length to handle different lookback offsets
        n      = min(len(cumret), len(test_dates)) if test_dates is not None else len(cumret)
        x      = test_dates[:n] if test_dates is not None else np.arange(n)
        color  = COLORS.get(result.name, "#333333")
        lw     = LINEWIDTHS.get(result.name, 1.2)
        ls     = "--" if result.name == "Buy & Hold (Equal Weight)" else "-"

        ax.plot(x, cumret[-n:], label=result.name, color=color,
                linewidth=lw, linestyle=ls, alpha=0.9)

    ax.axhline(100, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Portfolio Value (Index = 100)", fontsize=11)
    ax.set_title("Cumulative Returns — Test Set (2021–2024)",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
    return fig


def plot_drawdowns(
    results: List[BacktestResult],
    test_dates: Optional[pd.DatetimeIndex] = None,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Drawdown comparison across all systems."""
    fig, ax = plt.subplots(figsize=(12, 5))

    for result in results:
        cumret = np.exp(np.cumsum(result.daily_returns))
        peak   = np.maximum.accumulate(cumret)
        dd     = (cumret - peak) / peak * 100
        n      = len(dd)
        x      = test_dates[:n] if test_dates is not None else np.arange(n)
        color  = COLORS.get(result.name, "#333333")
        lw     = LINEWIDTHS.get(result.name, 1.2)

        ax.plot(x, dd[-len(x):], label=result.name, color=color, linewidth=lw, alpha=0.85)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.fill_between(
        x if test_dates is None else test_dates[:len(dd)],
        dd, 0,
        where=(dd < 0), alpha=0.05, color="red"
    )
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Drawdown (%)", fontsize=11)
    ax.set_title("Portfolio Drawdown — Test Set (2021–2024)",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
    return fig


def plot_weight_evolution(
    uarc_result: BacktestResult,
    test_dates: Optional[pd.DatetimeIndex] = None,
    assets: List[str] = ASSETS,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Portfolio weight evolution for the UARC system.
    Shows how the agent dynamically reallocates across assets.
    This is Figure 2 in the paper.
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    weights = uarc_result.weights_history  # (T, n_assets)
    T       = len(weights)
    x       = test_dates[:T] if test_dates is not None else np.arange(T)

    asset_colors = ["#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD"]

    ax.stackplot(x, weights.T,
                 labels=assets,
                 colors=asset_colors,
                 alpha=0.8)

    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Portfolio Weight", fontsize=11)
    ax.set_title("UARC Portfolio Weight Evolution — Test Set (2021–2024)",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9, ncol=5, framealpha=0.9)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.2, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
    return fig


def plot_regime_overlay(
    uarc_result:  BacktestResult,
    posteriors:   np.ndarray,
    test_dates:   Optional[pd.DatetimeIndex] = None,
    save_path:    Optional[Path] = None,
) -> plt.Figure:
    """
    UARC cumulative returns with regime posterior overlay.
    Shows agent behavior during regime transitions.
    This is Figure 3 in the paper — the most visually compelling.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8),
                                    gridspec_kw={"height_ratios": [3, 1]},
                                    sharex=True)

    T      = len(uarc_result.daily_returns)
    cumret = (1 + uarc_result.cumulative_returns) * 100
    x      = test_dates[:T] if test_dates is not None else np.arange(T)
    post   = posteriors[:T]

    # Top: cumulative returns with regime background shading
    dominant_regime = np.argmax(post, axis=1)
    ax1.plot(x, cumret, color=COLORS["UARC Full System (Ours)"],
             linewidth=2, label="UARC", zorder=3)

    # Shade background by dominant regime
    for regime_idx, (color, label) in enumerate(zip(REGIME_COLORS, REGIME_LABELS)):
        mask = dominant_regime == regime_idx
        if mask.any():
            ax1.fill_between(x, ax1.get_ylim()[0] if ax1.get_ylim()[0] != 0 else 80,
                             cumret.max() * 1.05,
                             where=mask, alpha=0.08, color=color, label=f"{label} regime")

    ax1.set_ylabel("Portfolio Value (Index = 100)", fontsize=11)
    ax1.set_title("UARC Returns with Market Regime Overlay",
                  fontsize=13, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3, linewidth=0.5)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Bottom: stacked regime posterior
    ax2.stackplot(x, post.T,
                  labels=REGIME_LABELS,
                  colors=REGIME_COLORS,
                  alpha=0.85)
    ax2.set_ylabel("Regime\nPosterior", fontsize=10)
    ax2.set_xlabel("Date", fontsize=11)
    ax2.set_ylim(0, 1)
    ax2.legend(loc="upper right", fontsize=8, ncol=3)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
    return fig


def plot_rolling_sharpe(
    results: List[BacktestResult],
    window:  int = 63,   # ~1 quarter
    test_dates: Optional[pd.DatetimeIndex] = None,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Rolling Sharpe ratio comparison (63-day window = 1 quarter)."""
    fig, ax = plt.subplots(figsize=(12, 5))

    for result in results:
        ret_series = pd.Series(result.daily_returns)
        rolling_sr = (ret_series.rolling(window).mean() /
                      ret_series.rolling(window).std() * np.sqrt(252))
        n = len(rolling_sr)
        x = test_dates[:n] if test_dates is not None else np.arange(n)
        color = COLORS.get(result.name, "#333333")
        lw    = LINEWIDTHS.get(result.name, 1.2)
        ax.plot(x, rolling_sr[-len(x):], label=result.name, color=color,
                linewidth=lw, alpha=0.85)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel(f"Rolling Sharpe ({window}-day)", fontsize=11)
    ax.set_title("Rolling Sharpe Ratio Comparison — Test Set (2021–2024)",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
    return fig


def save_results_csv(
    results: List[BacktestResult],
    save_path: Path,
):
    """Save results table as CSV for LaTeX table generation."""
    rows = []
    for r in results:
        rows.append({
            "System":           r.name,
            "Total Return":     r.total_return,
            "Ann Return":       r.annualized_return,
            "Sharpe":           r.sharpe_ratio,
            "Volatility":       r.annualized_volatility,
            "Max Drawdown":     r.max_drawdown,
            "CVaR 5pct":        r.cvar_5,
            "Calmar":           r.calmar_ratio,
            "Avg Turnover":     r.avg_turnover,
        })
    df = pd.DataFrame(rows).set_index("System")
    df.to_csv(save_path)
    return df
