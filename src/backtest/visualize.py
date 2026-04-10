"""
visualize.py
------------
Paper-ready visualizations for Stage 4 backtest results.

Generates:
    1. Cumulative returns plot (all systems)
    2. Drawdown comparison plot
    3. Portfolio weight evolution (UARC system)
    4. Regime overlay on UARC returns
    5. Rolling Sharpe ratio comparison
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional

from src.backtest.backtest import BacktestResult

# Color scheme: clean, publication-friendly
COLORS = {
    "Buy & Hold (Equal Weight)": "#AAAAAA",
    "HMM Hard Label + DQN": "#E07B54",
    "No Regime + IQN": "#5B9BD5",
    "HMM Posterior + DQN": "#70AD47",
    "UARC Full System (Ours)": "#1F3864",
}

# Thicker important lines: baseline + main method
LINEWIDTHS = {
    "Buy & Hold (Equal Weight)": 2.2,
    "HMM Hard Label + DQN": 1.2,
    "No Regime + IQN": 1.2,
    "HMM Posterior + DQN": 1.2,
    "UARC Full System (Ours)": 3.0,
}

REGIME_COLORS = ["#2196F3", "#4CAF50", "#F44336"]   # Bull, Sideways, Bear
REGIME_LABELS = ["Bull", "Sideways", "Bear"]


def plot_cumulative_returns(
    results: List[BacktestResult],
    test_dates: Optional[pd.DatetimeIndex] = None,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Main comparison figure: cumulative returns of all systems.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    for result in results:
        cumret = (1 + result.cumulative_returns) * 100  # Index to 100
        n = min(len(cumret), len(test_dates)) if test_dates is not None else len(cumret)
        x = test_dates[:n] if test_dates is not None else np.arange(n)
        color = COLORS.get(result.name, "#333333")
        lw = LINEWIDTHS.get(result.name, 1.2)
        ls = "--" if result.name == "Buy & Hold (Equal Weight)" else "-"

        ax.plot(
            x,
            cumret[-n:],
            label=result.name,
            color=color,
            linewidth=lw,
            linestyle=ls,
            alpha=0.8,
        )

    ax.axhline(100, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Portfolio Value (Index = 100)", fontsize=11)
    ax.set_title("Cumulative Returns — Test Set (2021–2024)", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    return fig


def plot_drawdowns(
    results: List[BacktestResult],
    test_dates: Optional[pd.DatetimeIndex] = None,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Drawdown comparison across all systems."""
    fig, ax = plt.subplots(figsize=(12, 5))

    dd_for_fill = None
    x_for_fill = None

    for result in results:
        cumret = np.exp(np.cumsum(result.daily_returns))
        peak = np.maximum.accumulate(cumret)
        dd = (cumret - peak) / peak * 100
        n = len(dd)
        x = test_dates[:n] if test_dates is not None else np.arange(n)
        color = COLORS.get(result.name, "#333333")
        lw = LINEWIDTHS.get(result.name, 1.2)

        ax.plot(
            x,
            dd[-len(x):],
            label=result.name,
            color=color,
            linewidth=lw,
            alpha=0.8,
        )

        if dd_for_fill is None:
            dd_for_fill = dd
            x_for_fill = x

    ax.axhline(0, color="black", linewidth=0.8)

    if dd_for_fill is not None and x_for_fill is not None:
        ax.fill_between(
            x_for_fill,
            dd_for_fill,
            0,
            where=(dd_for_fill < 0),
            alpha=0.05,
            color="red",
        )

    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Drawdown (%)", fontsize=11)
    ax.set_title("Portfolio Drawdown — Test Set (2021–2024)", fontsize=13, fontweight="bold")
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    return fig


def plot_weight_evolution(
    uarc_result: BacktestResult,
    test_dates: Optional[pd.DatetimeIndex] = None,
    assets: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Portfolio weight evolution for the UARC system.

    If asset names are not provided, names are auto-generated from the
    number of columns in weights_history.
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    weights = np.asarray(uarc_result.weights_history)
    if weights.ndim != 2:
        raise ValueError("weights_history must be a 2D array of shape (T, n_assets).")

    T, n_assets = weights.shape
    x = test_dates[:T] if test_dates is not None else np.arange(T)

    if assets is None:
        assets = [f"Asset {i + 1}" for i in range(n_assets)]

    if len(assets) != n_assets:
        raise ValueError(
            f"Number of asset labels ({len(assets)}) does not match "
            f"weights_history columns ({n_assets})."
        )

    # Use a colormap that supports many assets
    cmap = plt.get_cmap("tab20")
    asset_colors = [cmap(i % 20) for i in range(n_assets)]

    ax.stackplot(
        x,
        weights.T,
        labels=assets,
        colors=asset_colors,
        alpha=0.8,
    )

    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Portfolio Weight", fontsize=11)
    ax.set_title("UARC Portfolio Weight Evolution — Test Set (2021–2024)", fontsize=13, fontweight="bold")

    # Use multiple columns if many assets
    ncol = 5 if n_assets <= 10 else 4
    ax.legend(loc="upper right", fontsize=8, ncol=ncol, framealpha=0.9)

    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.2, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    return fig


def plot_regime_overlay(
    uarc_result: BacktestResult,
    posteriors: np.ndarray,
    test_dates: Optional[pd.DatetimeIndex] = None,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    UARC cumulative returns with regime posterior overlay.
    """
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )

    T = len(uarc_result.daily_returns)
    cumret = (1 + uarc_result.cumulative_returns) * 100
    x = test_dates[:T] if test_dates is not None else np.arange(T)
    post = posteriors[:T]

    dominant_regime = np.argmax(post, axis=1)
    ax1.plot(
        x,
        cumret,
        color=COLORS["UARC Full System (Ours)"],
        linewidth=2.5,
        label="UARC",
        zorder=3,
    )

    y_min = float(np.min(cumret) * 0.98)
    y_max = float(np.max(cumret) * 1.02)

    for regime_idx, (color, label) in enumerate(zip(REGIME_COLORS, REGIME_LABELS)):
        mask = dominant_regime == regime_idx
        if mask.any():
            ax1.fill_between(
                x,
                y_min,
                y_max,
                where=mask,
                alpha=0.08,
                color=color,
                label=f"{label} regime",
            )

    ax1.set_ylim(y_min, y_max)
    ax1.set_ylabel("Portfolio Value (Index = 100)", fontsize=11)
    ax1.set_title("UARC Returns with Market Regime Overlay", fontsize=13, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3, linewidth=0.5)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    ax2.stackplot(
        x,
        post.T,
        labels=REGIME_LABELS,
        colors=REGIME_COLORS,
        alpha=0.85,
    )
    ax2.set_ylabel("Regime\nPosterior", fontsize=10)
    ax2.set_xlabel("Date", fontsize=11)
    ax2.set_ylim(0, 1)
    ax2.legend(loc="upper right", fontsize=8, ncol=3)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    return fig


def plot_rolling_sharpe(
    results: List[BacktestResult],
    window: int = 63,   # ~1 quarter
    test_dates: Optional[pd.DatetimeIndex] = None,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Rolling Sharpe ratio comparison (63-day window = 1 quarter)."""
    fig, ax = plt.subplots(figsize=(12, 5))

    for result in results:
        ret_series = pd.Series(result.daily_returns)
        rolling_sr = (
            ret_series.rolling(window).mean()
            / ret_series.rolling(window).std()
            * np.sqrt(252)
        )

        n = len(rolling_sr)
        x = test_dates[:n] if test_dates is not None else np.arange(n)
        color = COLORS.get(result.name, "#333333")
        lw = LINEWIDTHS.get(result.name, 1.2)
        ls = "--" if "Buy & Hold" in result.name else "-"

        ax.plot(
            x,
            rolling_sr[-len(x):],
            label=result.name,
            color=color,
            linewidth=lw,
            linestyle=ls,
            alpha=0.8,
        )

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel(f"Rolling Sharpe ({window}-day)", fontsize=11)
    ax.set_title("Rolling Sharpe Ratio Comparison — Test Set (2021–2024)", fontsize=13, fontweight="bold")

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        fontsize=9,
        framealpha=0.9,
        borderaxespad=0,
    )

    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout(rect=[0, 0, 0.8, 1])

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    return fig


def save_results_csv(
    results: List[BacktestResult],
    save_path: Path,
):
    """Save results table as CSV for LaTeX table generation."""
    rows = []
    for r in results:
        rows.append(
            {
                "System": r.name,
                "Total Return": r.total_return,
                "Ann Return": r.annualized_return,
                "Sharpe": r.sharpe_ratio,
                "Volatility": r.annualized_volatility,
                "Max Drawdown": r.max_drawdown,
                "CVaR 5pct": r.cvar_5,
                "Calmar": r.calmar_ratio,
                "Avg Turnover": r.avg_turnover,
            }
        )

    df = pd.DataFrame(rows).set_index("System")
    df.to_csv(save_path)
    return df