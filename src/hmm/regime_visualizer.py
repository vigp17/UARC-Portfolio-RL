"""
regime_visualizer.py
--------------------
Visualization utilities for the BayesianMarketHMM.

Produces the plots you need for your paper:
  1. Regime posterior over time (stacked area chart)
  2. Price overlaid with regime coloring
  3. Emission distribution comparison across regimes
  4. BIC curve for regime count selection
  5. Regime transition heatmap
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
import seaborn as sns
from typing import Optional, List

# Color scheme: Bull=green, Bear=red, Sideways=orange, extras=blue shades
REGIME_COLORS = {
    "Bull":       "#2ca02c",   # green
    "Bear":       "#d62728",   # red
    "Sideways":   "#ff7f0e",   # orange
    "Sideways_0": "#ff7f0e",
    "Sideways_1": "#9467bd",
    "Unknown":    "#7f7f7f",   # gray fallback
}

def _get_color(label: str) -> str:
    return REGIME_COLORS.get(label, REGIME_COLORS["Unknown"])


def plot_regime_posterior(
    posteriors: np.ndarray,
    dates: pd.DatetimeIndex,
    regime_labels: dict,
    title: str = "Regime Posterior Over Time",
    save_path: Optional[str] = None,
    figsize: tuple = (14, 4),
) -> plt.Figure:
    """
    Stacked area chart of P(z_t | x_{1:t}) over time.

    This is Figure 1 in your paper — shows the model's uncertainty
    about regime at each timestep. Wide bands = high uncertainty.

    Parameters
    ----------
    posteriors : np.ndarray
        Shape (T, K).
    dates : pd.DatetimeIndex
        Length T.
    regime_labels : dict
        {state_int: "Bull"/"Bear"/"Sideways"} from hmm.regime_labels.
    """
    fig, ax = plt.subplots(figsize=figsize)

    K = posteriors.shape[1]
    colors = [_get_color(regime_labels.get(k, "Unknown")) for k in range(K)]
    labels = [regime_labels.get(k, f"State_{k}") for k in range(K)]

    ax.stackplot(dates, posteriors.T, labels=labels, colors=colors, alpha=0.8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Posterior Probability", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.7)
    ax.set_xlim(dates[0], dates[-1])

    # Shade major financial crises for context
    crises = [
        ("2007-10-01", "2009-03-31", "GFC"),
        ("2020-02-20", "2020-03-23", "COVID"),
        ("2022-01-01", "2022-10-15", "Rate Hikes"),
    ]
    for start, end, label in crises:
        try:
            ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                       alpha=0.08, color="black", label=f"_{label}")
            # Add label at top
            mid = pd.Timestamp(start) + (pd.Timestamp(end) - pd.Timestamp(start)) / 2
            ax.text(mid, 0.97, label, ha="center", va="top",
                    fontsize=7, color="gray", style="italic")
        except Exception:
            pass

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_price_with_regimes(
    prices: pd.DataFrame,
    viterbi_states: np.ndarray,
    regime_labels: dict,
    asset: str = "SPY",
    title: str = "Market Regimes on Price History",
    save_path: Optional[str] = None,
    figsize: tuple = (14, 5),
) -> plt.Figure:
    """
    Plots asset price with background shading by regime (Viterbi).

    NOTE: Uses Viterbi (not posterior) for visualization only —
    this is the post-hoc analysis plot for your paper figures.
    Never use Viterbi states in the actual trading loop.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot price
    price_series = prices[asset] if asset in prices.columns else prices.iloc[:, 0]

    # Align dates with viterbi states
    dates = price_series.index[-len(viterbi_states):]
    p     = price_series.loc[dates]

    ax.plot(dates, p, color="black", linewidth=0.8, zorder=3)

    # Shade regime regions
    prev_state = viterbi_states[0]
    start_idx  = 0
    for t in range(1, len(viterbi_states)):
        if viterbi_states[t] != prev_state or t == len(viterbi_states) - 1:
            label = regime_labels.get(prev_state, f"State_{prev_state}")
            color = _get_color(label)
            ax.axvspan(dates[start_idx], dates[min(t, len(dates)-1)],
                       alpha=0.25, color=color, zorder=1)
            prev_state = viterbi_states[t]
            start_idx  = t

    # Legend
    legend_patches = [
        mpatches.Patch(color=_get_color(v), alpha=0.5, label=v)
        for v in set(regime_labels.values())
    ]
    ax.legend(handles=legend_patches, loc="upper left", fontsize=9)
    ax.set_ylabel(f"{asset} Adjusted Close", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlim(dates[0], dates[-1])

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_emission_distributions(
    hmm_model,
    feature_names: List[str],
    n_features_to_plot: int = 4,
    save_path: Optional[str] = None,
    figsize: tuple = (14, 8),
) -> plt.Figure:
    """
    Compares emission distributions (mean ± std) across regimes.

    Validates that regimes are economically distinct — a key sanity
    check before trusting the HMM posteriors.
    """
    K = hmm_model.n_regimes
    means_scaled = hmm_model.model.means_
    means_orig   = hmm_model.scaler.inverse_transform(means_scaled)

    if hmm_model.covariance_type == "diag":
        stds_scaled = np.sqrt(hmm_model.model.covars_)
        # Approximate unscaling for stds (multiply by scaler scale_)
        stds_orig = stds_scaled * hmm_model.scaler.scale_
    else:
        stds_orig = np.ones_like(means_orig) * 0.01  # fallback

    n_plot = min(n_features_to_plot, len(feature_names))
    fig, axes = plt.subplots(1, n_plot, figsize=figsize)
    if n_plot == 1:
        axes = [axes]

    for fi, ax in enumerate(axes):
        if fi >= len(feature_names):
            ax.set_visible(False)
            continue

        feat_name = feature_names[fi]
        x         = np.linspace(
            means_orig[:, fi].min() - 3 * stds_orig[:, fi].max(),
            means_orig[:, fi].max() + 3 * stds_orig[:, fi].max(),
            300
        )

        for k in range(K):
            label = hmm_model.regime_labels.get(k, f"State_{k}")
            color = _get_color(label)
            mu    = float(np.ravel(means_orig[k])[fi])
            sigma = float(np.ravel(stds_orig[k])[fi])
            y     = _gaussian_pdf(x, mu, sigma)
            ax.plot(x, y, color=color, linewidth=2, label=label)
            ax.fill_between(x, y, alpha=0.15, color=color)
            ax.axvline(mu, color=color, linestyle="--", linewidth=1, alpha=0.6)

        ax.set_title(feat_name.replace("_", " ").title(), fontsize=9)
        ax.set_xlabel("Value", fontsize=8)
        if fi == 0:
            ax.set_ylabel("Density", fontsize=8)
            ax.legend(fontsize=8)
        ax.tick_params(labelsize=7)

    fig.suptitle("Regime Emission Distributions", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_bic_curve(
    bic_scores: dict,
    best_k: int,
    save_path: Optional[str] = None,
    figsize: tuple = (6, 4),
) -> plt.Figure:
    """
    Plots BIC vs number of regimes K.
    Use this to justify your choice of K=3 in the paper.
    """
    fig, ax = plt.subplots(figsize=figsize)

    ks   = list(bic_scores.keys())
    bics = [bic_scores[k] for k in ks]

    ax.plot(ks, bics, "o-", color=REGIME_COLORS["Bull"], linewidth=2, markersize=7)
    ax.axvline(best_k, color=REGIME_COLORS["Bear"], linestyle="--",
               label=f"Selected K={best_k}", linewidth=1.5)
    ax.scatter([best_k], [bic_scores[best_k]], color=REGIME_COLORS["Bear"],
               zorder=5, s=100)

    ax.set_xlabel("Number of Regimes (K)", fontsize=11)
    ax.set_ylabel("BIC Score (lower = better)", fontsize=11)
    ax.set_title("Regime Count Selection via BIC", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xticks(ks)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_transition_matrix(
    hmm_model,
    save_path: Optional[str] = None,
    figsize: tuple = (5, 4),
) -> plt.Figure:
    """
    Heatmap of the HMM transition matrix A.
    High diagonal = regimes are persistent (good for financial markets).
    """
    A      = hmm_model.model.transmat_
    labels = [hmm_model.regime_labels.get(k, f"State_{k}") for k in range(hmm_model.n_regimes)]

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        A, annot=True, fmt=".3f", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        ax=ax, cbar_kws={"shrink": 0.8},
        linewidths=0.5, linecolor="white",
        vmin=0, vmax=1
    )
    ax.set_xlabel("Next Regime", fontsize=10)
    ax.set_ylabel("Current Regime", fontsize=10)
    ax.set_title("Regime Transition Matrix", fontsize=12, fontweight="bold")
    ax.tick_params(labelsize=9)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_posterior_uncertainty(
    posteriors: np.ndarray,
    dates: pd.DatetimeIndex,
    prices: pd.DataFrame,
    asset: str = "SPY",
    save_path: Optional[str] = None,
    figsize: tuple = (14, 6),
) -> plt.Figure:
    """
    Two-panel plot:
    Top: asset price colored by most likely regime
    Bottom: regime entropy (uncertainty) over time

    Entropy = -sum_k p_k * log(p_k). High entropy = uncertain regime.
    This plot demonstrates the key advantage of your approach over hard labels.
    """
    entropy = -np.sum(posteriors * np.log(posteriors + 1e-10), axis=1)
    max_entropy = np.log(posteriors.shape[1])  # Maximum possible entropy

    price_series = prices[asset] if asset in prices.columns else prices.iloc[:, 0]
    price_dates  = price_series.index[-len(posteriors):]
    p            = price_series.loc[price_dates]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True,
                                    gridspec_kw={"height_ratios": [2.5, 1]})

    # Top: price
    ax1.plot(price_dates, p, color="black", linewidth=0.8)
    ax1.set_ylabel(f"{asset} Price", fontsize=10)
    ax1.set_title("Price History and Regime Uncertainty", fontsize=12, fontweight="bold")

    # Bottom: entropy
    ax2.fill_between(price_dates, entropy / max_entropy, alpha=0.6,
                     color="#9467bd", label="Regime Entropy (normalized)")
    ax2.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6,
                label="50% max entropy")
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Uncertainty", fontsize=10)
    ax2.legend(fontsize=8, loc="upper right")
    ax2.set_xlabel("Date", fontsize=10)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def _gaussian_pdf(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


# ── Convenience: generate all plots at once ────────────────────────────────────

def generate_all_plots(
    hmm_model,
    posteriors: np.ndarray,
    viterbi_states: np.ndarray,
    prices: pd.DataFrame,
    feature_names: list,
    bic_scores: dict,
    output_dir: str = "outputs/figures",
):
    """
    Generates and saves all regime visualization plots.
    Call this after fitting the HMM to produce paper-ready figures.
    """
    from pathlib import Path
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    dates = prices.index[-len(posteriors):]

    plot_regime_posterior(posteriors, dates, hmm_model.regime_labels,
        save_path=f"{output_dir}/01_regime_posterior.png")

    plot_price_with_regimes(prices, viterbi_states, hmm_model.regime_labels,
        save_path=f"{output_dir}/02_price_with_regimes.png")

    plot_emission_distributions(hmm_model, feature_names,
        save_path=f"{output_dir}/03_emission_distributions.png")

    best_k = min(bic_scores, key=bic_scores.get)
    plot_bic_curve(bic_scores, best_k,
        save_path=f"{output_dir}/04_bic_curve.png")

    plot_transition_matrix(hmm_model,
        save_path=f"{output_dir}/05_transition_matrix.png")

    plot_posterior_uncertainty(posteriors, dates, prices,
        save_path=f"{output_dir}/06_posterior_uncertainty.png")

    print(f"All plots saved to {output_dir}/")
