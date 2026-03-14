"""
run_stage3.py  —  Train all 4 UARC ablation agents across 5 seeds.

Outputs:
  outputs/models/agent_{agent_type}_seed{N}_best.pt   — best checkpoint per run
  outputs/stage3_results.csv                          — full results table
  outputs/figures/stage3/training_curves.png          — summary plot

Usage:
  python run_stage3.py                  # all agents, all seeds
  python run_stage3.py --agent uarc     # single agent, all seeds
  python run_stage3.py --seeds 0        # all agents, seed 0 only
"""

import argparse
import logging
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import torch
from pathlib import Path

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("outputs/stage3.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)

from src.agent.train import UARCTrainer, TrainConfig, AgentType

# ── Constants ─────────────────────────────────────────────────────────────────

SEEDS       = [42, 123, 456, 789, 1337]  # use --seeds 42 for quick diagnostic run
AGENT_TYPES = [
    AgentType.NO_REGIME_IQN,
    AgentType.HMM_HARD_DQN,
    AgentType.HMM_POSTERIOR_DQN,
    AgentType.UARC,
]

AGENT_LABELS = {
    AgentType.NO_REGIME_IQN:     "No Regime + IQN",
    AgentType.HMM_HARD_DQN:      "HMM Hard Label + DQN",
    AgentType.HMM_POSTERIOR_DQN: "HMM Posterior + DQN",
    AgentType.UARC:              "UARC Full System (Ours)",
}

PRETRAINED_ENCODER = "outputs/models/encoder_stage2.pt"


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data():
    logger.info("=" * 60)
    logger.info("STAGE 3: Multi-Agent Training (4 agents x 5 seeds)")
    logger.info("=" * 60)

    logger.info("\n[1/4] Loading data...")

    # Load precomputed features and posteriors
    enc_features_train = np.load("outputs/enc_features_train.npy")
    enc_features_val   = np.load("outputs/enc_features_val.npy")
    posteriors_train   = np.load("outputs/posteriors_train.npy")
    posteriors_val     = np.load("outputs/posteriors_val.npy")

    logger.info(f"  Train: enc {enc_features_train.shape} | post {posteriors_train.shape}")
    logger.info(f"  Val:   enc {enc_features_val.shape}   | post {posteriors_val.shape}")

    # Load prices directly from CSV
    prices_csv = "data/raw/prices.csv"
    logger.info(f"  Loading prices from {prices_csv}")
    prices_df  = pd.read_csv(prices_csv, index_col=0, parse_dates=True)
    prices_all = prices_df.values.astype(np.float32)

    # Temporal split
    # Train: 2000-2017, Val: 2018-2020
    # Infer split sizes from posterior shapes
    n_train = len(posteriors_train)
    n_val   = len(posteriors_val)

    prices_train = prices_all[:n_train]
    prices_val   = prices_all[n_train:n_train + n_val]

    # Align enc_features length to prices length
    if len(enc_features_train) > n_train:
        enc_features_train = enc_features_train[-n_train:]
    if len(enc_features_val) > n_val:
        enc_features_val = enc_features_val[-n_val:]

    # Use last 20% of training data as a held-out validation set
    # This avoids overfitting to the COVID crash/recovery in the 2018-2020 val period
    split = int(0.8 * n_train)
    enc_features_val   = enc_features_train[split:]
    posteriors_val     = posteriors_train[split:]
    prices_val         = prices_train[split:]
    enc_features_train = enc_features_train[:split]
    posteriors_train   = posteriors_train[:split]
    prices_train       = prices_train[:split]
    logger.info(f"  Replit train/val: train={len(prices_train)} val={len(prices_val)} days")

    logger.info(f"  Prices train: {prices_train.shape}  val: {prices_val.shape}")

    return (
        enc_features_train, posteriors_train, prices_train,
        enc_features_val,   posteriors_val,   prices_val,
    )


# ── Training loop ─────────────────────────────────────────────────────────────

def train_all(
    data,
    agent_types: list,
    seeds: list,
) -> pd.DataFrame:

    (enc_tr, post_tr, prc_tr,
     enc_vl, post_vl, prc_vl) = data

    records = []
    total   = len(agent_types) * len(seeds)
    done    = 0

    for agent_type in agent_types:
        agent_sharpes = []

        for seed in seeds:
            done += 1
            logger.info(
                f"\n[{done}/{total}] Training: {agent_type.value} | seed={seed}"
            )

            config = TrainConfig.for_agent(agent_type, seed=seed)

            trainer = UARCTrainer(
                config                  = config,
                enc_features_train      = enc_tr,
                posteriors_train        = post_tr,
                prices_train            = prc_tr,
                enc_features_val        = enc_vl,
                posteriors_val          = post_vl,
                prices_val              = prc_vl,
                pretrained_encoder_path = PRETRAINED_ENCODER,
            )

            result = trainer.train()
            records.append(result)
            agent_sharpes.append(result["best_val_sharpe"])

            logger.info(
                f"  -> best val Sharpe: {result['best_val_sharpe']:.4f}"
            )

        mean_s = np.mean(agent_sharpes)
        std_s  = np.std(agent_sharpes)
        logger.info(
            f"\n  [{agent_type.value}] "
            f"Val Sharpe across {len(seeds)} seeds: "
            f"{mean_s:.3f} +/- {std_s:.3f}"
        )

    return pd.DataFrame(records)


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame):
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 3 SUMMARY — Val Sharpe (mean +/- std across 5 seeds)")
    logger.info("=" * 60)

    summary = (
        df.groupby("agent_type")["best_val_sharpe"]
        .agg(["mean", "std", "min", "max"])
        .round(4)
    )

    for idx, row in summary.iterrows():
        label = next(
            (AGENT_LABELS[a] for a in AgentType if a.value == idx),
            idx
        )
        logger.info(
            f"  {label:<35} "
            f"Sharpe: {row['mean']:.3f} +/- {row['std']:.3f}  "
            f"[{row['min']:.3f}, {row['max']:.3f}]"
        )

    logger.info("=" * 60)


# ── Training curves plot ──────────────────────────────────────────────────────

def plot_summary(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = ["#2196F3", "#FF9800", "#9C27B0", "#4CAF50"]
    agent_order = [a.value for a in AGENT_TYPES]

    means, stds, labels = [], [], []
    for i, av in enumerate(agent_order):
        sub    = df[df["agent_type"] == av]["best_val_sharpe"]
        means.append(sub.mean())
        stds.append(sub.std())
        labels.append(
            next(AGENT_LABELS[a] for a in AgentType if a.value == av)
        )

    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=8, color=colors,
                  alpha=0.85, edgecolor="black", linewidth=0.8)

    ax.bar_label(bars, labels=[f"{m:.3f}" for m in means],
                 padding=5, fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("Val Sharpe Ratio (annualized)", fontsize=11)
    ax.set_title(
        "Ablation Study — Validation Sharpe Ratio (mean ± std, 5 seeds)",
        fontsize=12, fontweight="bold"
    )
    ax.set_ylim(0, max(means) * 1.4)
    ax.axhline(y=means[0], color=colors[0], linestyle="--",
               alpha=0.4, label="No Regime baseline")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = Path("outputs/figures/stage3")
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "ablation_val_sharpe.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved ablation plot -> {out / 'ablation_val_sharpe.png'}")


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--agent", type=str, default=None,
        choices=[a.value for a in AgentType],
        help="Train a single agent type (default: all)"
    )
    p.add_argument(
        "--seeds", type=int, nargs="+", default=None,
        help="Seed(s) to run (default: all 5)"
    )
    return p.parse_args()


def main():
    args = parse_args()

    agent_types = (
        [AgentType(args.agent)] if args.agent
        else AGENT_TYPES
    )
    seeds = args.seeds if args.seeds else SEEDS

    logger.info(f"Agents : {[a.value for a in agent_types]}")
    logger.info(f"Seeds  : {seeds}")

    data = load_data()

    logger.info("\n[2/4] Training all agents...")
    df = train_all(data, agent_types, seeds)

    logger.info("\n[3/4] Saving results...")
    Path("outputs").mkdir(exist_ok=True)
    df.to_csv("outputs/stage3_results.csv", index=False)
    logger.info("  Saved -> outputs/stage3_results.csv")

    print_summary(df)

    logger.info("\n[4/4] Generating summary plot...")
    if len(agent_types) == len(AGENT_TYPES):
        plot_summary(df)

    logger.info("\n" + "=" * 60)
    logger.info("Training complete.")
    logger.info("=" * 60)

    logger.info("\n  Best checkpoints:")
    for at in agent_types:
        sub      = df[df["agent_type"] == at.value]
        best_row = sub.loc[sub["best_val_sharpe"].idxmax()]
        logger.info(
            f"    {at.value:<30} "
            f"seed={int(best_row['seed'])}  "
            f"Sharpe={best_row['best_val_sharpe']:.4f}  "
            f"-> outputs/models/agent_{at.value}_seed{int(best_row['seed'])}_best.pt"
        )


if __name__ == "__main__":
    main()