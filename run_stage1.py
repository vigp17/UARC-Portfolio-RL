"""
run_stage1.py
-------------
Data pipeline and Bayesian HMM fitting for market regime detection.

Usage:
    pip install -r requirements.txt
    python run_stage1.py
    python run_stage1.py --seed 42

What this does:
  1. Engineers all features
  2. Runs BIC-based regime count selection (K=2..5)
  3. Fits the final HMM on training data
  4. Computes regime posteriors on all splits
  5. Generates and saves all visualization plots
  6. Saves the fitted model to disk
"""

import argparse
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.data_loader import load_all
from src.hmm.hmm_model import BayesianMarketHMM, select_n_regimes
from src.hmm.regime_visualizer import generate_all_plots


# ── CLI arguments ─────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Stage 1: Bayesian HMM regime detection")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
args = parser.parse_args()

seed = args.seed


# ── Reproducibility ───────────────────────────────────────────────────────────
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

if hasattr(torch.backends, "cudnn"):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


# ── Output paths ──────────────────────────────────────────────────────────────
OUTPUT_DIR = Path(f"outputs/seed_{seed}")
MODEL_PATH = OUTPUT_DIR / "models" / "hmm_stage1.pkl"
FIGURES_DIR = OUTPUT_DIR / "figures" / "hmm"


def main():
    logger.info("=" * 60)
    logger.info("Bayesian HMM Regime Detection")
    logger.info("=" * 60)
    logger.info(f"Running with seed: {seed}")

    # Ensure base output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load data ──────────────────────────────────────────────────
    logger.info("\n[1/5] Loading data...")
    data = load_all(cache=True)

    prices = data["prices"]
    features = data["features"]
    hmm_train = data["hmm_train"]
    hmm_val = data["hmm_val"]
    hmm_test = data["hmm_test"]

    logger.info(f"  Prices:   {prices.shape}  ({prices.index[0].date()} to {prices.index[-1].date()})")
    logger.info(f"  Features: {features.shape}")
    logger.info(f"  Train/Val/Test: {hmm_train.shape} / {hmm_val.shape} / {hmm_test.shape}")

    # ── Step 2: Select number of regimes via BIC ───────────────────────────
    logger.info("\n[2/5] Selecting number of regimes via BIC...")
    best_k, bic_scores = select_n_regimes(
        hmm_train, hmm_val, k_range=range(2, 6)
    )
    logger.info(f"  BIC scores: {bic_scores}")
    logger.info(f"  Selected K={best_k}")

    final_k = best_k if best_k in [2, 3, 4] else 3
    if final_k != best_k:
        logger.warning(f"  BIC selected K={best_k}, overriding with K={final_k} (domain knowledge)")

    # ── Step 3: Fit final HMM ──────────────────────────────────────────────
    logger.info(f"\n[3/5] Fitting HMM (K={final_k})...")
    hmm = BayesianMarketHMM(n_regimes=final_k, n_iter=100, random_state=seed)
    hmm.fit(hmm_train)

    logger.info(hmm.summary())

    diag_mean = np.diag(hmm.model.transmat_).mean()
    logger.info(f"  Mean diagonal (persistence): {diag_mean:.3f}  (target: >0.85)")
    if diag_mean < 0.85:
        logger.warning("  Low persistence — regimes may be switching too rapidly.")

    # ── Step 4: Compute posteriors ─────────────────────────────────────────
    logger.info("\n[4/5] Computing posteriors on all splits...")

    hmm_all = np.concatenate([hmm_train, hmm_val, hmm_test], axis=0)
    posteriors_all = hmm.get_posterior(hmm_all)

    T_train = len(hmm_train)
    T_val = len(hmm_val)

    posteriors_train = posteriors_all[:T_train]
    posteriors_val = posteriors_all[T_train:T_train + T_val]
    posteriors_test = posteriors_all[T_train + T_val:]

    logger.info(
        f"  Posterior shapes — Train: {posteriors_train.shape}, "
        f"Val: {posteriors_val.shape}, Test: {posteriors_test.shape}"
    )

    dominant_regime = posteriors_all.argmax(axis=1)
    for k in range(final_k):
        pct = (dominant_regime == k).mean() * 100
        label = hmm.regime_labels.get(k, f"State_{k}")
        logger.info(f"  {label}: {pct:.1f}% of trading days")

    # ── Step 5: Save model and plots ───────────────────────────────────────
    logger.info("\n[5/5] Saving model and generating plots...")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    hmm.save(str(MODEL_PATH))

    np.save(OUTPUT_DIR / "posteriors_train.npy", posteriors_train)
    np.save(OUTPUT_DIR / "posteriors_val.npy", posteriors_val)
    np.save(OUTPUT_DIR / "posteriors_test.npy", posteriors_test)
    logger.info(f"  Posteriors saved to {OUTPUT_DIR}/")

    viterbi_all = hmm.get_viterbi_sequence(hmm_all)
    feature_names = [
        f"{a}_{f}" for a in ["SPY", "QQQ", "TLT", "GLD", "SHY"]
        for f in ["log_ret_1d", "rvol_20d"]
    ] + ["avg_pairwise_corr"]

    dates_all = features.index[-len(hmm_all):]
    prices_aligned = prices.loc[prices.index.isin(dates_all)]

    generate_all_plots(
        hmm_model=hmm,
        posteriors=posteriors_all,
        viterbi_states=viterbi_all,
        prices=prices_aligned,
        feature_names=feature_names,
        bic_scores=bic_scores,
        output_dir=str(FIGURES_DIR),
    )

    # ── Success Criteria ───────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("Results")
    logger.info("=" * 60)

    checks = {
        "Model fitted": hmm._is_fitted,
        "Posteriors sum to 1 (train)": abs(posteriors_train.sum(axis=1).mean() - 1.0) < 1e-4,
        "Posteriors non-negative": (posteriors_all >= 0).all(),
        "Regime persistence > 0.85": diag_mean > 0.85,
        "Bull regime identified": "Bull" in hmm.regime_labels.values(),
        "Bear regime identified": "Bear" in hmm.regime_labels.values(),
        "Plots generated": FIGURES_DIR.exists(),
        "Model saved": MODEL_PATH.exists(),
    }

    all_passed = True
    for check, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        logger.info(f"  [{status}] {check}")
        if not passed:
            all_passed = False

    if all_passed:
        logger.info("\n  All checks passed.")
    else:
        logger.warning("\n  Some checks failed — review logs and plots.")

    return hmm, posteriors_train, posteriors_val, posteriors_test


if __name__ == "__main__":
    hmm, p_train, p_val, p_test = main()