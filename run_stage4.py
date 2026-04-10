"""
run_stage4.py
-------------
Backtest all systems on the holdout test set (2021-2024).

Run after training:
    python run_stage4.py
"""

import logging
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from src.data.data_loader import load_all, ASSETS, SPLIT_DATES
from src.encoder.itransformer import iTransformerEncoder
from src.agent.iqn import IQNAgent
from src.backtest.backtest import (
    backtest_buy_and_hold,
    backtest_equal_weight,
    backtest_risk_parity,
    backtest_momentum,
    backtest_mean_variance,
    backtest_hmm_hard_label_dqn,
    backtest_no_regime_iqn,
    backtest_hmm_posterior_dqn,
    backtest_uarc_full,
    build_results_table,
    print_results_table,
)
from src.backtest.visualize import (
    plot_cumulative_returns,
    plot_drawdowns,
    plot_weight_evolution,
    plot_regime_overlay,
    plot_rolling_sharpe,
    save_results_csv,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

OUTPUT_DIR  = Path("outputs")
MODEL_DIR   = OUTPUT_DIR / "models"
FIGURES_DIR = OUTPUT_DIR / "figures" / "backtest"
N_FEATURES  = 6
LOOKBACK    = 60


def main():
    logger.info("=" * 60)
    logger.info("STAGE 4: Backtesting")
    logger.info("=" * 60)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load test data ─────────────────────────────────────────────
    logger.info("\n[1/5] Loading test data...")
    data   = load_all(cache=True)
    prices = data["prices"]

    posteriors_test   = np.load(OUTPUT_DIR / "posteriors_test.npy")
    enc_features_test = np.load(OUTPUT_DIR / "enc_features_test.npy")
    K = posteriors_test.shape[1]

    val_end         = pd.Timestamp(SPLIT_DATES["val_end"])
    prices_test     = prices.loc[val_end:].iloc[1:]
    prices_test_np  = prices_test.values[:len(enc_features_test)]
    test_dates      = None  # handled per-result in visualize.py

    logger.info(f"  Test enc features: {enc_features_test.shape}")
    logger.info(f"  Test posteriors:   {posteriors_test.shape}")
    logger.info(f"  Test prices:       {prices_test_np.shape}")
    logger.info(f"  Test period:       {prices_test.index[0].date()} to {prices_test.index[-1].date()}")

    # ── Step 2: Load models ────────────────────────────────────────────────
    logger.info("\n[2/5] Loading trained models...")

    device = torch.device(
        "mps"  if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    def load_encoder(enc_ckpt_name):
        """Load per-agent encoder from joint training."""
        enc = iTransformerEncoder(
            n_assets = len(ASSETS),
            lookback = LOOKBACK * N_FEATURES,
            d_model  = 64,
            n_heads  = 4,
            n_layers = 2,
            dropout  = 0.0,
        ).to(device)
        enc.load_state_dict(
            torch.load(MODEL_DIR / enc_ckpt_name, map_location=device)
        )
        enc.eval()
        for p in enc.parameters():
            p.requires_grad_(False)
        return enc

    def load_agent(ckpt_name):
        a = IQNAgent(
            n_assets    = len(ASSETS),
            encoder_dim = 64,
            n_regimes   = K,
            d_model     = 256,
            n_cos       = 64,
            n_tau_train = 32,
            n_tau_eval  = 64,
        ).to(device)
        a.load_state_dict(
            torch.load(MODEL_DIR / ckpt_name, map_location=device)
        )
        a.eval()
        return a

    def compute_embeddings(encoder, enc_features):
        T    = len(enc_features)
        embs = np.zeros((T, 64), dtype=np.float32)
        with torch.no_grad():
            for t in range(LOOKBACK, T):
                window = enc_features[t - LOOKBACK: t]
                enc_in = window.transpose(1, 0, 2).reshape(
                    1, len(ASSETS), LOOKBACK * N_FEATURES
                )
                embs[t] = encoder(
                    torch.from_numpy(enc_in).to(device)
                ).cpu().numpy()
        return embs

    # Each agent has its own encoder from joint training
    enc_no_regime  = load_encoder("encoder_no_regime_iqn_seed42_best.pt")
    enc_hard_dqn   = load_encoder("encoder_hmm_hard_dqn_seed42_best.pt")
    enc_post_dqn   = load_encoder("encoder_hmm_posterior_dqn_seed42_best.pt")
    enc_uarc       = load_encoder("encoder_uarc_seed42_best.pt")

    agent_no_regime = load_agent("agent_no_regime_iqn_seed42_best.pt")
    agent_hard_dqn  = load_agent("agent_hmm_hard_dqn_seed42_best.pt")
    agent_post_dqn  = load_agent("agent_hmm_posterior_dqn_seed42_best.pt")
    agent_uarc      = load_agent("agent_uarc_seed42_best.pt")

    logger.info(f"  Loaded 4 agent + encoder checkpoints (seed=42)")
    logger.info(f"  Device: {device}")

    # Per-agent embeddings from per-agent encoders
    logger.info("  Pre-computing per-agent test embeddings...")
    embs_no_regime = compute_embeddings(enc_no_regime,  enc_features_test)
    embs_hard_dqn  = compute_embeddings(enc_hard_dqn,   enc_features_test)
    embs_post_dqn  = compute_embeddings(enc_post_dqn,   enc_features_test)
    embs_uarc      = compute_embeddings(enc_uarc,        enc_features_test)
    logger.info(f"  Embeddings shape: {embs_uarc.shape}")

    # ── Step 3: Run backtests ──────────────────────────────────────────────
    logger.info("\n[3/5] Running backtests...")

    logger.info("  [1/9] Buy & Hold...")
    bh = backtest_buy_and_hold(prices_test_np, n_assets=len(ASSETS))

    logger.info("  [2/9] Equal Weight...")
    eq = backtest_equal_weight(prices_test_np, lookback=LOOKBACK)

    logger.info("  [3/9] Risk Parity...")
    rp = backtest_risk_parity(prices_test_np, lookback=LOOKBACK)

    logger.info("  [4/9] Momentum...")
    mom = backtest_momentum(prices_test_np, lookback=LOOKBACK)

    logger.info("  [5/9] Mean-Variance...")
    mv = backtest_mean_variance(prices_test_np, lookback=LOOKBACK)

    logger.info("  [6/9] HMM Hard Label + DQN...")
    hard_dqn = backtest_hmm_hard_label_dqn(
        enc_hard_dqn, agent_hard_dqn, embs_hard_dqn, posteriors_test, prices_test_np,
        device=device, lookback=LOOKBACK, n_features=N_FEATURES,
        regime_mode="hard",
    )

    logger.info("  [7/9] No Regime + IQN...")
    no_regime = backtest_no_regime_iqn(
        enc_no_regime, agent_no_regime, embs_no_regime, prices_test_np,
        device=device, lookback=LOOKBACK, n_features=N_FEATURES,
        n_regimes=K,
    )

    logger.info("  [8/9] HMM Posterior + DQN...")
    post_dqn = backtest_hmm_posterior_dqn(
        enc_post_dqn, agent_post_dqn, embs_post_dqn, posteriors_test, prices_test_np,
        device=device, lookback=LOOKBACK, n_features=N_FEATURES,
        regime_mode="posterior",
    )

    logger.info("  [9/9] UARC Full System...")
    uarc = backtest_uarc_full(
        enc_uarc, agent_uarc, embs_uarc, posteriors_test, prices_test_np,
        device=device, lookback=LOOKBACK, n_features=N_FEATURES,
    )

    results = [bh, eq, rp, mom, mv, hard_dqn, no_regime, post_dqn, uarc]

    # ── Step 4: Print results ──────────────────────────────────────────────
    print_results_table(results)

    csv_path = OUTPUT_DIR / "backtest_results.csv"
    df = save_results_csv(results, csv_path)
    logger.info(f"  Results saved to {csv_path}")

    # ── Step 5: Generate figures ───────────────────────────────────────────
    logger.info("\n[4/5] Generating paper figures...")

    plot_cumulative_returns(
        results, test_dates=test_dates,
        save_path=FIGURES_DIR / "fig1_cumulative_returns.png"
    )
    logger.info("  [SAVED] fig1_cumulative_returns.png")

    plot_drawdowns(
        results, test_dates=test_dates,
        save_path=FIGURES_DIR / "fig2_drawdowns.png"
    )
    logger.info("  [SAVED] fig2_drawdowns.png")

    plot_weight_evolution(
        uarc, test_dates=test_dates, assets=ASSETS,
        save_path=FIGURES_DIR / "fig3_weight_evolution.png"
    )
    logger.info("  [SAVED] fig3_weight_evolution.png")

    post_test_aligned = posteriors_test[:len(uarc.daily_returns)]
    plot_regime_overlay(
        uarc, posteriors=post_test_aligned, test_dates=None,
        save_path=FIGURES_DIR / "fig4_regime_overlay.png"
    )
    logger.info("  [SAVED] fig4_regime_overlay.png")

    plot_rolling_sharpe(
        results, window=63, test_dates=test_dates,
        save_path=FIGURES_DIR / "fig5_rolling_sharpe.png"
    )
    logger.info("  [SAVED] fig5_rolling_sharpe.png")

    # ── Success criteria ───────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("Backtest complete.")
    logger.info("=" * 60)

    checks = {
        f"UARC Sharpe ({uarc.sharpe_ratio:.3f}) > Buy&Hold ({bh.sharpe_ratio:.3f})":
            uarc.sharpe_ratio > bh.sharpe_ratio,
        f"UARC CVaR ({uarc.cvar_5:.4f}) >= Buy&Hold ({bh.cvar_5:.4f})":
            uarc.cvar_5 >= bh.cvar_5,
        f"UARC Max Drawdown ({uarc.max_drawdown:.1%}) < 30%":
            uarc.max_drawdown > -0.30,
        "Results CSV saved": csv_path.exists(),
        "All 5 figures saved": all(
            (FIGURES_DIR / f"fig{i+1}_{n}.png").exists()
            for i, n in enumerate([
                "cumulative_returns", "drawdowns",
                "weight_evolution", "regime_overlay", "rolling_sharpe"
            ])
        ),
    }

    for check, passed in checks.items():
        logger.info(f"  [{'PASS' if passed else 'WARN'}] {check}")

    logger.info("\n" + "=" * 60)
    logger.info("PAPER RESULTS SUMMARY")
    logger.info("=" * 60)
    for r in results:
        marker = " <- OUR SYSTEM" if "UARC" in r.name else ""
        logger.info(
            f"  {r.name:<35} Sharpe: {r.sharpe_ratio:+.3f}  "
            f"CVaR: {r.cvar_5:.4f}  MaxDD: {r.max_drawdown:.1%}{marker}"
        )

    if all(checks.values()):
        logger.info("\n  All checks passed.")
    else:
        logger.info("\n  Some checks produced warnings — review results above.")
    return results


if __name__ == "__main__":
    results = main()