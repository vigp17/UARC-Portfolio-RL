"""
run_stage2.py
-------------
Stage 2: iTransformer Encoder validation and visualization.

Run after Stage 1:
    python run_stage2.py

What this does:
  1. Loads Stage 1 outputs (prices, posteriors)
  2. Builds encoder features for all splits
  3. Fits normalizer on training data
  4. Instantiates and validates the iTransformerEncoder
  5. Runs a forward pass on real data — checks shapes and values
  6. Generates attention weight visualization (cross-asset correlations)
  7. Saves encoder + normalizer to disk for Stage 3

Stage 2 is DONE when:
  - All shape checks pass
  - No NaN/Inf in encoder output
  - Attention heatmap shows economically sensible asset-pair weights
    (e.g. SPY-QQQ high attention, TLT-SPY negative correlation visible)
  - Encoder + normalizer saved to outputs/models/
"""

import logging
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.data.data_loader import load_all, ASSETS
from src.encoder.itransformer import iTransformerEncoder
from src.encoder.features import (
    build_encoder_features, FeatureNormalizer,
    PortfolioSequenceDataset, LOOKBACK
)
import pickle

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

OUTPUT_DIR  = Path("outputs")
MODEL_DIR   = OUTPUT_DIR / "models"
FIGURES_DIR = OUTPUT_DIR / "figures" / "stage2"
N_FEATURES  = 6    # per asset features
D_MODEL     = 64
N_HEADS     = 4
N_LAYERS    = 2
LOOKBACK_T  = LOOKBACK   # 60 days


def main():
    logger.info("=" * 60)
    logger.info("STAGE 2: iTransformer Encoder")
    logger.info("=" * 60)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load data ──────────────────────────────────────────────────
    logger.info("\n[1/6] Loading data and Stage 1 outputs...")
    data   = load_all(cache=True)
    prices = data["prices"]

    posteriors_train = np.load(OUTPUT_DIR / "posteriors_train.npy")
    posteriors_val   = np.load(OUTPUT_DIR / "posteriors_val.npy")
    posteriors_test  = np.load(OUTPUT_DIR / "posteriors_test.npy")
    K = posteriors_train.shape[1]
    logger.info(f"  Posteriors loaded — K={K} regimes")

    # ── Step 2: Build encoder features ────────────────────────────────────
    logger.info("\n[2/6] Building encoder features...")
    X_all = build_encoder_features(prices, assets=ASSETS)
    logger.info(f"  Full feature array: {X_all.shape}")

    # Align with splits (same logic as data_loader)
    train_end  = data["train"].index[-1]
    val_end    = data["val"].index[-1]

    # Map feature rows back to dates — align by length from the end
    valid_dates = prices.index[-len(X_all):]
    X_df        = pd.DataFrame(X_all.reshape(len(X_all), -1), index=valid_dates)

    X_train_df = X_df.loc[:train_end]
    X_val_df   = X_df.loc[train_end:val_end].iloc[1:]   # avoid overlap
    X_test_df  = X_df.loc[val_end:].iloc[1:]

    X_train = X_train_df.values.reshape(-1, len(ASSETS), N_FEATURES)
    X_val   = X_val_df.values.reshape(-1, len(ASSETS), N_FEATURES)
    X_test  = X_test_df.values.reshape(-1, len(ASSETS), N_FEATURES)

    logger.info(f"  Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

    # ── Step 3: Fit normalizer ─────────────────────────────────────────────
    logger.info("\n[3/6] Fitting feature normalizer on training data...")
    normalizer = FeatureNormalizer()
    X_train_norm = normalizer.fit_transform(X_train)
    X_val_norm   = normalizer.transform(X_val)
    X_test_norm  = normalizer.transform(X_test)
    logger.info(f"  Train mean: {X_train_norm.mean():.4f}  std: {X_train_norm.std():.4f}")

    # ── Step 4: Build encoder ──────────────────────────────────────────────
    logger.info("\n[4/6] Building iTransformerEncoder...")
    encoder = iTransformerEncoder(
        n_assets  = len(ASSETS),
        lookback  = LOOKBACK_T * N_FEATURES,  # flattened input per asset
        d_model   = D_MODEL,
        n_heads   = N_HEADS,
        n_layers  = N_LAYERS,
        d_ff      = D_MODEL * 4,
        dropout   = 0.1,
        pooling   = "mean",
    )
    logger.info(f"  Parameters: {encoder.count_parameters():,}")

    # ── Step 5: Forward pass validation ───────────────────────────────────
    logger.info("\n[5/6] Validating encoder with real data...")
    encoder.eval()

    # Build a batch of windows from training data
    T_train = len(X_train_norm)
    windows = []
    for t in range(LOOKBACK_T, min(LOOKBACK_T + 32, T_train)):
        window = X_train_norm[t - LOOKBACK_T: t]           # (lookback, n_assets, n_features)
        window_flat = window.transpose(1, 0, 2).reshape(len(ASSETS), -1)  # (n_assets, lookback*n_feat)
        windows.append(window_flat)

    batch = torch.from_numpy(np.stack(windows, axis=0))    # (32, n_assets, lookback*n_feat)
    logger.info(f"  Batch shape: {batch.shape}")

    with torch.no_grad():
        embeddings = encoder(batch)

    logger.info(f"  Output shape: {embeddings.shape}  (expected: (32, {D_MODEL}))")
    logger.info(f"  Output mean: {embeddings.mean().item():.4f}")
    logger.info(f"  Output std:  {embeddings.std().item():.4f}")
    logger.info(f"  NaN count:   {torch.isnan(embeddings).sum().item()}")
    logger.info(f"  Inf count:   {torch.isinf(embeddings).sum().item()}")

    # Verify IQN input dimension
    regime_posterior_dim = K
    iqn_input_dim = D_MODEL + regime_posterior_dim
    logger.info(f"\n  IQN agent input dim: {D_MODEL} (encoder) + {K} (HMM) = {iqn_input_dim}")

    # ── Step 6: Attention visualization ───────────────────────────────────
    logger.info("\n[6/6] Generating attention visualizations...")
    attn_weights = encoder.get_attention_weights(batch[:8])

    for layer_idx, weights in enumerate(attn_weights):
        # Average across batch and heads: (n_assets, n_assets)
        avg_weights = weights.mean(dim=(0, 1)).numpy()

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            avg_weights,
            annot=True, fmt=".3f",
            xticklabels=ASSETS,
            yticklabels=ASSETS,
            cmap="Blues", ax=ax,
            vmin=0, vmax=avg_weights.max(),
            linewidths=0.5, linecolor="white",
        )
        ax.set_title(f"Cross-Asset Attention Weights — Layer {layer_idx + 1}\n"
                     f"(averaged over batch and heads)",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Attends To", fontsize=9)
        ax.set_ylabel("Query Asset", fontsize=9)
        plt.tight_layout()
        path = FIGURES_DIR / f"attention_layer{layer_idx + 1}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"  Saved attention plot: {path}")

    # ── Save encoder and normalizer ────────────────────────────────────────
    torch.save(encoder.state_dict(), MODEL_DIR / "encoder_stage2.pt")
    with open(MODEL_DIR / "normalizer_stage2.pkl", "wb") as f:
        pickle.dump(normalizer, f)
    logger.info(f"\n  Encoder saved to {MODEL_DIR}/encoder_stage2.pt")
    logger.info(f"  Normalizer saved to {MODEL_DIR}/normalizer_stage2.pkl")

    # Save normalized features for Stage 3
    np.save(OUTPUT_DIR / "enc_features_train.npy", X_train_norm)
    np.save(OUTPUT_DIR / "enc_features_val.npy",   X_val_norm)
    np.save(OUTPUT_DIR / "enc_features_test.npy",  X_test_norm)

    # ── Stage 2 Success Criteria ───────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 2 COMPLETE — Success Criteria Check")
    logger.info("=" * 60)

    checks = {
        "Output shape correct (32, 64)":        embeddings.shape == (32, D_MODEL),
        "No NaN in output":                      not torch.isnan(embeddings).any().item(),
        "No Inf in output":                      not torch.isinf(embeddings).any().item(),
        "Gradients flow (differentiable)":       _check_gradients(encoder, batch),
        "Encoder saved":                         (MODEL_DIR / "encoder_stage2.pt").exists(),
        "Normalizer saved":                      (MODEL_DIR / "normalizer_stage2.pkl").exists(),
        "Attention plots generated":             (FIGURES_DIR / "attention_layer1.png").exists(),
        "Features saved for Stage 3":            (OUTPUT_DIR / "enc_features_train.npy").exists(),
    }

    all_passed = True
    for check, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        logger.info(f"  [{status}] {check}")
        if not passed:
            all_passed = False

    if all_passed:
        logger.info("\n  ALL CHECKS PASSED. Ready for Stage 3 (IQN Agent).")
        logger.info(f"\n  IQN agent state input: encoder({D_MODEL}) + HMM({K}) = {D_MODEL + K} dims")
    else:
        logger.warning("\n  Some checks failed — review logs before proceeding.")

    return encoder, normalizer


def _check_gradients(encoder, batch):
    """Quick gradient check — returns True if gradients flow."""
    try:
        x = batch.clone().requires_grad_(True)
        out = encoder(x)
        out.sum().backward()
        return x.grad is not None and not torch.isnan(x.grad).any()
    except Exception:
        return False


if __name__ == "__main__":
    encoder, normalizer = main()