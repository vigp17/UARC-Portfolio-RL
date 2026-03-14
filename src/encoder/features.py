"""
features.py
-----------
Feature pipeline that converts raw price data into normalized tensors
ready for the iTransformer encoder.

The encoder expects: (batch, n_assets, lookback)
  - Each asset is a token
  - Its features are its lookback-day return history
  - All features are normalized per-asset (z-score on training stats)

This module also builds the SequenceDataset used for training the
full UARC system (encoder + agent together in Stage 3).
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)

ASSETS  = ["SPY", "QQQ", "TLT", "GLD", "SHY"]
LOOKBACK = 60   # Trading days of history per sample


class FeatureNormalizer:
    """
    Per-asset z-score normalization fitted on training data.

    Fitted once on train split, applied to val and test without refitting.
    Stored alongside the HMM model for inference.
    """

    def __init__(self):
        self.means = None   # shape (n_assets, n_features)
        self.stds  = None
        self._fitted = False

    def fit(self, X: np.ndarray):
        """
        Parameters
        ----------
        X : np.ndarray
            Shape (T, n_assets, n_features)
        """
        self.means   = X.mean(axis=0, keepdims=True)    # (1, n_assets, n_features)
        self.stds    = X.std(axis=0, keepdims=True) + 1e-8
        self._fitted = True
        logger.info(f"Normalizer fitted on shape {X.shape}")

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call .fit() before .transform()")
        return ((X - self.means) / self.stds).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)


def build_encoder_features(
    prices: pd.DataFrame,
    assets: list = ASSETS,
) -> np.ndarray:
    """
    Builds the per-asset feature matrix used as iTransformer input.

    For each asset, computes:
        - 1-day log return
        - 5-day log return
        - 20-day log return
        - 20-day realized volatility (annualized)
        - 14-day RSI (normalized to [-1, 1])
        - MACD signal line

    Returns
    -------
    np.ndarray
        Shape (T, n_assets, n_features) where n_features=6 per asset.
        NaN rows (from lookback computation) are dropped.
    """
    n_assets   = len(assets)
    n_features = 6
    feature_list = []

    for asset in assets:
        p    = prices[asset]
        feat = np.zeros((len(p), n_features))

        # Log returns
        log_ret = np.log(p / p.shift(1))
        feat[:, 0] = log_ret.values                              # 1d return
        feat[:, 1] = np.log(p / p.shift(5)).values              # 5d return
        feat[:, 2] = np.log(p / p.shift(20)).values             # 20d return

        # Realized volatility (annualized)
        feat[:, 3] = log_ret.rolling(20).std().values * np.sqrt(252)

        # RSI normalized to [-1, 1]
        rsi = _rsi(p, 14)
        feat[:, 4] = (rsi.values / 50.0) - 1.0

        # MACD signal
        feat[:, 5] = _macd_signal(p).values

        feature_list.append(feat)

    # Stack: shape (T, n_assets, n_features)
    X = np.stack(feature_list, axis=1).astype(np.float32)

    # Drop rows with NaN (first ~26 days due to lookback)
    valid_mask = ~np.isnan(X).any(axis=(1, 2))
    X = X[valid_mask]

    logger.info(f"Encoder features: shape={X.shape}  "
                f"({n_assets} assets x {n_features} features, {X.shape[0]} timesteps)")
    return X


def _rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd_signal(prices: pd.Series, fast=12, slow=26, signal=9) -> pd.Series:
    ema_fast    = prices.ewm(span=fast, adjust=False).mean()
    ema_slow    = prices.ewm(span=slow, adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - signal_line


class PortfolioSequenceDataset(Dataset):
    """
    PyTorch Dataset for the full UARC training loop (used in Stage 3).

    Each sample is one trading day t, containing:
        - encoder_input : (n_assets, lookback)   — last 60 days of features
        - regime_posterior: (K,)                 — HMM posterior at time t
        - portfolio_weights: (n_assets,)          — previous day's weights
        - returns: (n_assets,)                    — next-day log returns (label)

    Parameters
    ----------
    encoder_features : np.ndarray
        Shape (T, n_assets, n_features) — from build_encoder_features()
    posteriors : np.ndarray
        Shape (T, K) — from BayesianMarketHMM.get_posterior()
    prices : pd.DataFrame
        Shape (T, n_assets) — used to compute forward returns
    lookback : int
        Sequence length for encoder input window.
    """

    def __init__(
        self,
        encoder_features: np.ndarray,
        posteriors:        np.ndarray,
        prices:            pd.DataFrame,
        lookback:          int = LOOKBACK,
    ):
        super().__init__()
        self.lookback = lookback

        # Align all arrays to the same length
        T = min(len(encoder_features), len(posteriors), len(prices))
        self.enc_feat   = encoder_features[-T:].astype(np.float32)
        self.posteriors = posteriors[-T:].astype(np.float32)

        # Compute daily log returns for all assets
        log_rets = np.log(prices.values / prices.shift(1).values)
        self.returns = log_rets[-T:].astype(np.float32)

        # Valid start index: need lookback days of history
        self.start = lookback

        logger.info(f"Dataset: {len(self)} samples  "
                    f"(T={T}, lookback={lookback}, "
                    f"n_assets={self.enc_feat.shape[1]}, "
                    f"n_features={self.enc_feat.shape[2]}, "
                    f"K={self.posteriors.shape[1]})")

    def __len__(self) -> int:
        return len(self.enc_feat) - self.lookback - 1

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        t = idx + self.lookback

        # Encoder input: past `lookback` days of features
        # Shape: (n_assets, lookback, n_features)
        window = self.enc_feat[t - self.lookback: t]  # (lookback, n_assets, n_features)
        # Reshape to (n_assets, lookback * n_features) — flatten time+features per asset
        # This is the correct iTransformer input: each asset's full history as a vector
        n_assets   = window.shape[1]
        n_features = window.shape[2]
        encoder_input = window.transpose(1, 0, 2).reshape(n_assets, self.lookback * n_features)

        return {
            "encoder_input":     torch.from_numpy(encoder_input),          # (n_assets, lookback*n_feat)
            "regime_posterior":  torch.from_numpy(self.posteriors[t]),     # (K,)
            "next_returns":      torch.from_numpy(self.returns[t + 1]) if t + 1 < len(self.returns)
                                 else torch.zeros(n_assets),                # (n_assets,)
        }


def build_dataloaders(
    encoder_features_train: np.ndarray,
    encoder_features_val:   np.ndarray,
    posteriors_train:       np.ndarray,
    posteriors_val:         np.ndarray,
    prices_train:           pd.DataFrame,
    prices_val:             pd.DataFrame,
    lookback:               int = LOOKBACK,
    batch_size:             int = 256,
    num_workers:            int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """
    Builds train and validation DataLoaders for Stage 3 training.

    Returns
    -------
    train_loader, val_loader : DataLoader
    """
    train_dataset = PortfolioSequenceDataset(
        encoder_features_train, posteriors_train, prices_train, lookback
    )
    val_dataset = PortfolioSequenceDataset(
        encoder_features_val, posteriors_val, prices_val, lookback
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,      # NEVER shuffle financial time series
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    logger.info(f"Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")
    return train_loader, val_loader