"""
data_loader.py
--------------
Downloads OHLCV data for the 5-asset universe and engineers features
used by both the HMM regime module and the iTransformer encoder.

Assets: SPY, QQQ, TLT, GLD, SHY
Period: 2000-01-01 to 2024-12-31
Splits: Train / Val / Test (strict temporal, no leakage)
"""

import os
import logging
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from typing import Tuple, Dict, Optional

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

ASSETS = ["SPY", "QQQ", "TLT", "GLD", "SHY"]

SPLIT_DATES = {
    "train_start":  "2000-01-01",
    "train_end":    "2017-12-31",
    "val_start":    "2018-01-01",
    "val_end":      "2020-12-31",
    "test_start":   "2021-01-01",
    "test_end":     "2024-12-31",
}

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


# ── Downloader ─────────────────────────────────────────────────────────────────

def download_prices(
    assets: list = ASSETS,
    start: str = SPLIT_DATES["train_start"],
    end: str = SPLIT_DATES["test_end"],
    cache: bool = True,
) -> pd.DataFrame:
    """
    Downloads adjusted closing prices for all assets.
    Caches to data/raw/prices.csv to avoid repeated API calls.

    Returns
    -------
    pd.DataFrame
        Shape (T, N) — daily adjusted close prices, forward-filled.
    """
    cache_path = DATA_DIR / "raw" / "prices.csv"

    if cache and cache_path.exists():
        logger.info(f"Loading cached prices from {cache_path}")
        prices = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return prices

    logger.info(f"Downloading prices for {assets} from {start} to {end}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    raw = yf.download(
        assets,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )["Close"]

    # Ensure consistent column order
    raw = raw[assets]

    # Forward fill missing values (holidays, early closes)
    raw = raw.ffill().dropna()

    if cache:
        raw.to_csv(cache_path)
        logger.info(f"Saved prices to {cache_path}")

    return raw


# ── Feature Engineering ────────────────────────────────────────────────────────

def compute_features(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Computes all features used by the HMM and iTransformer encoder.

    Features per asset (returned as flat columns):
        - log_return_1d      : 1-day log return
        - log_return_5d      : 5-day log return
        - log_return_20d     : 20-day log return
        - realized_vol_20d   : 20-day rolling realized volatility (annualized)
        - rsi_14             : 14-day RSI (momentum)
        - macd_signal        : MACD line minus signal line (trend)

    Cross-asset features (single columns):
        - avg_pairwise_corr  : Rolling 60-day average pairwise correlation

    Parameters
    ----------
    prices : pd.DataFrame
        Shape (T, N) adjusted close prices.

    Returns
    -------
    pd.DataFrame
        Shape (T, N*per_asset_features + cross_features).
        Rows with NaN (due to lookback windows) are dropped.
    """
    feature_frames = []

    for asset in prices.columns:
        p = prices[asset]
        feat = pd.DataFrame(index=prices.index)

        # Log returns
        feat[f"{asset}_log_ret_1d"]  = np.log(p / p.shift(1))
        feat[f"{asset}_log_ret_5d"]  = np.log(p / p.shift(5))
        feat[f"{asset}_log_ret_20d"] = np.log(p / p.shift(20))

        # Realized volatility (annualized)
        log_ret = feat[f"{asset}_log_ret_1d"]
        feat[f"{asset}_rvol_20d"] = log_ret.rolling(20).std() * np.sqrt(252)

        # RSI
        feat[f"{asset}_rsi_14"] = _rsi(p, period=14)

        # MACD (12-26 EMA diff, signal = 9-day EMA of that diff)
        feat[f"{asset}_macd"]   = _macd_signal(p)

        feature_frames.append(feat)

    features = pd.concat(feature_frames, axis=1)

    # Rolling pairwise correlation (cross-asset feature)
    log_rets = np.log(prices / prices.shift(1))
    rolling_corr = log_rets.rolling(60).corr()
    # Average off-diagonal correlation per day
    avg_corr = []
    for date in prices.index:
        try:
            mat = rolling_corr.loc[date]
            if isinstance(mat, pd.DataFrame):
                vals = mat.values
                mask = ~np.eye(vals.shape[0], dtype=bool)
                avg_corr.append(np.nanmean(vals[mask]))
            else:
                avg_corr.append(np.nan)
        except KeyError:
            avg_corr.append(np.nan)

    features["avg_pairwise_corr"] = avg_corr

    # Drop rows with any NaN (from lookback windows)
    features = features.dropna()

    return features


def _rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = prices.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd_signal(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """MACD line minus signal line."""
    ema_fast   = prices.ewm(span=fast, adjust=False).mean()
    ema_slow   = prices.ewm(span=slow, adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - signal_line


# ── Splits ─────────────────────────────────────────────────────────────────────

def get_splits(
    features: pd.DataFrame,
    split_dates: dict = SPLIT_DATES,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Applies strict temporal splits. No overlap, no leakage.

    Returns
    -------
    train, val, test : pd.DataFrame
        Each is a slice of the features DataFrame.
    """
    train = features.loc[split_dates["train_start"]: split_dates["train_end"]]
    val   = features.loc[split_dates["val_start"]  : split_dates["val_end"]]
    test  = features.loc[split_dates["test_start"] : split_dates["test_end"]]

    logger.info(f"Train: {len(train)} days  |  Val: {len(val)} days  |  Test: {len(test)} days")
    assert len(train) > 0 and len(val) > 0 and len(test) > 0, "Empty split — check date ranges"

    return train, val, test


# ── HMM-Specific Feature Extraction ───────────────────────────────────────────

def get_hmm_features(features: pd.DataFrame) -> np.ndarray:
    """
    Selects and returns the feature subset used for HMM fitting.

    The HMM operates on a compact representation that captures regime-relevant
    market dynamics: returns, volatility, and cross-asset correlation.

    Returns
    -------
    np.ndarray
        Shape (T, 11) — market-level features for HMM.
    """
    hmm_cols = []

    for asset in ASSETS:
        # Return and vol per asset — core regime signals
        hmm_cols += [
            f"{asset}_log_ret_1d",
            f"{asset}_rvol_20d",
        ]

    # Cross-asset correlation — regime signal at portfolio level
    hmm_cols.append("avg_pairwise_corr")

    available = [c for c in hmm_cols if c in features.columns]
    return features[available].values.astype(np.float32)


# ── Main Entry Point ───────────────────────────────────────────────────────────

def load_all(cache: bool = True) -> Dict[str, object]:
    """
    Full pipeline: download -> feature engineering -> split.

    Returns
    -------
    dict with keys:
        prices          : raw price DataFrame
        features        : full feature DataFrame
        train/val/test  : split DataFrames
        hmm_train/val/test : np.ndarray for HMM fitting (compact features)
    """
    prices   = download_prices(cache=cache)
    features = compute_features(prices)

    train, val, test = get_splits(features)

    return {
        "prices"    : prices,
        "features"  : features,
        "train"     : train,
        "val"       : val,
        "test"      : test,
        "hmm_train" : get_hmm_features(train),
        "hmm_val"   : get_hmm_features(val),
        "hmm_test"  : get_hmm_features(test),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    data = load_all(cache=True)
    print("\n=== Data Summary ===")
    print(f"Prices shape:   {data['prices'].shape}")
    print(f"Features shape: {data['features'].shape}")
    print(f"Train days:     {len(data['train'])}")
    print(f"Val days:       {len(data['val'])}")
    print(f"Test days:      {len(data['test'])}")
    print(f"HMM train:      {data['hmm_train'].shape}")
    print(f"\nFeature columns:\n{list(data['features'].columns)}")
