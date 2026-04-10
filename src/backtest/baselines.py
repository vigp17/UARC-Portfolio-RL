import numpy as np


def equal_weight_from_returns(returns_window: np.ndarray) -> np.ndarray:
    n_assets = returns_window.shape[1]
    w = np.ones(n_assets, dtype=np.float32)
    return w / w.sum()


def risk_parity_from_returns(returns_window: np.ndarray) -> np.ndarray:
    vol = returns_window.std(axis=0)
    inv_vol = 1.0 / (vol + 1e-8)
    w = inv_vol / inv_vol.sum()
    return w.astype(np.float32)


def momentum_from_prices(prices_window: np.ndarray, lookback: int = 60) -> np.ndarray:
    n_assets = prices_window.shape[1]
    if len(prices_window) < lookback:
        w = np.ones(n_assets, dtype=np.float32)
        return w / w.sum()

    mom = prices_window[-1] / (prices_window[-lookback] + 1e-8) - 1.0
    scores = np.exp(mom - np.max(mom))
    w = scores / scores.sum()
    return w.astype(np.float32)


def mean_variance_from_returns(returns_window: np.ndarray) -> np.ndarray:
    mu = returns_window.mean(axis=0)
    cov = np.cov(returns_window.T)

    inv_cov = np.linalg.pinv(cov)
    raw = inv_cov @ mu
    raw = np.maximum(raw, 0.0)

    if raw.sum() < 1e-8:
        w = np.ones(len(mu), dtype=np.float32)
        return w / w.sum()

    w = raw / raw.sum()
    return w.astype(np.float32)