import numpy as np
from src.backtest.baselines import (
    equal_weight_from_returns,
    risk_parity_from_returns,
    momentum_from_prices,
    mean_variance_from_returns,
)


def test_equal_weight_sums_to_one():
    x = np.random.randn(100, 5)
    w = equal_weight_from_returns(x)
    assert np.isclose(w.sum(), 1.0)
    assert np.all(w >= 0)


def test_risk_parity_sums_to_one():
    x = np.random.randn(100, 5)
    w = risk_parity_from_returns(x)
    assert np.isclose(w.sum(), 1.0)
    assert np.all(w >= 0)


def test_momentum_sums_to_one():
    x = np.abs(np.random.randn(100, 5)) + 100.0
    w = momentum_from_prices(x, lookback=20)
    assert np.isclose(w.sum(), 1.0)
    assert np.all(w >= 0)


def test_mean_variance_sums_to_one():
    x = np.random.randn(100, 5)
    w = mean_variance_from_returns(x)
    assert np.isclose(w.sum(), 1.0)
    assert np.all(w >= 0)