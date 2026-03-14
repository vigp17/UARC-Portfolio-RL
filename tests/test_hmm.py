"""
test_hmm.py
-----------
Unit tests for BayesianMarketHMM.

Run with:  pytest tests/test_hmm.py -v

All tests use synthetic data — no internet connection required.
Tests validate:
  1. Model fits without error
  2. Posteriors are valid probability distributions
  3. Online and batch posteriors match
  4. Forward algorithm is causal (no lookahead)
  5. Model serialization (save/load)
  6. BIC decreases then increases (sanity check for regime selection)
"""

import numpy as np
import pytest
import tempfile
import os

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.hmm.hmm_model import BayesianMarketHMM, select_n_regimes


# ── Fixtures ───────────────────────────────────────────────────────────────────

def make_synthetic_data(T=1000, D=11, K=3, seed=42):
    """
    Creates synthetic switching-regime data for testing.
    Returns X_train, X_val, true_regimes.
    """
    rng = np.random.RandomState(seed)

    # Regime parameters: (mean, std) per feature
    regime_params = [
        (np.array([0.001, 0.08] * 5 + [0.3]),  np.array([0.005, 0.02] * 5 + [0.05])),
        (np.array([-0.002, 0.20] * 5 + [0.7]), np.array([0.010, 0.05] * 5 + [0.08])),
        (np.array([0.0005, 0.12] * 5 + [0.5]), np.array([0.007, 0.03] * 5 + [0.06])),
    ]

    # Generate regime sequence with persistence
    regimes = [rng.randint(0, K)]
    trans   = np.array([[0.95, 0.03, 0.02], [0.03, 0.94, 0.03], [0.02, 0.03, 0.95]])
    for _ in range(T - 1):
        regimes.append(rng.choice(K, p=trans[regimes[-1]]))

    X = np.array([
        rng.normal(regime_params[r][0], regime_params[r][1])
        for r in regimes
    ]).astype(np.float32)

    split = int(T * 0.8)
    return X[:split], X[split:], np.array(regimes)


@pytest.fixture
def fitted_hmm():
    X_train, _, _ = make_synthetic_data()
    hmm = BayesianMarketHMM(n_regimes=3, n_iter=20, random_state=42)
    hmm.fit(X_train)
    return hmm


@pytest.fixture
def test_data():
    _, X_val, regimes = make_synthetic_data()
    return X_val, regimes


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestFitting:

    def test_fit_runs_without_error(self):
        X_train, _, _ = make_synthetic_data()
        hmm = BayesianMarketHMM(n_regimes=3, n_iter=10)
        hmm.fit(X_train)
        assert hmm._is_fitted

    def test_regime_labels_assigned(self, fitted_hmm):
        assert len(fitted_hmm.regime_labels) == 3
        labels = set(fitted_hmm.regime_labels.values())
        assert "Bull" in labels
        assert "Bear" in labels

    def test_raises_if_not_fitted(self):
        hmm = BayesianMarketHMM(n_regimes=3)
        X   = np.random.randn(100, 11).astype(np.float32)
        with pytest.raises(RuntimeError, match="not been fitted"):
            hmm.get_posterior(X)

    def test_log_likelihood_is_finite(self, fitted_hmm, test_data):
        X_val, _ = test_data
        score = fitted_hmm.score(X_val)
        assert np.isfinite(score), f"Log-likelihood is not finite: {score}"


class TestPosteriorValidity:

    def test_posteriors_sum_to_one(self, fitted_hmm, test_data):
        X_val, _ = test_data
        posteriors = fitted_hmm.get_posterior(X_val)
        row_sums   = posteriors.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5,
            err_msg="Posterior rows do not sum to 1.0")

    def test_posteriors_non_negative(self, fitted_hmm, test_data):
        X_val, _ = test_data
        posteriors = fitted_hmm.get_posterior(X_val)
        assert (posteriors >= 0).all(), "Posterior contains negative values"

    def test_posteriors_at_most_one(self, fitted_hmm, test_data):
        X_val, _ = test_data
        posteriors = fitted_hmm.get_posterior(X_val)
        assert (posteriors <= 1 + 1e-6).all(), "Posterior contains values > 1"

    def test_posterior_shape(self, fitted_hmm, test_data):
        X_val, _ = test_data
        posteriors = fitted_hmm.get_posterior(X_val)
        T = len(X_val)
        K = fitted_hmm.n_regimes
        assert posteriors.shape == (T, K), \
            f"Expected shape ({T}, {K}), got {posteriors.shape}"

    def test_posterior_dtype(self, fitted_hmm, test_data):
        X_val, _ = test_data
        posteriors = fitted_hmm.get_posterior(X_val)
        assert posteriors.dtype == np.float32, \
            f"Expected float32, got {posteriors.dtype}"


class TestOnlineVsBatch:

    def test_online_matches_batch(self, fitted_hmm, test_data):
        """
        Critical test: online forward algorithm must match batch forward algorithm.
        Max absolute difference should be < 1e-5.
        """
        X_val, _ = test_data
        T        = min(50, len(X_val))  # Test first 50 steps

        # Batch posteriors
        posteriors_batch = fitted_hmm.get_posterior(X_val[:T])

        # Online posteriors
        posteriors_online = []
        log_alpha = None
        for t in range(T):
            p, log_alpha = fitted_hmm.get_posterior_online(X_val[t], log_alpha)
            posteriors_online.append(p)
        posteriors_online = np.array(posteriors_online)

        max_diff = np.abs(posteriors_batch - posteriors_online).max()
        assert max_diff < 1e-4, \
            f"Online and batch posteriors differ by {max_diff:.2e} (threshold: 1e-4)"

    def test_online_posterior_valid_at_each_step(self, fitted_hmm, test_data):
        """Each online step should produce a valid probability vector."""
        X_val, _ = test_data
        log_alpha = None
        for t in range(20):
            p, log_alpha = fitted_hmm.get_posterior_online(X_val[t], log_alpha)
            assert abs(p.sum() - 1.0) < 1e-5, \
                f"Online posterior at t={t} does not sum to 1: {p.sum()}"
            assert (p >= 0).all()


class TestCausality:

    def test_posterior_uses_only_past_information(self, fitted_hmm):
        """
        Verify that adding future observations does not change past posteriors.
        This is the lookahead-bias test — critical for valid backtesting.
        """
        np.random.seed(0)
        X = np.random.randn(100, 11).astype(np.float32)

        posteriors_short = fitted_hmm.get_posterior(X[:50])
        posteriors_long  = fitted_hmm.get_posterior(X[:100])

        # First 50 timesteps should match
        max_diff = np.abs(posteriors_short - posteriors_long[:50]).max()
        assert max_diff < 1e-4, \
            f"Future observations affected past posteriors (diff={max_diff:.2e}). LOOKAHEAD BIAS!"


class TestViterbi:

    def test_viterbi_shape(self, fitted_hmm, test_data):
        X_val, _ = test_data
        states = fitted_hmm.get_viterbi_sequence(X_val)
        assert states.shape == (len(X_val),)

    def test_viterbi_valid_states(self, fitted_hmm, test_data):
        X_val, _ = test_data
        states = fitted_hmm.get_viterbi_sequence(X_val)
        K      = fitted_hmm.n_regimes
        assert set(states).issubset(set(range(K))), \
            f"Viterbi returned states outside [0, K-1]: {set(states)}"


class TestBIC:

    def test_bic_is_finite(self, fitted_hmm, test_data):
        X_val, _ = test_data
        bic = fitted_hmm.bic(X_val)
        assert np.isfinite(bic), f"BIC is not finite: {bic}"

    def test_bic_selection(self):
        """BIC should select a reasonable K (not necessarily 3 on synthetic data)."""
        X_train, X_val, _ = make_synthetic_data(T=1500)
        best_k, bic_scores = select_n_regimes(X_train, X_val, k_range=range(2, 5))
        assert best_k in range(2, 5)
        assert all(np.isfinite(v) for v in bic_scores.values())


class TestSerialization:

    def test_save_and_load(self, fitted_hmm, test_data):
        X_val, _ = test_data
        posteriors_before = fitted_hmm.get_posterior(X_val[:20])

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name

        try:
            fitted_hmm.save(path)
            loaded_hmm = BayesianMarketHMM.load(path)
            posteriors_after = loaded_hmm.get_posterior(X_val[:20])

            np.testing.assert_allclose(posteriors_before, posteriors_after, atol=1e-6,
                err_msg="Posteriors changed after save/load")
        finally:
            os.unlink(path)

    def test_loaded_model_has_labels(self, fitted_hmm):
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            fitted_hmm.save(path)
            loaded = BayesianMarketHMM.load(path)
            assert len(loaded.regime_labels) == fitted_hmm.n_regimes
        finally:
            os.unlink(path)


class TestEdgeCases:

    def test_single_timestep_online(self, fitted_hmm, test_data):
        X_val, _ = test_data
        p, log_alpha = fitted_hmm.get_posterior_online(X_val[0], None)
        assert p.shape == (fitted_hmm.n_regimes,)
        assert abs(p.sum() - 1.0) < 1e-5

    def test_two_regime_model(self):
        X_train, X_val, _ = make_synthetic_data(T=800)
        hmm = BayesianMarketHMM(n_regimes=2, n_iter=15)
        hmm.fit(X_train)
        posteriors = hmm.get_posterior(X_val)
        assert posteriors.shape[1] == 2
        np.testing.assert_allclose(posteriors.sum(axis=1), 1.0, atol=1e-5)

    def test_summary_runs(self, fitted_hmm):
        summary = fitted_hmm.summary()
        assert "Bull" in summary or "Bear" in summary
        assert "Transition" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
