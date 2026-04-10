"""
hmm_model.py
------------
Bayesian Hidden Markov Model with online posterior inference.

This is the regime module for the UARC project. It differs from
standard HMM usage in one critical way: instead of returning hard
regime labels via Viterbi (which requires the full sequence and
introduces lookahead bias), it returns the FILTERING POSTERIOR
p(z_t | x_{1:t}) via the forward algorithm.

This K-dimensional probability vector becomes the regime conditioning
signal fed into the IQN agent at each timestep.

Architecture role:
    Market data -> [THIS MODULE] -> regime posterior p(z_t | x_{1:t})
                                         |
                                    (concatenated with iTransformer output)
                                         |
                                    IQN Agent -> portfolio weights
"""

import logging
import numpy as np
import pickle
from pathlib import Path
from typing import Optional, Tuple

from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class BayesianMarketHMM:
    """
    Gaussian HMM with Bayesian posterior inference via the forward algorithm.

    Parameters
    ----------
    n_regimes : int
        Number of hidden market regimes (default: 3 — Bull, Bear, Sideways).
    n_iter : int
        Max Baum-Welch iterations for fitting.
    random_state : int
        Reproducibility seed.
    covariance_type : str
        Emission covariance structure: 'full', 'diag', 'tied', 'spherical'.
        'diag' recommended for financial returns (fewer params, less overfitting).

    Attributes
    ----------
    model : GaussianHMM
        Fitted hmmlearn model.
    scaler : StandardScaler
        Fitted on training data; applied before all inference.
    regime_labels : dict
        Maps integer state index -> human-readable label (set post-fit).
    """

    def __init__(
        self,
        n_regimes: int = 3,
        n_iter: int = 100,
        random_state: int = 42,
        covariance_type: str = "diag",
    ):
        self.n_regimes       = n_regimes
        self.n_iter          = n_iter
        self.random_state    = random_state
        self.covariance_type = covariance_type

        self.model         = None
        self.scaler        = StandardScaler()
        self.regime_labels = {}
        self._is_fitted    = False

    # ── Fitting ────────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray) -> "BayesianMarketHMM":
        """
        Fit the HMM via Baum-Welch (EM) on training data.

        Parameters
        ----------
        X : np.ndarray
            Shape (T, D) — feature matrix from data_loader.get_hmm_features().
            Typically D=11: 2 features per asset (return + vol) + pairwise corr.

        Returns
        -------
        self
        """
        logger.info(f"Fitting HMM: {self.n_regimes} regimes, {X.shape[1]} features, {len(X)} timesteps")

        # Standardize — HMM emission distributions are sensitive to scale
        X_scaled = self.scaler.fit_transform(X)

        self.model = GaussianHMM(
            n_components=self.n_regimes,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state,
            verbose=False,
            tol=1e-4,
            init_params="",
            params="stmc",
        )
        self._initialize_model_params(X_scaled)
        self.model.fit(X_scaled)

        self._is_fitted = True
        logger.info(f"HMM fitted. Log-likelihood: {self.model.score(X_scaled):.2f}")

        # Auto-label regimes based on emission means
        self._label_regimes()

        return self

    def _initialize_model_params(self, X_scaled: np.ndarray):
        """
        Seed HMM parameters without sklearn KMeans.

        This avoids a native crash observed in the test environment during the
        default hmmlearn initialization path while still giving EM a sensible
        starting point.
        """
        rng = np.random.RandomState(self.random_state)
        T, D = X_scaled.shape
        K = self.n_regimes

        self.model.startprob_ = np.full(K, 1.0 / K, dtype=np.float64)

        transmat = np.full((K, K), 1.0 / K, dtype=np.float64)
        np.fill_diagonal(transmat, 0.8)
        off_diag = (1.0 - 0.8) / max(K - 1, 1)
        transmat[~np.eye(K, dtype=bool)] = off_diag
        self.model.transmat_ = transmat

        if T >= K:
            mean_idx = np.linspace(0, T - 1, K, dtype=int)
        else:
            mean_idx = rng.choice(T, size=K, replace=True)
        jitter = rng.normal(scale=1e-2, size=(K, D))
        self.model.means_ = X_scaled[mean_idx].astype(np.float64) + jitter

        empirical_var = np.var(X_scaled, axis=0) + 1e-3
        if self.covariance_type == "diag":
            self.model.covars_ = np.tile(empirical_var, (K, 1)).astype(np.float64)
        elif self.covariance_type == "full":
            base_cov = np.diag(empirical_var)
            self.model.covars_ = np.tile(base_cov[None, :, :], (K, 1, 1)).astype(np.float64)
        elif self.covariance_type == "tied":
            self.model.covars_ = np.diag(empirical_var).astype(np.float64)
        else:
            self.model.covars_ = np.full(K, float(empirical_var.mean()), dtype=np.float64)

    def _label_regimes(self):
        """
        Assigns human-readable labels to regimes based on emission statistics.

        Heuristic: sort regimes by mean return of the first asset (SPY log return).
        - Highest mean return -> Bull
        - Lowest mean return  -> Bear
        - Middle              -> Sideways / High-Vol
        """
        # Emission means in original (unscaled) space
        means_scaled = self.model.means_  # shape (K, D)
        means_orig   = self.scaler.inverse_transform(means_scaled)

        # Column 0 = SPY log return (from data_loader feature ordering)
        spy_ret_per_regime = means_orig[:, 0]
        sorted_idx         = np.argsort(spy_ret_per_regime)

        label_map = {
            sorted_idx[0]: "Bear",
            sorted_idx[-1]: "Bull",
        }
        # Middle regime(s) = Sideways
        for i, idx in enumerate(sorted_idx[1:-1]):
            label_map[idx] = f"Sideways_{i}"

        self.regime_labels = label_map
        logger.info(f"Regime labels: {label_map}")
        logger.info(f"Mean SPY returns per regime: {spy_ret_per_regime}")

    # ── Posterior Inference ────────────────────────────────────────────────────

    def get_posterior(self, X: np.ndarray) -> np.ndarray:
        """
        Computes the filtering posterior p(z_t | x_{1:t}) for all t.

        This is the KEY METHOD. It runs the forward algorithm causally —
        each timestep only uses information available up to that point.
        No lookahead bias. Suitable for use in a trading simulation.

        Parameters
        ----------
        X : np.ndarray
            Shape (T, D) — feature matrix.

        Returns
        -------
        posteriors : np.ndarray
            Shape (T, K) — each row is a probability vector over K regimes.
            posteriors[t, k] = P(z_t = k | x_{1:t})

        Example
        -------
        >>> posteriors = hmm.get_posterior(X_test)
        >>> posteriors[100]   # e.g., [0.72, 0.18, 0.10]
        >>> # -> 72% Bull, 18% Bear, 10% Sideways at timestep 100
        """
        self._check_fitted()
        X_scaled   = self.scaler.transform(X)
        posteriors = self._forward_algorithm(X_scaled)
        return posteriors

    def _forward_algorithm(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        Manual implementation of the forward (filtering) algorithm.

        Uses log-space computation to avoid numerical underflow,
        which is critical for long financial time series (T ~ 4500 days).

        Parameters
        ----------
        X_scaled : np.ndarray
            Shape (T, D) — standardized features.

        Returns
        -------
        posteriors : np.ndarray
            Shape (T, K) — filtered posterior at each timestep.
        """
        T = len(X_scaled)
        K = self.n_regimes

        # Pre-compute log emission probabilities: log P(x_t | z_t = k)
        log_emission = self._log_emission_probs(X_scaled)  # shape (T, K)

        # Log transition matrix
        log_A  = np.log(self.model.transmat_ + 1e-300)      # shape (K, K)
        log_pi = np.log(self.model.startprob_ + 1e-300)     # shape (K,)

        # Forward pass in log space
        log_alpha = np.zeros((T, K))

        # Initialization: t=0
        log_alpha[0] = log_pi + log_emission[0]

        # Recursion: t=1..T-1
        for t in range(1, T):
            for j in range(K):
                # log sum_k [ alpha_{t-1}(k) * A_{k,j} ]
                log_alpha[t, j] = _log_sum_exp(log_alpha[t-1] + log_A[:, j]) \
                                  + log_emission[t, j]

        # Convert to probabilities (normalize each row)
        # log_alpha[t] - log_sum_exp(log_alpha[t]) = log posterior
        log_posteriors = log_alpha - _log_sum_exp(log_alpha, axis=1, keepdims=True)
        posteriors     = np.exp(log_posteriors)

        # Numerical safety: ensure rows sum to 1
        posteriors = posteriors / posteriors.sum(axis=1, keepdims=True)

        return posteriors.astype(np.float32)

    def _log_emission_probs(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        Computes log P(x_t | z_t = k) for all t, k using Gaussian emissions.

        Returns
        -------
        np.ndarray
            Shape (T, K).
        """
        from scipy.stats import multivariate_normal

        T = len(X_scaled)
        K = self.n_regimes
        log_probs = np.zeros((T, K))

        for k in range(K):
            mean = self.model.means_[k]

            if self.covariance_type == "diag":
                cov = np.diag(self.model.covars_[k])
            elif self.covariance_type == "full":
                cov = self.model.covars_[k]
            elif self.covariance_type == "tied":
                cov = self.model.covars_
            else:  # spherical
                cov = np.eye(mean.shape[0]) * self.model.covars_[k]

            try:
                rv = multivariate_normal(mean=mean, cov=cov, allow_singular=True)
                log_probs[:, k] = rv.logpdf(X_scaled)
            except Exception as e:
                logger.warning(f"Emission prob error for regime {k}: {e}. Using -inf.")
                log_probs[:, k] = -1e10

        return log_probs

    # ── Online Inference ───────────────────────────────────────────────────────

    def get_posterior_online(self, x_t: np.ndarray, prev_log_alpha: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Single-step online posterior update. Used during live trading simulation.

        Call this once per new trading day. Maintains the forward variable
        across calls without reprocessing the full history.

        Parameters
        ----------
        x_t : np.ndarray
            Shape (D,) — feature vector for the current timestep.
        prev_log_alpha : np.ndarray or None
            Shape (K,) — log forward variable from previous step.
            Pass None to initialize (first timestep).

        Returns
        -------
        posterior : np.ndarray
            Shape (K,) — P(z_t | x_{1:t}).
        log_alpha : np.ndarray
            Shape (K,) — updated log forward variable (pass back next call).

        Example
        -------
        >>> log_alpha = None
        >>> for t in range(T):
        ...     posterior, log_alpha = hmm.get_posterior_online(X[t], log_alpha)
        ...     agent_input = np.concatenate([encoder_output[t], posterior])
        """
        self._check_fitted()

        x_scaled = self.scaler.transform(x_t.reshape(1, -1))[0]
        log_A    = np.log(self.model.transmat_ + 1e-300)
        log_pi   = np.log(self.model.startprob_ + 1e-300)

        # Emission: log P(x_t | z_t = k)
        log_emit = self._log_emission_probs(x_scaled.reshape(1, -1))[0]

        if prev_log_alpha is None:
            # First step: use initial distribution
            log_alpha = log_pi + log_emit
        else:
            # Recursive update
            K         = self.n_regimes
            log_alpha = np.array([
                _log_sum_exp(prev_log_alpha + log_A[:, j]) + log_emit[j]
                for j in range(K)
            ])

        # Normalize to get posterior
        log_posterior = log_alpha - _log_sum_exp(log_alpha)
        posterior     = np.exp(log_posterior).astype(np.float32)

        return posterior, log_alpha

    # ── Viterbi (for analysis only — NOT used in trading) ─────────────────────

    def get_viterbi_sequence(self, X: np.ndarray) -> np.ndarray:
        """
        Returns the MAP state sequence via Viterbi. FOR ANALYSIS ONLY.

        *** DO NOT use this in the trading loop — it requires the full
        sequence and is NOT causal. Using Viterbi for trading creates
        lookahead bias and will invalidate your backtest results. ***

        Use this only for:
          - Visualizing regime segments on historical data
          - Validating that regimes are economically meaningful
          - Generating the post-hoc regime labels in your paper figures

        Returns
        -------
        np.ndarray
            Shape (T,) — integer regime index per timestep (0 to K-1).
        """
        self._check_fitted()
        X_scaled = self.scaler.transform(X)
        states   = self.model.predict(X_scaled)
        return states

    # ── Evaluation ────────────────────────────────────────────────────────────

    def score(self, X: np.ndarray) -> float:
        """
        Returns log-likelihood of X under the fitted model.
        Use for model selection (BIC/AIC across different K values).
        """
        self._check_fitted()
        X_scaled = self.scaler.transform(X)
        return self.model.score(X_scaled)

    def bic(self, X: np.ndarray) -> float:
        """
        Bayesian Information Criterion for model selection.
        Lower BIC = better model (penalizes complexity).
        Use this to select optimal n_regimes (typically K=2,3,4).
        """
        n_params = self._count_params()
        T        = len(X)
        ll       = self.score(X) * T  # score() returns per-sample LL
        return -2 * ll + n_params * np.log(T)

    def _count_params(self) -> int:
        """Count free parameters for BIC computation."""
        K = self.n_regimes
        D = self.model.means_.shape[1]

        transition_params = K * (K - 1)  # each row sums to 1
        initial_params    = K - 1
        mean_params       = K * D

        if self.covariance_type == "diag":
            cov_params = K * D
        elif self.covariance_type == "full":
            cov_params = K * D * (D + 1) // 2
        elif self.covariance_type == "tied":
            cov_params = D * (D + 1) // 2
        else:  # spherical
            cov_params = K

        return transition_params + initial_params + mean_params + cov_params

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str):
        """Save fitted model to disk."""
        self._check_fitted()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "BayesianMarketHMM":
        """Load a saved model from disk."""
        with open(path, "rb") as f:
            model = pickle.load(f)
        logger.info(f"Model loaded from {path}")
        return model

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted. Call .fit(X) first.")

    def summary(self) -> str:
        """Print model summary."""
        self._check_fitted()
        lines = [
            f"\n{'='*50}",
            f"BayesianMarketHMM Summary",
            f"{'='*50}",
            f"  Regimes (K)      : {self.n_regimes}",
            f"  Covariance type  : {self.covariance_type}",
            f"  Regime labels    : {self.regime_labels}",
            f"\n  Transition matrix (A):",
        ]
        for i, row in enumerate(self.model.transmat_):
            label = self.regime_labels.get(i, f"State_{i}")
            probs = "  ".join([f"{p:.3f}" for p in row])
            lines.append(f"    {label:12s}: [{probs}]")

        lines.append(f"\n  Emission means (SPY log ret, SPY vol, ...):")
        means_orig = self.scaler.inverse_transform(self.model.means_)
        for i, mean in enumerate(means_orig):
            label = self.regime_labels.get(i, f"State_{i}")
            lines.append(f"    {label:12s}: SPY_ret={mean[0]:.4f}  SPY_vol={mean[1]:.4f}")

        lines.append(f"{'='*50}\n")
        return "\n".join(lines)


# ── Model Selection Helper ─────────────────────────────────────────────────────

def select_n_regimes(
    X_train: np.ndarray,
    X_val: np.ndarray,
    k_range: range = range(2, 6),
) -> Tuple[int, dict]:
    """
    Fits HMMs for each K in k_range and selects best K by validation BIC.

    Parameters
    ----------
    X_train : np.ndarray
        Training features for fitting.
    X_val : np.ndarray
        Validation features for BIC evaluation.
    k_range : range
        Values of K to try.

    Returns
    -------
    best_k : int
    results : dict
        BIC scores per K.

    Example
    -------
    >>> best_k, bic_scores = select_n_regimes(hmm_train, hmm_val)
    >>> print(f"Best K={best_k}  BIC scores: {bic_scores}")
    """
    results = {}
    logger.info("Running regime count selection via BIC...")

    for k in k_range:
        hmm = BayesianMarketHMM(n_regimes=k)
        hmm.fit(X_train)
        bic_val = hmm.bic(X_val)
        results[k] = bic_val
        logger.info(f"  K={k}  BIC={bic_val:.2f}")

    best_k = min(results, key=results.get)
    logger.info(f"Selected K={best_k} (lowest BIC={results[best_k]:.2f})")
    return best_k, results


# ── Log-sum-exp utilities ──────────────────────────────────────────────────────

def _log_sum_exp(log_probs: np.ndarray, axis: int = -1, keepdims: bool = False) -> np.ndarray:
    """
    Numerically stable log-sum-exp.
    Equivalent to log(sum(exp(log_probs))) but avoids overflow/underflow.
    """
    max_val = np.max(log_probs, axis=axis, keepdims=True)
    result  = np.log(np.sum(np.exp(log_probs - max_val), axis=axis, keepdims=keepdims)) \
              + np.squeeze(max_val, axis=axis) if not keepdims else \
              np.log(np.sum(np.exp(log_probs - max_val), axis=axis, keepdims=True)) + max_val
    return result


# ── Quick test / entry point ───────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    # Synthetic data test (runs without downloading real data)
    np.random.seed(42)
    T, D = 2000, 11

    # Simulate 3-regime data: Bull (high ret, low vol), Bear (low ret, high vol), Sideways
    regime_params = [
        {"mean": [0.001, 0.08] * 5 + [0.3],  "std": [0.005, 0.02] * 5 + [0.05]},   # Bull
        {"mean": [-0.002, 0.20] * 5 + [0.7], "std": [0.010, 0.05] * 5 + [0.08]},   # Bear
        {"mean": [0.0005, 0.12] * 5 + [0.5], "std": [0.007, 0.03] * 5 + [0.06]},   # Sideways
    ]
    # Generate switching data
    X = np.zeros((T, D))
    regime_seq = np.random.choice(3, size=T, p=[0.5, 0.25, 0.25])
    for t in range(T):
        r = regime_seq[t]
        X[t] = np.random.normal(regime_params[r]["mean"], regime_params[r]["std"])

    # Split
    X_train, X_val = X[:1600], X[1600:]

    # Fit
    hmm = BayesianMarketHMM(n_regimes=3)
    hmm.fit(X_train)

    # Get posteriors
    posteriors = hmm.get_posterior(X_val)
    print(f"\nPosterior shape: {posteriors.shape}")
    print(f"First 5 posteriors:\n{posteriors[:5]}")
    print(f"Row sums (should all be ~1.0): {posteriors[:5].sum(axis=1)}")

    # Online inference test
    print("\nOnline inference test:")
    log_alpha = None
    for t in range(5):
        p, log_alpha = hmm.get_posterior_online(X_val[t], log_alpha)
        print(f"  t={t}: posterior={p.round(3)}")

    # Compare online vs batch (should match)
    posteriors_batch  = hmm.get_posterior(X_val[:5])
    posteriors_online = []
    log_alpha = None
    for t in range(5):
        p, log_alpha = hmm.get_posterior_online(X_val[t], log_alpha)
        posteriors_online.append(p)
    posteriors_online = np.array(posteriors_online)

    max_diff = np.abs(posteriors_batch - posteriors_online).max()
    print(f"\nMax diff batch vs online: {max_diff:.6f}  (should be < 1e-5)")

    # Model summary
    print(hmm.summary())

    # BIC-based selection (small range for speed)
    best_k, bic_scores = select_n_regimes(X_train, X_val, k_range=range(2, 5))
    print(f"\nBest K={best_k}, BIC scores: {bic_scores}")
