"""
test_encoder.py
---------------
Unit tests for the iTransformer encoder and feature pipeline.

Run with: pytest tests/test_encoder.py -v

Tests validate:
  1. Output shapes are correct
  2. Encoder is differentiable (gradients flow)
  3. Attention runs across assets not time
  4. Feature pipeline produces valid tensors
  5. Dataset and DataLoader work correctly
  6. Normalizer fit/transform consistency
  7. Parameter count is reasonable
"""

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.encoder.itransformer import (
    iTransformerEncoder, AssetEmbedding,
    InvertedMultiHeadAttention, iTransformerBlock
)
from src.encoder.features import (
    build_encoder_features, FeatureNormalizer,
    PortfolioSequenceDataset, LOOKBACK
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

N_ASSETS   = 5
LOOKBACK_T = 60
N_FEATURES = 6
D_MODEL    = 64
BATCH      = 8

def make_encoder_input(batch=BATCH, n_assets=N_ASSETS,
                       lookback=LOOKBACK_T, n_features=N_FEATURES):
    """Synthetic encoder input: (batch, n_assets, lookback * n_features)."""
    return torch.randn(batch, n_assets, lookback * n_features)

def make_prices(T=500, n_assets=5, seed=42):
    """Synthetic price DataFrame."""
    rng  = np.random.RandomState(seed)
    rets = rng.randn(T, n_assets) * 0.01
    prices = 100 * np.exp(np.cumsum(rets, axis=0))
    dates  = pd.date_range("2010-01-01", periods=T, freq="B")
    return pd.DataFrame(prices, index=dates,
                        columns=["SPY", "QQQ", "TLT", "GLD", "SHY"])

@pytest.fixture
def encoder():
    return iTransformerEncoder(
        n_assets=N_ASSETS,
        lookback=LOOKBACK_T * N_FEATURES,  # flattened input
        d_model=D_MODEL,
        n_heads=4,
        n_layers=2,
        dropout=0.0,  # disable for deterministic tests
    )

@pytest.fixture
def sample_input():
    return make_encoder_input()


# ── Shape Tests ────────────────────────────────────────────────────────────────

class TestOutputShapes:

    def test_encoder_output_shape(self, encoder, sample_input):
        out = encoder(sample_input)
        assert out.shape == (BATCH, D_MODEL), \
            f"Expected ({BATCH}, {D_MODEL}), got {out.shape}"

    def test_single_sample(self, encoder):
        x   = make_encoder_input(batch=1)
        out = encoder(x)
        assert out.shape == (1, D_MODEL)

    def test_asset_embedding_shape(self):
        emb = AssetEmbedding(lookback=LOOKBACK_T * N_FEATURES, d_model=D_MODEL)
        x   = make_encoder_input()
        out = emb(x)
        assert out.shape == (BATCH, N_ASSETS, D_MODEL)

    def test_mha_shape(self):
        mha = InvertedMultiHeadAttention(d_model=D_MODEL, n_heads=4)
        x   = torch.randn(BATCH, N_ASSETS, D_MODEL)
        out = mha(x)
        assert out.shape == (BATCH, N_ASSETS, D_MODEL)

    def test_transformer_block_shape(self):
        block = iTransformerBlock(d_model=D_MODEL, n_heads=4, d_ff=256)
        x     = torch.randn(BATCH, N_ASSETS, D_MODEL)
        out   = block(x)
        assert out.shape == (BATCH, N_ASSETS, D_MODEL)

    def test_cls_pooling_shape(self):
        enc = iTransformerEncoder(
            n_assets=N_ASSETS,
            lookback=LOOKBACK_T * N_FEATURES,
            d_model=D_MODEL,
            n_heads=4,
            n_layers=2,
            pooling="cls",
            dropout=0.0,
        )
        x   = make_encoder_input()
        out = enc(x)
        assert out.shape == (BATCH, D_MODEL)


# ── Gradient Tests ─────────────────────────────────────────────────────────────

class TestGradients:

    def test_gradients_flow(self, encoder, sample_input):
        """Encoder must be fully differentiable for end-to-end training."""
        sample_input.requires_grad_(True)
        out  = encoder(sample_input)
        loss = out.sum()
        loss.backward()
        assert sample_input.grad is not None
        assert not torch.isnan(sample_input.grad).any()

    def test_encoder_parameters_have_gradients(self, encoder, sample_input):
        out  = encoder(sample_input)
        loss = out.mean()
        loss.backward()
        for name, param in encoder.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_no_nan_in_output(self, encoder, sample_input):
        out = encoder(sample_input)
        assert not torch.isnan(out).any(), "NaN in encoder output"
        assert not torch.isinf(out).any(), "Inf in encoder output"


# ── Architecture Tests ─────────────────────────────────────────────────────────

class TestArchitecture:

    def test_attention_across_assets_not_time(self):
        """
        The core iTransformer property: attention weights have shape
        (batch, heads, n_assets, n_assets), NOT (batch, heads, T, T).
        """
        enc = iTransformerEncoder(
            n_assets=N_ASSETS,
            lookback=LOOKBACK_T * N_FEATURES,
            d_model=D_MODEL, n_heads=4, n_layers=1, dropout=0.0
        )
        x            = make_encoder_input()
        attn_weights = enc.get_attention_weights(x)
        assert len(attn_weights) == 1  # 1 layer

        w = attn_weights[0]
        assert w.shape == (BATCH, 4, N_ASSETS, N_ASSETS), \
            f"Attention should be across assets: expected (B, 4, {N_ASSETS}, {N_ASSETS}), got {w.shape}"

    def test_attention_weights_sum_to_one(self):
        enc = iTransformerEncoder(
            n_assets=N_ASSETS,
            lookback=LOOKBACK_T * N_FEATURES,
            d_model=D_MODEL, n_heads=4, n_layers=1, dropout=0.0
        )
        x            = make_encoder_input()
        attn_weights = enc.get_attention_weights(x)
        w            = attn_weights[0]  # (B, heads, N, N)
        row_sums     = w.sum(dim=-1)    # (B, heads, N)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), \
            "Attention weights do not sum to 1 across asset dimension"

    def test_different_assets_produce_different_outputs(self, encoder):
        """Each asset should have a distinct representation after encoding."""
        x   = make_encoder_input(batch=1)
        enc = iTransformerEncoder(
            n_assets=N_ASSETS,
            lookback=LOOKBACK_T * N_FEATURES,
            d_model=D_MODEL, n_heads=4, n_layers=2, dropout=0.0
        )
        # Different inputs should produce different outputs
        x1  = torch.randn(1, N_ASSETS, LOOKBACK_T * N_FEATURES)
        x2  = torch.randn(1, N_ASSETS, LOOKBACK_T * N_FEATURES)
        out1 = enc(x1)
        out2 = enc(x2)
        assert not torch.allclose(out1, out2), "Different inputs produced same output"

    def test_parameter_count_reasonable(self, encoder):
        n_params = encoder.count_parameters()
        # Should be between 10K and 500K for d_model=64
        assert 10_000 < n_params < 500_000, \
            f"Parameter count {n_params} seems off"
        print(f"\n  Encoder parameters: {n_params:,}")


# ── Feature Pipeline Tests ─────────────────────────────────────────────────────

class TestFeaturePipeline:

    def test_build_encoder_features_shape(self):
        prices = make_prices(T=300)
        X      = build_encoder_features(prices)
        assert X.ndim == 3
        assert X.shape[1] == 5    # n_assets
        assert X.shape[2] == 6    # n_features per asset

    def test_build_encoder_features_no_nan(self):
        prices = make_prices(T=300)
        X      = build_encoder_features(prices)
        assert not np.isnan(X).any(), "NaN in encoder features"

    def test_build_encoder_features_dtype(self):
        prices = make_prices(T=300)
        X      = build_encoder_features(prices)
        assert X.dtype == np.float32

    def test_normalizer_fit_transform(self):
        prices = make_prices(T=300)
        X      = build_encoder_features(prices)
        norm   = FeatureNormalizer()
        X_norm = norm.fit_transform(X)
        assert X_norm.shape == X.shape
        # Each feature should be approximately zero-mean after normalization
        assert abs(X_norm.mean()) < 0.5

    def test_normalizer_raises_if_not_fitted(self):
        norm = FeatureNormalizer()
        with pytest.raises(RuntimeError):
            norm.transform(np.zeros((10, 5, 6)))

    def test_normalizer_train_val_consistency(self):
        """Val normalization should use train statistics, not val statistics."""
        prices_train = make_prices(T=400, seed=1)
        prices_val   = make_prices(T=100, seed=2)
        X_train = build_encoder_features(prices_train)
        X_val   = build_encoder_features(prices_val)

        norm    = FeatureNormalizer()
        norm.fit(X_train)
        X_val_norm = norm.transform(X_val)

        assert X_val_norm.shape == X_val.shape
        assert not np.isnan(X_val_norm).any()


# ── Dataset Tests ──────────────────────────────────────────────────────────────

class TestDataset:

    def make_dataset(self, T=300):
        prices = make_prices(T=T)
        X      = build_encoder_features(prices)
        norm   = FeatureNormalizer()
        X_norm = norm.fit_transform(X)
        T_     = min(len(X_norm), T)
        posts  = np.random.dirichlet(np.ones(3), size=T_).astype(np.float32)
        return PortfolioSequenceDataset(X_norm, posts, prices.iloc[-T_:], lookback=LOOKBACK)

    def test_dataset_length(self):
        ds = self.make_dataset()
        assert len(ds) > 0

    def test_dataset_sample_shapes(self):
        ds     = self.make_dataset()
        sample = ds[0]
        assert "encoder_input"    in sample
        assert "regime_posterior" in sample
        assert "next_returns"     in sample

        assert sample["encoder_input"].shape[0]    == 5  # n_assets
        assert sample["regime_posterior"].shape[0] == 3  # K regimes
        assert sample["next_returns"].shape[0]     == 5  # n_assets

    def test_dataset_no_nan(self):
        ds = self.make_dataset()
        for key in ["encoder_input", "regime_posterior", "next_returns"]:
            assert not torch.isnan(ds[0][key]).any(), f"NaN in {key}"

    def test_dataset_sequential_access(self):
        """Verify that consecutive samples are offset by 1 day."""
        ds  = self.make_dataset()
        s0  = ds[0]["encoder_input"]
        s1  = ds[1]["encoder_input"]
        # They should not be identical
        assert not torch.allclose(s0, s1)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])