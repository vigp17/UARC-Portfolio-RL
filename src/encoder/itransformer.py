"""
itransformer.py
---------------
iTransformer encoder for multi-asset return dynamics.

Standard transformers treat each TIMESTEP as a token — attention runs
across time. iTransformer (Liu et al., 2024) inverts this: each ASSET
is a token, and attention runs across assets. This is the right inductive
bias for portfolio management because cross-asset correlations matter more
than individual temporal patterns.

Architecture role:
    60-day price history (5 assets) -> [THIS MODULE] -> 64-dim embedding
                                                              |
                                                    (concatenated with
                                                     HMM posterior)
                                                              |
                                                        IQN Agent

Input shape:  (batch, n_assets, lookback)   e.g. (256, 5, 60)
Output shape: (batch, d_model)              e.g. (256, 64)

Reference: Liu et al. (2024) "iTransformer: Inverted Transformers Are
           Effective for Time Series Forecasting" — ICLR 2024
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class AssetEmbedding(nn.Module):
    """
    Projects each asset's lookback window into d_model space.

    Each asset is a token with `lookback` features (its return history).
    This linear projection maps (lookback,) -> (d_model,) per asset,
    equivalent to the patch embedding in vision transformers.
    """

    def __init__(self, lookback: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.projection = nn.Linear(lookback, d_model)
        self.norm       = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, n_assets, lookback)

        Returns
        -------
        torch.Tensor
            Shape (batch, n_assets, d_model)
        """
        return self.dropout(self.norm(self.projection(x)))


class InvertedMultiHeadAttention(nn.Module):
    """
    Multi-head self-attention across the ASSET dimension.

    Standard MHA is applied but the sequence dimension is assets, not time.
    Each asset attends to all other assets, learning which assets are
    informationally relevant to each other (e.g. SPY-QQQ correlation,
    TLT-SPY negative correlation in risk-off periods).
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale   = math.sqrt(self.d_head)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, n_assets, d_model) — assets are the sequence
        mask : Optional[torch.Tensor]
            Shape (n_assets, n_assets) — optional attention mask

        Returns
        -------
        torch.Tensor
            Shape (batch, n_assets, d_model)
        """
        B, N, D = x.shape

        # Project to Q, K, V
        Q = self.W_q(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_k(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        # Shape: (B, n_heads, N, d_head)

        # Scaled dot-product attention across assets
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        # Shape: (B, n_heads, N, N)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = self.dropout(F.softmax(scores, dim=-1))

        # Aggregate values
        out = torch.matmul(attn_weights, V)
        # Shape: (B, n_heads, N, d_head)

        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.W_o(out)


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network applied independently per asset.
    Standard transformer FFN: Linear -> GELU -> Dropout -> Linear
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class iTransformerBlock(nn.Module):
    """
    Single iTransformer layer:
        LayerNorm -> InvertedMHA -> Residual
        LayerNorm -> FFN          -> Residual
    Pre-norm formulation (more stable training than post-norm).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn  = InvertedMultiHeadAttention(d_model, n_heads, dropout)
        self.ff    = FeedForward(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm + residual for attention
        x = x + self.attn(self.norm1(x))
        # Pre-norm + residual for FFN
        x = x + self.ff(self.norm2(x))
        return x


class iTransformerEncoder(nn.Module):
    """
    Full iTransformer encoder for multi-asset portfolio management.

    Processes N assets x T lookback window into a single d_model-dimensional
    embedding vector that captures both per-asset temporal patterns and
    cross-asset correlations.

    Parameters
    ----------
    n_assets : int
        Number of assets in the portfolio (default: 5).
    lookback : int
        Number of past trading days used as input (default: 60).
    d_model : int
        Internal embedding dimension (default: 64).
    n_heads : int
        Number of attention heads (default: 4).
    n_layers : int
        Number of iTransformer blocks (default: 2).
    d_ff : int
        Feed-forward hidden dimension (default: 256 = 4 * d_model).
    dropout : float
        Dropout rate (default: 0.1).
    pooling : str
        How to aggregate asset embeddings into a single vector.
        'mean' = average pooling, 'cls' = learnable CLS token.

    Input
    -----
    x : torch.Tensor
        Shape (batch, n_assets, lookback) — normalized return features.

    Output
    ------
    embedding : torch.Tensor
        Shape (batch, d_model) — fed to IQN agent alongside HMM posterior.
    """

    def __init__(
        self,
        n_assets:  int   = 5,
        lookback:  int   = 60,
        d_model:   int   = 64,
        n_heads:   int   = 4,
        n_layers:  int   = 2,
        d_ff:      int   = 256,
        dropout:   float = 0.1,
        pooling:   str   = "mean",
    ):
        super().__init__()

        self.n_assets = n_assets
        self.lookback = lookback
        self.d_model  = d_model
        self.pooling  = pooling

        # Asset embedding: projects each asset's lookback window to d_model
        self.asset_embedding = AssetEmbedding(lookback, d_model, dropout)

        # Optional CLS token for 'cls' pooling
        if pooling == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Stack of iTransformer blocks
        self.blocks = nn.ModuleList([
            iTransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Final layer norm
        self.norm = nn.LayerNorm(d_model)

        # Output projection (optional — keeps d_model consistent)
        self.output_proj = nn.Linear(d_model, d_model)

        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, n_assets, lookback)
            Each asset's lookback-day return history.

        Returns
        -------
        torch.Tensor
            Shape (batch, d_model)
        """
        B = x.shape[0]

        # Step 1: Embed each asset's time series -> (B, N, d_model)
        tokens = self.asset_embedding(x)

        # Step 2: Prepend CLS token if using CLS pooling
        if self.pooling == "cls":
            cls = self.cls_token.expand(B, -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)

        # Step 3: Apply iTransformer blocks (attention across assets)
        for block in self.blocks:
            tokens = block(tokens)

        tokens = self.norm(tokens)

        # Step 4: Pool across asset dimension -> (B, d_model)
        if self.pooling == "cls":
            embedding = tokens[:, 0, :]          # CLS token
        else:
            embedding = tokens.mean(dim=1)       # Mean pooling

        # Step 5: Output projection
        embedding = self.output_proj(embedding)

        return embedding

    def get_attention_weights(self, x: torch.Tensor) -> list:
        """
        Returns attention weight matrices for visualization.
        Useful for paper figures showing which asset-pairs attend to each other.

        Returns
        -------
        list of torch.Tensor
            Each element: shape (batch, n_heads, n_assets, n_assets)
            One per transformer block.
        """
        B = x.shape[0]
        tokens = self.asset_embedding(x)

        if self.pooling == "cls":
            cls = self.cls_token.expand(B, -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)

        attn_weights = []
        for block in self.blocks:
            # Extract attention weights from this block
            normed = block.norm1(tokens)
            Q = block.attn.W_q(normed)
            K = block.attn.W_k(normed)
            N, D = Q.shape[1], Q.shape[2]
            n_heads = block.attn.n_heads
            d_head  = D // n_heads

            Q = Q.view(B, N, n_heads, d_head).transpose(1, 2)
            K = K.view(B, N, n_heads, d_head).transpose(1, 2)
            scores = torch.matmul(Q, K.transpose(-2, -1)) / block.attn.scale
            weights = F.softmax(scores, dim=-1)
            attn_weights.append(weights.detach())

            # Continue forward pass
            tokens = tokens + block.attn(block.norm1(tokens))
            tokens = tokens + block.ff(block.norm2(tokens))

        return attn_weights

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)