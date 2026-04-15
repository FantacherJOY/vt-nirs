"""
encoder.py — Base Temporal Transformer Encoder
===============================================
Encodes multivariate ICU time series into fixed-dimensional patient embeddings.

This is the BASE encoder (analogous to graphspa/training/layer.py).
The novel survival-aware variant is in encoder_sa.py (analogous to layer_dev.py).

Architecture:
  Input: (batch, T, D) — T timesteps, D covariates
  → Sinusoidal positional encoding (Vaswani et al., NeurIPS 2017, Section 3.5)
  → TransformerEncoder (4 layers, 4 heads)
  → Temporal aggregation (mean over time)
  → Linear projection → (batch, d_model)

Design choices:
  - Sinusoidal positional encoding rather than learned, following Vaswani et al.
    NeurIPS 2017 (Section 3.5: "sinusoidal version... would allow the model
    to extrapolate to sequence lengths longer than the ones encountered during
    training"). We use this because ICU stays have variable observation windows.
    # Ref: Vaswani et al. NeurIPS 2017, Section 3.5, Eq. (3)-(4)

  - TransformerEncoder over LSTM because attention captures long-range
    dependencies across the 24h pre-treatment window without vanishing gradients.
    Justified by CLEF (Li et al., arXiv 2502.03569) and TIMING (Jang et al.,
    ICML 2025) which both use Transformer encoders for ICU time series.
    # Ref: CLEF model.py lines 46-78 use TransformerEncoder for EHR sequences
    # Ref: TIMING txai/models/encoders/transformer_simple.py uses TransformerMVTS

  - Mean temporal aggregation (not last-token) because ICU data has irregular
    sampling and padding; mean pooling is robust to sequence length variation.
    This follows graphspa (training/net.py lines 118-120) which uses
    AdaptiveAvgPool2d for temporal aggregation.
    # Ref: graphspa net.py lines 118-120: nn.AdaptiveAvgPool2d
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding from Vaswani et al. (NeurIPS 2017).

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    # Ref: Vaswani et al. NeurIPS 2017, Eq. (3)-(4)
    # Ref: TIMING models/layers.py PositionalEncodingTF uses same formulation
    """

    def __init__(self, d_model, max_len=500, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model) with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TemporalTransformerEncoder(nn.Module):
    """
    Base Transformer encoder for ICU multivariate time series.

    Maps (batch, T, n_covariates) → (batch, d_model) patient embedding.

    Architecture follows the standard Transformer encoder (Vaswani et al. 2017)
    adapted for continuous-valued time series input (no token embedding layer).
    Input projection replaces the token embedding.

    # Ref: Vaswani et al. NeurIPS 2017, Section 3.1 (Encoder architecture)
    # Ref: CLEF model.py lines 46-78 — same pattern of input_proj → PE → TransformerEncoder
    # Ref: TIMING txai/models/encoders/transformer_simple.py — TransformerMVTS

    Args:
        n_covariates: Number of input features (23 for VT-NIRS)
        d_model: Internal embedding dimension (default: 128)
        n_heads: Number of attention heads (default: 4)
        n_layers: Number of TransformerEncoderLayers (default: 4)
        d_ff: Feed-forward dimension (default: 256)
        dropout: Dropout rate (default: 0.1)
        max_len: Maximum sequence length for positional encoding (default: 500)
    """

    def __init__(
        self,
        n_covariates,
        d_model=128,
        n_heads=4,
        n_layers=4,
        d_ff=256,
        dropout=0.1,
        max_len=500,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_covariates = n_covariates

        self.input_proj = nn.Sequential(
            nn.Linear(n_covariates, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

        self.output_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x, pad_mask=None):
        """
        Args:
            x: (batch, T, n_covariates) — raw time series
            pad_mask: (batch, T) — True where padded (to be ignored)
        Returns:
            emb: (batch, d_model) — patient embedding
            attn_out: (batch, T, d_model) — per-timestep representations
        """
        h = self.input_proj(x)

        h = self.pos_encoder(h)

        attn_out = self.transformer(
            h,
            src_key_padding_mask=pad_mask,
        )

        if pad_mask is not None:
            valid_mask = (~pad_mask).unsqueeze(-1).float()
            emb = (attn_out * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1)
        else:
            emb = attn_out.mean(dim=1)

        emb = self.output_proj(emb)

        return emb, attn_out
