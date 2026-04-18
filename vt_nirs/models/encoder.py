
import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):

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
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TemporalTransformerEncoder(nn.Module):

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
