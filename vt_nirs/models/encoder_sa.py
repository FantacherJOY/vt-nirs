

import torch
import torch.nn as nn

from .encoder import TemporalTransformerEncoder, PositionalEncoding


class SurvivalAttentionGate(nn.Module):

    def __init__(self, d_model, hidden_dim=64):
        super().__init__()

        self.gate_net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, d_model),
            nn.Sigmoid(),
        )

    def forward(self, emb):
        gate = self.gate_net(emb)
        emb_survival = emb * gate
        emb_vfd = emb * (1.0 - gate)

        return gate, emb_survival, emb_vfd


class SurvivalAwareTransformerEncoder(nn.Module):

    def __init__(
        self,
        n_covariates,
        d_model=128,
        n_heads=4,
        n_layers=4,
        d_ff=256,
        dropout=0.1,
        max_len=500,
        gate_hidden=64,
    ):
        super().__init__()

        self.base_encoder = TemporalTransformerEncoder(
            n_covariates=n_covariates,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            max_len=max_len,
        )

        self.survival_gate = SurvivalAttentionGate(d_model, gate_hidden)

        self.d_model = d_model

    def forward(self, x, pad_mask=None):
    
        emb, attn_out = self.base_encoder(x, pad_mask)

        gate, emb_survival, emb_vfd = self.survival_gate(emb)

        return emb, emb_survival, emb_vfd, gate, attn_out
