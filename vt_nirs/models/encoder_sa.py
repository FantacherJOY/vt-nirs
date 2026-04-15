"""
encoder_sa.py — Survival-Aware Transformer Encoder (NOVEL MODIFICATION)
========================================================================
Extends the base TemporalTransformerEncoder with a survival-aware attention
gate that modulates how the encoder attends to temporal patterns based on
a learned censoring signal.

This file is the NOVEL contribution (analogous to graphspa/training/layer_dev.py
which adds Dynamic Adjacency Enhancement to the base layer.py).

Motivation:
  In VFD-28, patients who die before day 28 receive VFD=0 regardless of their
  ventilation trajectory. This creates two distinct patient subpopulations with
  fundamentally different outcome-generating processes:
    (a) Survivors: VFD-28 driven by ventilation duration
    (b) Non-survivors: VFD-28 = 0, driven by mortality risk

  A standard encoder treats both identically, but the temporal signals that
  predict ventilation duration differ from those predicting mortality. The
  survival-aware gate learns to route attention appropriately.

  # Ref: Fine JP, Gray RJ. "A Proportional Hazards Model for the
  #      Subdistribution of a Competing Risk." JASA 1999.
  #      — Establishes competing risks framework: death competes with
  #        ventilation liberation as events. Lines 1-3 of Introduction.

  # Ref: Analogous to graphspa layer_dev.py which adds attention_weights
  #      parameter to modulate static adjacency. Here we add survival_gate
  #      to modulate temporal attention.

Novel component — SurvivalAttentionGate:
  - Learns a gating vector g ∈ [0,1]^d_model from the encoder output
  - g modulates which representation dimensions are used for the survival
    prediction head vs the conditional VFD head
  - The gate is trained end-to-end through the adversarial loss
  - This is a single nn.Sequential + Sigmoid — minimal and surgical,
    just like graphspa adds a single attention_weights parameter.

Design reference:
  # Ref: graphspa layer_dev.py lines 25-26: adds self.attention_weights
  #      as a single learnable parameter to the base embedding class.
  #      We follow the same pattern: single gating module added to base encoder.
  # Ref: mcem attr/models/mcextremal_mask.py lines 138-143: adds F.dropout
  #      on the mask for uncertainty. Similarly minimal modification.
"""

import torch
import torch.nn as nn

from .encoder import TemporalTransformerEncoder, PositionalEncoding


class SurvivalAttentionGate(nn.Module):
    """
    Learns a soft gate that separates survival-relevant from VFD-relevant
    representation dimensions.

    Gate output g ∈ [0,1]^d_model:
      - emb_survival = emb * g          (dimensions attending to mortality risk)
      - emb_vfd      = emb * (1 - g)    (dimensions attending to ventilation duration)

    This decomposition is motivated by the competing risks literature:
    # Ref: Fine & Gray JASA 1999, Section 2 — death and discharge are
    #      competing events requiring separate hazard modeling.
    # Ref: Conceptually similar to CLEF model.py concept decomposition
    #      (lines 85-110) where learned concepts modulate predictions.
    #      CLEF decomposes temporal concepts; we decompose survival vs VFD signals.

    Args:
        d_model: Embedding dimension
        hidden_dim: Gate network hidden dimension (default: 64)
    """

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
        """
        Args:
            emb: (batch, d_model) — patient embedding from encoder
        Returns:
            gate: (batch, d_model) — soft gate values in [0, 1]
            emb_survival: (batch, d_model) — survival-relevant embedding
            emb_vfd: (batch, d_model) — VFD-relevant embedding
        """
        gate = self.gate_net(emb)
        emb_survival = emb * gate
        emb_vfd = emb * (1.0 - gate)

        return gate, emb_survival, emb_vfd


class SurvivalAwareTransformerEncoder(nn.Module):
    """
    Transformer encoder with survival-aware gating (NOVEL).

    Extends TemporalTransformerEncoder by adding SurvivalAttentionGate
    that decomposes the patient embedding into survival-relevant and
    VFD-relevant components.

    This is the encoder_sa.py ↔ encoder.py relationship, directly
    mirroring graphspa's layer_dev.py ↔ layer.py relationship:
      - layer.py:     multi_shallow_embedding with static adj
      - layer_dev.py: multi_shallow_embedding with attention_weights (NOVEL)
      - encoder.py:   TemporalTransformerEncoder (standard)
      - encoder_sa.py: + SurvivalAttentionGate (NOVEL)

    # Ref: graphspa layer_dev.py — adds ~15 lines of attention gating
    #      to the base embedding. We add SurvivalAttentionGate module.
    # Ref: mcem adds dropout + variance loss to base ExtremalMask.
    #      Similarly minimal: one new module, same base architecture.

    Args:
        n_covariates: Number of input features
        d_model: Embedding dimension (default: 128)
        n_heads, n_layers, d_ff, dropout, max_len: passed to base encoder
        gate_hidden: Hidden dim for survival gate (default: 64)
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
        """
        Args:
            x: (batch, T, n_covariates) — raw time series
            pad_mask: (batch, T) — True where padded
        Returns:
            emb: (batch, d_model) — full patient embedding
            emb_survival: (batch, d_model) — survival-gated embedding
            emb_vfd: (batch, d_model) — VFD-gated embedding
            gate: (batch, d_model) — gate values for interpretability
            attn_out: (batch, T, d_model) — per-timestep Transformer outputs
        """
        emb, attn_out = self.base_encoder(x, pad_mask)

        gate, emb_survival, emb_vfd = self.survival_gate(emb)

        return emb, emb_survival, emb_vfd, gate, attn_out
