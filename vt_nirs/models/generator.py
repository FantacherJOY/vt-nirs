"""
generator.py — Censoring-Aware Counterfactual Generator
========================================================
Generates counterfactual outcomes Y(0) and Y(1) under both treatments,
decomposed into survival probability and conditional VFD-28.

Base: GANITE CounterfactualGenerator (Yoon et al. ICML 2018, Section 3.1)
  - Input: patient embedding + treatment indicator + noise
  - Output: counterfactual outcomes under both treatments
  # Ref: Yoon et al. ICML 2018, Section 3.1 "Generator"
  # Ref: DT_ITE_Final.ipynb class CounterfactualGenerator (line 2699)

Novel modification: Two-headed output with survival decomposition
  - Head 1: P(survive to day 28 | treatment=t) ∈ [0,1]
  - Head 2: E[VFD-28 | survived, treatment=t] ∈ [0,28]
  - Final: VFD(t) = P(survive|t) × E[VFD|alive,t]

  This decomposition is motivated by:
  # Ref: Fine & Gray JASA 1999 — competing risks: death competes
  #      with ventilation liberation. Section 2, Eqs (2.1)-(2.3).
  # Ref: Yehya N et al. "Ventilator-Free Days: What Is the Right
  #      Outcome for Lung Injury Trials?" AJRCCM 2019 — defines VFD-28
  #      and discusses the death-as-zero problem (Section "Limitations of VFDs").

  The two-head design follows the same pattern as CLEF model.py (lines 85-110)
  which decomposes predictions into learned temporal concepts × historical values.
  # Ref: CLEF model.py lines 85-110: concept decomposition for prediction
"""

import torch
import torch.nn as nn


class CounterfactualGenerator(nn.Module):
    """
    Two-headed counterfactual generator for VFD-28 with survival decomposition.

    Follows GANITE's generator structure (embedding + treatment + noise → outcomes)
    but outputs a survival probability and conditional VFD for each treatment arm.

    # Ref: Yoon et al. ICML 2018, Section 3.1 — Generator takes
    #      (x, t, z) as input and produces counterfactual G(x,t,z).
    # Ref: DT_ITE_Final.ipynb line 2699-2719 — our existing GANITE generator
    #      uses nn.Sequential with Linear→ReLU→Dropout→Linear→Sigmoid.
    #      We extend this pattern with two output heads.

    Args:
        emb_dim: Patient embedding dimension (from encoder)
        noise_dim: Random noise dimension (default: 8, matching AMIA code)
        hidden_dim: Generator hidden layer size (default: 128)
        dropout: Dropout rate (default: 0.2)
    """

    def __init__(self, emb_dim=128, noise_dim=8, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.emb_dim = emb_dim
        self.noise_dim = noise_dim

        input_dim = emb_dim + 1 + noise_dim

        self.shared_trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.survival_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        self.vfd_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(),
        )

    def forward(self, emb, treatment, noise):
        """
        Generate counterfactual outcomes for a given treatment.

        Args:
            emb: (batch, emb_dim) — patient embedding (can be full, survival, or VFD)
            treatment: (batch, 1) — treatment indicator (0=IMV, 1=NIRS)
            noise: (batch, noise_dim) — random noise for stochasticity

        Returns:
            p_survive: (batch, 1) — P(survive to day 28 | treatment)
            vfd_cond: (batch, 1) — E[VFD-28 | survived, treatment] in [0, 28]
            vfd_composite: (batch, 1) — p_survive × vfd_cond (final VFD-28 estimate)
        """
        h = torch.cat([emb, treatment, noise], dim=-1)

        h = self.shared_trunk(h)

        p_survive = self.survival_head(h)
        vfd_cond = self.vfd_head(h).clamp(max=28.0)

        vfd_composite = p_survive * vfd_cond

        return p_survive, vfd_cond, vfd_composite

    def forward_with_gated_emb(self, emb, treatment, noise, emb_survival, emb_vfd):
        """
        [v6 NEW — Phase 2b] Generate outcomes using gated embeddings.

        Routes emb_survival to survival head and emb_vfd to VFD head,
        enforcing the competing risks decomposition architecturally.

        # Ref: Fine & Gray JASA 1999 — competing risks require separate modeling
        #      of death and ventilation liberation hazards. By routing gated
        #      embeddings to the corresponding heads, we enforce this separation
        #      architecturally rather than relying solely on loss-based incentives.
        # Ref: This addresses Deficiency #9 in our optimization proposal — the
        #      gate was previously disconnected from the generator heads.

        Args:
            emb: (batch, emb_dim) — full embedding (for shared trunk)
            treatment: (batch, 1)
            noise: (batch, noise_dim)
            emb_survival: (batch, emb_dim) — survival-gated embedding
            emb_vfd: (batch, emb_dim) — VFD-gated embedding
        """
        h = torch.cat([emb, treatment, noise], dim=-1)
        h = self.shared_trunk(h)

        h_surv = h + emb_survival[:, :h.size(-1)]
        p_survive = self.survival_head(h_surv)

        h_vfd = h + emb_vfd[:, :h.size(-1)]
        vfd_cond = self.vfd_head(h_vfd).clamp(max=28.0)

        vfd_composite = p_survive * vfd_cond
        return p_survive, vfd_cond, vfd_composite

    def generate_counterfactuals(self, emb, noise,
                                  emb_survival=None, emb_vfd=None):
        """
        Generate outcomes under BOTH treatments (for discriminator training).

        [v6 UPDATE] Accepts optional gated embeddings for Phase 2b routing.

        Args:
            emb: (batch, emb_dim) — patient embedding
            noise: (batch, noise_dim)
            emb_survival: (batch, emb_dim) — survival-gated embedding [v6 NEW]
            emb_vfd: (batch, emb_dim) — VFD-gated embedding [v6 NEW]
        Returns:
            Dict with keys: p_surv_0, vfd_cond_0, vfd_0 (IMV arm)
                           p_surv_1, vfd_cond_1, vfd_1 (NIRS arm)
        """
        batch_size = emb.size(0)
        device = emb.device

        t0 = torch.zeros(batch_size, 1, device=device)
        t1 = torch.ones(batch_size, 1, device=device)

        if emb_survival is not None and emb_vfd is not None:
            p_surv_0, vfd_cond_0, vfd_0 = self.forward_with_gated_emb(
                emb, t0, noise, emb_survival, emb_vfd)
            p_surv_1, vfd_cond_1, vfd_1 = self.forward_with_gated_emb(
                emb, t1, noise, emb_survival, emb_vfd)
        else:
            p_surv_0, vfd_cond_0, vfd_0 = self.forward(emb, t0, noise)
            p_surv_1, vfd_cond_1, vfd_1 = self.forward(emb, t1, noise)

        return {
            'p_surv_0': p_surv_0, 'vfd_cond_0': vfd_cond_0, 'vfd_0': vfd_0,
            'p_surv_1': p_surv_1, 'vfd_cond_1': vfd_cond_1, 'vfd_1': vfd_1,
        }
