"""
ite_predictor.py — Two-Headed ITE Predictor (Inference Module)
===============================================================
Predicts individualized treatment effects at inference time,
without requiring noise input (unlike the Generator).

Base: GANITE ITEPredictor (Yoon et al. ICML 2018, Section 3.3)
  # Ref: Yoon et al. ICML 2018, Section 3.3 "ITE Estimation"
  #      — separate supervised network trained on Generator's outputs
  #      to predict ITE without noise at inference time.
  # Ref: DT_ITE_Final.ipynb class ITEPredictor (line 2741)
  #      Input: embedding(128) → Linear(128,64) → ReLU → Linear(64,2) → Sigmoid

Modification: Same two-headed structure as Generator for consistency:
  - Survival head → P(survive | treatment=t) for t ∈ {0,1}
  - VFD head → E[VFD-28 | survived, treatment=t] for t ∈ {0,1}
  - ITE computed as: VFD(1) - VFD(0)  [NIRS effect relative to IMV]

[v7 UPDATE] Added unconstrained direct ITE head supervised by DR-IPCW
pseudo-outcomes. The decomposed heads (survival × conditional VFD) compress
ITE via multiplicative coupling when both arms have similar survival probs.
The direct head bypasses this by learning tau(x) directly from the embedding,
supervised by doubly-robust pseudo-outcomes (Kennedy, Ann Stat 2023).
  # Ref: Nie & Wager. "Quasi-oracle estimation of heterogeneous treatment
  #      effects." Biometrika 2021. Section 2: R-learner directly targets
  #      the CATE function tau(x) rather than individual potential outcomes.
  # Ref: Kennedy. "Towards optimal doubly robust estimation of heterogeneous
  #      causal effects." Ann Stat 2023. Theorem 1: DR pseudo-outcome
  #      provides sqrt(n)-consistent CATE estimation.
  # Ref: Foster & Syrgkanis. "Orthogonal statistical learning." Ann Stat
  #      2023. Section 3: direct CATE estimation via orthogonal loss.
"""

import torch
import torch.nn as nn


class ITEPredictor(nn.Module):
    """
    Inference-time ITE predictor with survival decomposition.

    Trained on pseudo-labels from the Generator (GANITE Stage 2).
    At inference, takes only the patient embedding (no noise).

    # Ref: Yoon et al. ICML 2018, Section 3.3, Eq. (7):
    #      ITE predictor minimizes MSE to Generator's outputs.
    # Ref: DT_ITE_Final.ipynb line 2741-2758: ITEPredictor class

    Two-headed output matches Generator structure for consistency.
    # Ref: TIMING txai/trainers/train_mv6_consistency.py —
    #      uses consistency loss between branches (lines 85-120).
    #      We ensure Generator and Predictor share output structure.

    Args:
        emb_dim: Patient embedding dimension
        hidden_dim: Predictor hidden size (default: 128)
        dropout: Dropout rate (default: 0.1)
    """

    def __init__(self, emb_dim=128, hidden_dim=128, dropout=0.1):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.surv_head_0 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        self.surv_head_1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        self.vfd_head_0 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(),
        )
        self.vfd_head_1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(),
        )

        self.direct_ite_head = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(self, emb):
        """
        Args:
            emb: (batch, emb_dim) — patient embedding
        Returns:
            Dict with per-arm survival and VFD predictions + ITE
            'ite' is the PRIMARY ITE from the direct head (unconstrained)
            'ite_decomposed' is the auxiliary ITE from survival × VFD heads
        """
        h = self.shared(emb)

        p_surv_0 = self.surv_head_0(h)
        vfd_cond_0 = self.vfd_head_0(h).clamp(max=28.0)
        vfd_0 = p_surv_0 * vfd_cond_0

        p_surv_1 = self.surv_head_1(h)
        vfd_cond_1 = self.vfd_head_1(h).clamp(max=28.0)
        vfd_1 = p_surv_1 * vfd_cond_1

        ite_decomposed = vfd_1 - vfd_0

        h_ite = torch.cat([h, ite_decomposed.detach()], dim=-1)
        ite = self.direct_ite_head(h_ite)

        ite_survival = p_surv_1 - p_surv_0
        ite_vfd_cond = vfd_cond_1 - vfd_cond_0

        return {
            'p_surv_0': p_surv_0, 'vfd_cond_0': vfd_cond_0, 'vfd_0': vfd_0,
            'p_surv_1': p_surv_1, 'vfd_cond_1': vfd_cond_1, 'vfd_1': vfd_1,
            'ite': ite,
            'ite_decomposed': ite_decomposed,
            'ite_survival': ite_survival,
            'ite_vfd_cond': ite_vfd_cond,
        }
