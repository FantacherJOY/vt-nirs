"""
discriminator.py — Treatment-Aware Discriminator
=================================================
Distinguishes real observed outcomes from generator-produced counterfactuals.

Base: GANITE Discriminator (Yoon et al. ICML 2018, Section 3.2)
  # Ref: Yoon et al. ICML 2018, Section 3.2 "Discriminator"
  # Ref: DT_ITE_Final.ipynb class Discriminator (line 2722)
  #      Input: [embedding(128), y0(1), y1(1)] = 130-dim → Sigmoid

Modification for VFD-28:
  - Input now includes survival probabilities AND VFD values for both arms
  - Deeper network with spectral-norm stabilization for adversarial training
  # Ref: Miyato et al. "Spectral Normalization for GANs." ICLR 2018
  #      — spectral norm prevents mode collapse in adversarial training.
  #      Section 2, Eq. (6): constrains Lipschitz constant of discriminator.

[v6 UPDATE — Phase 2c] Deepened discriminator + gradient penalty
  # Ref: Miyato et al. ICLR 2018: recommended deeper architectures when using
  #      spectral normalization to compensate for per-layer Lipschitz constraint.
  # Ref: Gulrajani et al. "Improved training of Wasserstein GANs." NeurIPS 2017,
  #      Section 4: gradient penalty enforces 1-Lipschitz on interpolated samples.
  # Ref: CMU ML Blog "Why spectral normalization stabilizes GANs" (2022):
  #      coupling spectral norm with zero-centered gradient penalty overcomes
  #      convergence issues and alleviates mode collapse.
"""

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class TreatmentDiscriminator(nn.Module):
    """
    Discriminator that receives patient embedding + generated outcomes
    and outputs P(real) — whether the outcomes are observed (real) or
    generated (fake).

    # Ref: Yoon et al. ICML 2018, Section 3.2: D takes (x, y^(0), y^(1))
    #      and outputs which y is observed vs generated.
    # Ref: DT_ITE_Final.ipynb line 2722-2738: Discriminator class
    #      with nn.Linear(emb_dim + 2, 64) → ReLU → Linear(64, 1) → Sigmoid

    [v6 UPDATE — Phase 2c] Deepened from 3 to 5 layers for better realism
    enforcement on 6-dimensional outcome space. Added gradient penalty method.
    # Ref: Miyato et al. ICLR 2018 — deeper SN discriminators improve sample quality
    # Ref: Gulrajani et al. NeurIPS 2017 — gradient penalty for WGAN-GP stability

    Args:
        emb_dim: Patient embedding dimension
        hidden_dim: Discriminator hidden size (default: 128)
    """

    def __init__(self, emb_dim=128, hidden_dim=128):
        super().__init__()
        input_dim = emb_dim + 6

        self.net = nn.Sequential(
            spectral_norm(nn.Linear(input_dim, hidden_dim)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            spectral_norm(nn.Linear(hidden_dim, hidden_dim)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            spectral_norm(nn.Linear(hidden_dim, hidden_dim // 2)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(hidden_dim // 2, hidden_dim // 4)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(hidden_dim // 4, 1)),
            nn.Sigmoid(),
        )

    def forward(self, emb, outcomes):
        """
        Args:
            emb: (batch, emb_dim) — patient embedding
            outcomes: (batch, 6) — [p_surv_0, vfd_cond_0, vfd_0,
                                     p_surv_1, vfd_cond_1, vfd_1]
        Returns:
            p_real: (batch, 1) — probability that outcomes are real
        """
        h = torch.cat([emb, outcomes], dim=-1)
        return self.net(h)

    def gradient_penalty(self, emb, real_outcomes, fake_outcomes, lambda_gp=10.0):
        """
        [v6 NEW — Phase 2c] Gradient penalty for improved training stability.

        Enforces 1-Lipschitz constraint on the discriminator by penalizing
        the gradient norm on interpolated samples between real and fake outcomes.

        # Ref: Gulrajani et al. "Improved training of Wasserstein GANs."
        #      NeurIPS 2017, Section 4, Eq. (3): penalty on gradient norm
        #      of critic evaluated at random interpolations between real/fake.
        #      λ_gp = 10 is the default from their paper.
        # Ref: CMU ML Blog (2022): combining spectral norm + gradient penalty
        #      provides better convergence than either alone.

        Args:
            emb: (batch, emb_dim) — patient embeddings
            real_outcomes: (batch, 6) — real outcome tensor
            fake_outcomes: (batch, 6) — generator outcome tensor
            lambda_gp: gradient penalty coefficient (default: 10.0)
        Returns:
            gp: scalar — gradient penalty loss
        """
        batch_size = real_outcomes.size(0)
        device = real_outcomes.device

        alpha = torch.rand(batch_size, 1, device=device)

        interpolated = (alpha * real_outcomes + (1 - alpha) * fake_outcomes)
        interpolated.requires_grad_(True)

        d_interpolated = self.forward(emb.detach(), interpolated)

        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True,
        )[0]

        gradient_norm = gradients.view(batch_size, -1).norm(2, dim=1)
        gp = lambda_gp * ((gradient_norm - 1.0) ** 2).mean()

        return gp
