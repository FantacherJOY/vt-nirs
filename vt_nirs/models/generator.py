

import torch
import torch.nn as nn


class CounterfactualGenerator(nn.Module):

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

        h = torch.cat([emb, treatment, noise], dim=-1)

        h = self.shared_trunk(h)

        p_survive = self.survival_head(h)
        vfd_cond = self.vfd_head(h).clamp(max=28.0)

        vfd_composite = p_survive * vfd_cond

        return p_survive, vfd_cond, vfd_composite

    def forward_with_gated_emb(self, emb, treatment, noise, emb_survival, emb_vfd):
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
