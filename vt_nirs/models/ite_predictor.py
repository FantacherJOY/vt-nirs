
import torch
import torch.nn as nn


class ITEPredictor(nn.Module):

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
