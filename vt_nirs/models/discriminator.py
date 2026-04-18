

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class TreatmentDiscriminator(nn.Module):

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
        h = torch.cat([emb, outcomes], dim=-1)
        return self.net(h)

    def gradient_penalty(self, emb, real_outcomes, fake_outcomes, lambda_gp=10.0):
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
