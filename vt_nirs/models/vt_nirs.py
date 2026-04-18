

import torch
import torch.nn as nn

from .encoder_sa import SurvivalAwareTransformerEncoder
from .generator import CounterfactualGenerator
from .discriminator import TreatmentDiscriminator
from .ite_predictor import ITEPredictor


class VTNIRSModel(nn.Module):

    def __init__(
        self,
        n_covariates=23,
        d_model=128,
        n_heads=4,
        n_layers=4,
        d_ff=256,
        noise_dim=8,
        hidden_dim=128,
        dropout=0.1,
    ):
        super().__init__()

        self.n_covariates = n_covariates
        self.d_model = d_model
        self.noise_dim = noise_dim


        self.encoder = SurvivalAwareTransformerEncoder(
            n_covariates=n_covariates,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
        )

        self.generator = CounterfactualGenerator(
            emb_dim=d_model,
            noise_dim=noise_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self.discriminator = TreatmentDiscriminator(
            emb_dim=d_model,
            hidden_dim=hidden_dim,
        )

        self.predictor = ITEPredictor(
            emb_dim=d_model,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self.propensity_head = nn.Sequential(
            nn.Linear(d_model, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def encode(self, x, pad_mask=None):
        return self.encoder(x, pad_mask)

    def forward_generator(self, x, treatment, pad_mask=None, noise=None):
        batch_size = x.size(0)
        device = x.device

        emb, emb_survival, emb_vfd, gate, attn_out = self.encode(x, pad_mask)

        if noise is None:
            noise = torch.randn(batch_size, self.noise_dim, device=device)

        gen_outputs = self.generator.generate_counterfactuals(
            emb, noise, emb_survival=emb_survival, emb_vfd=emb_vfd)

        propensity_logits = self.propensity_head(emb.detach())

        encoder_outputs = (emb, emb_survival, emb_vfd, gate, attn_out,
                          propensity_logits)

        return gen_outputs, encoder_outputs

    def forward_discriminator(self, emb, gen_outputs):
        outcomes = torch.cat([
            gen_outputs['p_surv_0'], gen_outputs['vfd_cond_0'], gen_outputs['vfd_0'],
            gen_outputs['p_surv_1'], gen_outputs['vfd_cond_1'], gen_outputs['vfd_1'],
        ], dim=-1)

        return self.discriminator(emb.detach(), outcomes)

    def forward_predictor(self, x, pad_mask=None):
        emb, emb_survival, emb_vfd, gate, attn_out = self.encode(x, pad_mask)
        pred_outputs = self.predictor(emb)

        propensity_logits = self.propensity_head(emb.detach())
        encoder_outputs = (emb, emb_survival, emb_vfd, gate, attn_out,
                          propensity_logits)

        return pred_outputs, encoder_outputs

    def predict_ite(self, x, pad_mask=None):
        pred_outputs, _ = self.forward_predictor(x, pad_mask)
        return pred_outputs['ite']

    def get_treatment_recommendation(self, x, pad_mask=None):
        pred_outputs, _ = self.forward_predictor(x, pad_mask)
        ite = pred_outputs['ite']

        rec = (ite > 0).squeeze(-1).long()

        return rec, ite, pred_outputs
