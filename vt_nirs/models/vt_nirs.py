

import torch
import torch.nn as nn

from .encoder_sa import SurvivalAwareTransformerEncoder
from .generator import CounterfactualGenerator
from .discriminator import TreatmentDiscriminator
from .ite_predictor import ITEPredictor


class VTNIRSModel(nn.Module):
    """
    Virtual Twin for Non-Invasive Respiratory Support (VT-NIRS).

    Complete model assembling encoder, generator, discriminator, and predictor.

    # Ref: graphspa net.py lines 6-129: GNNStack assembles layers, embeddings,
    #      pooling into one model. We follow the same pattern.
    # Ref: Yoon et al. ICML 2018: GANITE 4-module architecture
    # Ref: DT_ITE_Final.ipynb lines 2792-2795: module instantiation

    Args:
        n_covariates: Number of input features (default: 23)
        d_model: Encoder embedding dimension (default: 128)
        n_heads: Transformer attention heads (default: 4)
        n_layers: Transformer layers (default: 4)
        d_ff: Feed-forward dimension (default: 256)
        noise_dim: Generator noise dimension (default: 8)
        hidden_dim: Hidden dim for generator/discriminator/predictor (default: 128)
        dropout: Dropout rate (default: 0.1)
    """

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
        """
        Encode patient time series into embeddings.

        Args:
            x: (batch, T, n_covariates)
            pad_mask: (batch, T) — True where padded
        Returns:
            emb, emb_survival, emb_vfd, gate, attn_out
        """
        return self.encoder(x, pad_mask)

    def forward_generator(self, x, treatment, pad_mask=None, noise=None):
        """
        Stage 1 forward pass: Encoder → Generator (+ Propensity Head).

        [v6 UPDATE] Now also returns propensity logits for overlap weighting
        and DR correction.

        Args:
            x: (batch, T, n_covariates) — patient time series
            treatment: (batch, 1) — observed treatment
            pad_mask: (batch, T)
            noise: (batch, noise_dim) — if None, sampled randomly
        Returns:
            gen_outputs: Dict with counterfactual outcomes
            encoder_outputs: Tuple (emb, emb_survival, emb_vfd, gate, attn_out,
                                    propensity_logits)  [v6: added propensity]
        """
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
        """
        Stage 1 discriminator pass.

        Args:
            emb: (batch, d_model) — patient embedding
            gen_outputs: Dict from generator
        Returns:
            p_real: (batch, 1)
        """
        outcomes = torch.cat([
            gen_outputs['p_surv_0'], gen_outputs['vfd_cond_0'], gen_outputs['vfd_0'],
            gen_outputs['p_surv_1'], gen_outputs['vfd_cond_1'], gen_outputs['vfd_1'],
        ], dim=-1)

        return self.discriminator(emb.detach(), outcomes)

    def forward_predictor(self, x, pad_mask=None):
        """
        Stage 2 / inference: Encoder → ITEPredictor.

        [v6 UPDATE] Also returns propensity scores for DR loss computation.

        Args:
            x: (batch, T, n_covariates)
            pad_mask: (batch, T)
        Returns:
            pred_outputs: Dict with ITE predictions
            encoder_outputs: Tuple (includes propensity_logits)
        """
        emb, emb_survival, emb_vfd, gate, attn_out = self.encode(x, pad_mask)
        pred_outputs = self.predictor(emb)

        propensity_logits = self.propensity_head(emb.detach())
        encoder_outputs = (emb, emb_survival, emb_vfd, gate, attn_out,
                          propensity_logits)

        return pred_outputs, encoder_outputs

    def predict_ite(self, x, pad_mask=None):
        """
        Convenience method for inference: returns just the ITE.
        """
        pred_outputs, _ = self.forward_predictor(x, pad_mask)
        return pred_outputs['ite']

    def get_treatment_recommendation(self, x, pad_mask=None):
        """
        Returns treatment recommendation for each patient.

        Returns:
            rec: (batch,) — 1 if NIRS recommended, 0 if IMV recommended
            ite: (batch, 1) — ITE values
            pred_outputs: Full prediction dict
        """
        pred_outputs, _ = self.forward_predictor(x, pad_mask)
        ite = pred_outputs['ite']

        rec = (ite > 0).squeeze(-1).long()

        return rec, ite, pred_outputs
