

import torch
import torch.nn as nn
import torch.nn.functional as F


class CensoringAwareAdversarialLoss(nn.Module):
    def __init__(
        self,
        lambda_adv=1.0,
        lambda_surv=1.0,
        lambda_vfd=1.0,
        lambda_consist=0.5,
        lambda_gate=0.1,
        lambda_ipm=1.0,
        lambda_dr=0.5,
    ):
        super().__init__()
        self.lambda_adv = lambda_adv
        self.lambda_surv = lambda_surv
        self.lambda_vfd = lambda_vfd
        self.lambda_consist = lambda_consist
        self.lambda_gate = lambda_gate
        self.lambda_ipm = lambda_ipm
        self.lambda_dr = lambda_dr

        self.bce = nn.BCELoss(reduction='mean')
        self.mse = nn.MSELoss(reduction='none')

    def adversarial_loss_generator(self, p_real_fake):
        target = torch.ones_like(p_real_fake)
        return self.bce(p_real_fake, target)

    def adversarial_loss_discriminator(self, p_real, p_fake):
        real_loss = self.bce(p_real, torch.ones_like(p_real))
        fake_loss = self.bce(p_fake, torch.zeros_like(p_fake))
        return 0.5 * (real_loss + fake_loss)

    def survival_loss(self, p_survive_pred, delta, event_times=None):
        bce_loss = self.bce(p_survive_pred, delta)

        if event_times is None:
            return bce_loss

        p_surv = p_survive_pred.squeeze()
        t = event_times.squeeze()
        d = delta.squeeze()

        n = p_surv.size(0)
        if n < 2:
            return bce_loss

        dead_mask = (d == 0)
        if dead_mask.sum() == 0 or (~dead_mask).sum() == 0:
            return bce_loss

        dead_p = p_surv[dead_mask]
        dead_t = t[dead_mask]
        alive_p = p_surv[~dead_mask]

        margin = 0.1
        n_pairs = min(dead_p.size(0), alive_p.size(0), 64)
        idx_dead = torch.randperm(dead_p.size(0))[:n_pairs]
        idx_alive = torch.randperm(alive_p.size(0))[:n_pairs]

        ranking_loss = torch.clamp(
            dead_p[idx_dead] - alive_p[idx_alive] + margin, min=0.0).mean()

        return bce_loss + 0.5 * ranking_loss

    def conditional_vfd_loss(self, vfd_cond_pred, vfd_observed, delta):
        per_sample_mse = self.mse(vfd_cond_pred, vfd_observed)

        masked_mse = per_sample_mse * delta

        n_survivors = delta.sum().clamp(min=1.0)
        return masked_mse.sum() / n_survivors

    def consistency_loss(self, gen_outputs, pred_outputs, observed_treatment):
        t = observed_treatment.squeeze(-1)

        gen_vfd = torch.where(
            t.unsqueeze(-1) == 1,
            gen_outputs['vfd_1'],
            gen_outputs['vfd_0']
        )
        pred_vfd = torch.where(
            t.unsqueeze(-1) == 1,
            pred_outputs['vfd_1'],
            pred_outputs['vfd_0']
        )

        return F.mse_loss(pred_vfd, gen_vfd.detach())

    def gate_entropy_loss(self, gate):
        eps = 1e-8
        entropy = -(gate * torch.log(gate + eps) +
                     (1 - gate) * torch.log(1 - gate + eps))
        return entropy.mean()

    def mmd_loss(self, emb_treated, emb_control, kernel='rbf', bandwidth=None):
        if emb_treated.size(0) == 0 or emb_control.size(0) == 0:
            return torch.tensor(0.0, device=emb_treated.device)

        if kernel == 'linear':
            mean_t = emb_treated.mean(dim=0)
            mean_c = emb_control.mean(dim=0)
            return ((mean_t - mean_c) ** 2).sum()

        all_emb = torch.cat([emb_treated, emb_control], dim=0)
        pairwise_dist = torch.cdist(all_emb, all_emb, p=2)
        if bandwidth is None:
            bandwidth = torch.median(pairwise_dist[pairwise_dist > 0]).detach()
            bandwidth = bandwidth.clamp(min=1e-6)

        gamma = 1.0 / (2.0 * bandwidth ** 2)

        n_t = emb_treated.size(0)
        n_c = emb_control.size(0)

        K_tt = torch.exp(-gamma * torch.cdist(emb_treated, emb_treated, p=2) ** 2)
        K_cc = torch.exp(-gamma * torch.cdist(emb_control, emb_control, p=2) ** 2)
        K_tc = torch.exp(-gamma * torch.cdist(emb_treated, emb_control, p=2) ** 2)

        mmd = (K_tt.sum() / (n_t * n_t)
               - 2.0 * K_tc.sum() / (n_t * n_c)
               + K_cc.sum() / (n_c * n_c))

        return mmd

    def propensity_loss(self, propensity_logits, observed_treatment):
        return F.binary_cross_entropy_with_logits(propensity_logits, observed_treatment)

    def compute_overlap_weights(self, propensity_scores, treatment):
        e = propensity_scores.clamp(0.01, 0.99)
        w = treatment * (1 - e) + (1 - treatment) * e
        w = w * w.size(0) / w.sum().clamp(min=1e-6)
        return w.detach()

    def doubly_robust_loss(self, pred_ite, gen_outputs, propensity_scores,
                           observed_treatment, vfd_observed, delta):
        W = observed_treatment
        Y = delta * vfd_observed
        e = propensity_scores.clamp(0.01, 0.99)

        mu_1 = gen_outputs['vfd_1'].detach()
        mu_0 = gen_outputs['vfd_0'].detach()

        dr_pseudo = (mu_1 - mu_0
                     + W / e * (Y - mu_1)
                     - (1 - W) / (1 - e) * (Y - mu_0))

        return F.mse_loss(pred_ite, dr_pseudo.detach())

    def generator_loss(
        self,
        p_real_fake,
        gen_outputs,
        observed_treatment,
        vfd_observed,
        delta,
        gate,
        emb=None,
        propensity_logits=None,
    ):
        t = observed_treatment.squeeze(-1)

        l_adv = self.adversarial_loss_generator(p_real_fake)

        p_surv_obs = torch.where(
            t.unsqueeze(-1) == 1,
            gen_outputs['p_surv_1'],
            gen_outputs['p_surv_0']
        )
        l_surv = self.survival_loss(p_surv_obs, delta)

        vfd_cond_obs = torch.where(
            t.unsqueeze(-1) == 1,
            gen_outputs['vfd_cond_1'],
            gen_outputs['vfd_cond_0']
        )
        l_vfd = self.conditional_vfd_loss(vfd_cond_obs, vfd_observed, delta)

        l_gate = self.gate_entropy_loss(gate)

        l_mmd = torch.tensor(0.0, device=gate.device)
        if emb is not None:
            t_mask = (t == 1)
            c_mask = (t == 0)
            if t_mask.sum() > 0 and c_mask.sum() > 0:
                l_mmd = self.mmd_loss(emb[t_mask], emb[c_mask])

        l_prop = torch.tensor(0.0, device=gate.device)
        if propensity_logits is not None:
            l_prop = self.propensity_loss(propensity_logits, observed_treatment)

        total = (
            self.lambda_adv * l_adv
            + self.lambda_surv * l_surv
            + self.lambda_vfd * l_vfd
            + self.lambda_gate * l_gate
            + self.lambda_ipm * l_mmd
            + l_prop
        )

        return total, {
            'l_adv_G': l_adv.item(),
            'l_surv': l_surv.item(),
            'l_vfd': l_vfd.item(),
            'l_gate': l_gate.item(),
            'l_mmd': l_mmd.item(),
            'l_prop': l_prop.item(),
            'l_total_G': total.item(),
        }

    def predictor_loss(self, gen_outputs, pred_outputs, observed_treatment,
                        propensity_scores=None, vfd_observed=None, delta=None):
        l_consist_obs = self.consistency_loss(gen_outputs, pred_outputs,
                                              observed_treatment)

        t = observed_treatment.squeeze(-1)
        gen_vfd_cf = torch.where(
            t.unsqueeze(-1) == 1,
            gen_outputs['vfd_0'],
            gen_outputs['vfd_1']
        )
        pred_vfd_cf = torch.where(
            t.unsqueeze(-1) == 1,
            pred_outputs['vfd_0'],
            pred_outputs['vfd_1']
        )
        l_consist_cf = F.mse_loss(pred_vfd_cf, gen_vfd_cf.detach())

        l_dr = torch.tensor(0.0, device=observed_treatment.device)
        if (propensity_scores is not None and vfd_observed is not None
                and delta is not None):
            l_dr = self.doubly_robust_loss(
                pred_outputs['ite'], gen_outputs, propensity_scores,
                observed_treatment, vfd_observed, delta)

        total = (self.lambda_consist * l_consist_obs
                 + self.lambda_consist * l_consist_cf
                 + self.lambda_dr * l_dr)

        return total, {
            'l_consist_obs': l_consist_obs.item(),
            'l_consist_cf': l_consist_cf.item(),
            'l_dr': l_dr.item(),
            'l_consist': total.item(),
        }
