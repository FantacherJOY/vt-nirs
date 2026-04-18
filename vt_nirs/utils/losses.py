

import torch
import torch.nn as nn
import torch.nn.functional as F


class CensoringAwareAdversarialLoss(nn.Module):
    """
    Multi-objective loss for VT-NIRS training.

    Combines adversarial, survival, VFD, consistency, gate, IPM balancing,
    propensity overlap, and doubly-robust losses.

    # Ref: mcem attr/models/mcextremal_mask.py line 295:
    #      total_loss = alpha * base_loss + beta * entropy_loss + delta * variance
    #      We follow the same weighted combination pattern.

    [v6 UPDATE — Phase 1a] Added MMD representation balancing loss
    # Ref: Shalit, Johansson & Sontag. "Estimating individual treatment effect:
    #      generalization bounds and algorithms." ICML 2017.
    #      Theorem 1: ITE error <= factual prediction error + IPM(treated, control).
    #      Section 3, Algorithm 1: CFRNet uses IPM penalty (MMD or Wasserstein).
    #      We adopt MMD (Gaussian kernel) following their Section 4.1 experiments.

    [v6 UPDATE — Phase 1b] Added propensity-based overlap weighting
    # Ref: Li, Morgan & Zaslavsky. "Balancing covariates via propensity score
    #      weighting." JASA 2018. Section 2: overlap weights w(x) = min(e, 1-e)
    #      target the overlap population, avoiding extrapolation.
    # Ref: Matsouaka et al. "Causal inference in the absence of positivity:
    #      the role of overlap weights." Biometrical Journal 2024. Sections 2-3.

    [v6 UPDATE — Phase 2a] Added AIPW doubly-robust pseudo-outcome loss
    # Ref: Kennedy. "Towards optimal doubly robust estimation of heterogeneous
    #      causal effects." arXiv 2022. Section 2: DR pseudo-outcome formula.
    # Ref: Butler et al. "DR-VIDAL." AMIA 2022. Section 3.3: DR block in
    #      adversarial framework.

    Args:
        lambda_adv: Weight for adversarial loss (default: 1.0)
        lambda_surv: Weight for survival BCE loss (default: 1.0)
        lambda_vfd: Weight for conditional VFD MSE loss (default: 1.0)
        lambda_consist: Weight for consistency loss (default: 0.5)
        lambda_gate: Weight for gate entropy regularization (default: 0.1)
        lambda_ipm: Weight for IPM (MMD) representation balancing (default: 1.0)  [v6 NEW]
        lambda_dr: Weight for doubly-robust pseudo-outcome loss (default: 0.5)    [v6 NEW]
    """

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
        """
        Generator wants discriminator to think fake outcomes are real.
        # Ref: Yoon et al. ICML 2018, Section 3.1, Eq. (4):
        #      min_G max_D E[log D(x)] + E[log(1 - D(G(x,t,z)))]
        # Ref: DT_ITE_Final.ipynb training loop line 2840:
        #      loss_G = bce(disc(fake), real_label)
        """
        target = torch.ones_like(p_real_fake)
        return self.bce(p_real_fake, target)

    def adversarial_loss_discriminator(self, p_real, p_fake):
        """
        Discriminator distinguishes real from generated outcomes.
        # Ref: Yoon et al. ICML 2018, Section 3.2, Eq. (5)
        # Ref: DT_ITE_Final.ipynb line 2833-2838:
        #      loss_D = bce(disc(real), 1) + bce(disc(fake), 0)
        """
        real_loss = self.bce(p_real, torch.ones_like(p_real))
        fake_loss = self.bce(p_fake, torch.zeros_like(p_fake))
        return 0.5 * (real_loss + fake_loss)

    def survival_loss(self, p_survive_pred, delta, event_times=None):
        """
        Survival prediction loss with optional discrete-time extension.

        [v5 ORIGINAL]: Binary cross-entropy only (coarse binary signal).
        [v6 UPDATE]: If event_times provided, adds discrete-time hazard loss
        for finer-grained survival modeling.

        # Ref: Fine & Gray JASA 1999, Section 2, Eq. (2.1):
        #      subdistribution hazard separates death from the event of interest
        # Ref: Robins & Finkelstein Biometrics 2000 — IPCW handles censoring
        #      via weighting. We instead embed it directly in the loss.

        [v6 Phase 3c] Discrete-time survival ranking loss:
        # Ref: Lee, Zame, Yoon, van der Schaar. "DeepHit: A deep learning
        #      approach to survival analysis with competing risks." AAAI 2018.
        #      Section 3.2, Eq. (3): ranking loss encourages model to assign
        #      higher survival probability to patients who survive longer.
        # Ref: Kvamme et al. "Continuous and discrete-time survival prediction
        #      with neural networks." Lifetime Data Analysis 2021. Section 4.

        NOTE: We INCLUDE death patients (delta=0). VFD-28 assigns them VFD=0
        and the survival head learns to predict who dies.

        Args:
            p_survive_pred: (batch, 1) — predicted P(survive to day 28)
            delta: (batch, 1) — 1=survived, 0=died
            event_times: (batch, 1) — days to event (optional, for ranking loss)
        """
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
        """
        MSE for conditional VFD-28, computed ONLY for survivors.

        For patients who died (delta=0), we have no information about
        their ventilation duration — VFD=0 is assigned by definition,
        not observed. So we only train the VFD head on survivors.

        # Ref: Yehya et al. AJRCCM 2019, "Limitations of VFDs" section:
        #      VFD=0 conflates death and prolonged ventilation.
        #      Our decomposition resolves this by training VFD head only
        #      on observed ventilation durations (survivors).

        Args:
            vfd_cond_pred: (batch, 1) — predicted E[VFD | survived]
            vfd_observed: (batch, 1) — observed VFD-28
            delta: (batch, 1) — 1=survived, 0=died
        """
        per_sample_mse = self.mse(vfd_cond_pred, vfd_observed)

        masked_mse = per_sample_mse * delta

        n_survivors = delta.sum().clamp(min=1.0)
        return masked_mse.sum() / n_survivors

    def consistency_loss(self, gen_outputs, pred_outputs, observed_treatment):
        """
        Jensen-Shannon divergence between Generator and Predictor outputs
        for the OBSERVED treatment arm.

        Ensures the Predictor learns to match the Generator's counterfactuals.

        # Ref: TIMING txai/trainers/train_mv6_consistency.py:
        #      clf_loss = JS_divergence(pred_mask, pred)
        #      Lines 85-120: consistency between prediction branches.
        # Ref: Yoon et al. ICML 2018, Section 3.3:
        #      Predictor trained to match Generator's outputs.

        Args:
            gen_outputs: Dict from Generator
            pred_outputs: Dict from ITEPredictor
            observed_treatment: (batch, 1) — 0=IMV, 1=NIRS
        """
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
        """
        Encourages the survival gate to be sharp (close to 0 or 1).

        Binary entropy: H(g) = -g*log(g) - (1-g)*log(1-g)
        Minimizing this pushes gate values toward binary, creating
        clear separation between survival-relevant and VFD-relevant dims.

        # Ref: mcem attr/models/mcextremal_mask.py lines 289-290:
        #      entropy_loss = -(mask * log(mask) + (1-mask) * log(1-mask))
        #      Same entropy minimization for crisp mask values.
        # Ref: TIMING txai/utils/predictors/loss.py DimEntropy:
        #      entropy = -Σ p_d * log(p_d) for mask sharpness.

        Args:
            gate: (batch, d_model) — gate values from SurvivalAttentionGate
        """
        eps = 1e-8
        entropy = -(gate * torch.log(gate + eps) +
                     (1 - gate) * torch.log(1 - gate + eps))
        return entropy.mean()

    def mmd_loss(self, emb_treated, emb_control, kernel='rbf', bandwidth=None):
        """
        Maximum Mean Discrepancy (MMD) between treated and control embeddings.

        Minimizing MMD enforces balanced representations, ensuring the encoder
        does not learn treatment-predictive features that confound ITE estimates.

        # Ref: Shalit, Johansson & Sontag. ICML 2017, Theorem 1:
        #      ε_ITE ≤ ε_F(h,Φ) + IPM_G(p_Φ^t=0, p_Φ^t=1)
        #      where IPM is an integral probability metric (MMD is one such metric).
        #      Section 4.1: CFRNet-MMD uses Gaussian RBF kernel.
        #
        # Ref: Gretton et al. "A kernel two-sample test." JMLR 2012.
        #      Eq. (3): MMD^2 = E[k(x,x')] - 2E[k(x,y)] + E[k(y,y')]
        #      with Gaussian kernel k(x,y) = exp(-||x-y||^2 / (2σ^2)).
        #
        # Ref: Johansson et al. "Learning representations for counterfactual
        #      inference." ICML 2016. Section 4: MMD penalty on learned representations.

        Args:
            emb_treated: (n_t, d) — embeddings of treated patients (NIRS)
            emb_control: (n_c, d) — embeddings of control patients (IMV)
            kernel: 'rbf' (Gaussian) or 'linear'
            bandwidth: RBF bandwidth σ. If None, uses median heuristic.
        Returns:
            mmd: scalar — MMD^2 estimate
        """
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
        """
        Binary cross-entropy for propensity score estimation.

        # Ref: Shi, Blei & Veitch. "Adapting neural networks for the estimation
        #      of treatment effects." NeurIPS 2019. Section 3: Dragonnet uses
        #      a propensity score head jointly trained with outcome heads.
        #      Targeted regularization further improves convergence.

        Args:
            propensity_logits: (batch, 1) — raw logits from propensity head
            observed_treatment: (batch, 1) — 0=IMV, 1=NIRS
        """
        return F.binary_cross_entropy_with_logits(propensity_logits, observed_treatment)

    def compute_overlap_weights(self, propensity_scores, treatment):
        """
        Compute overlap weights that focus on the equipoise population.

        # Ref: Li, Morgan & Zaslavsky. JASA 2018. Eq. (5):
        #      w_overlap(x) = W*(1-e(x)) + (1-W)*e(x)
        #      These weights naturally downweight patients with extreme propensities.
        # Ref: Matsouaka et al. Biometrical Journal 2024. Section 2:
        #      Overlap weights target the "overlap population" where both treatments
        #      have substantial probability, preventing extrapolation.

        Args:
            propensity_scores: (batch, 1) — e(x) = P(W=1|X)
            treatment: (batch, 1) — 0 or 1
        Returns:
            weights: (batch, 1) — overlap weights (normalized to sum to batch_size)
        """
        e = propensity_scores.clamp(0.01, 0.99)
        w = treatment * (1 - e) + (1 - treatment) * e
        w = w * w.size(0) / w.sum().clamp(min=1e-6)
        return w.detach()

    def doubly_robust_loss(self, pred_ite, gen_outputs, propensity_scores,
                           observed_treatment, vfd_observed, delta):
        """
        Doubly-robust (AIPW) pseudo-outcome loss for ITE predictor.

        The DR pseudo-outcome is consistent if EITHER the outcome model OR
        the propensity model is correctly specified (not both required).

        # Ref: Kennedy. "Towards optimal doubly robust estimation of
        #      heterogeneous causal effects." arXiv 2022, Section 2, Eq. (2):
        #      Γ_DR(x) = [μ₁(x) - μ₀(x)]
        #                + W/e(x) * [Y - μ₁(x)]
        #                - (1-W)/(1-e(x)) * [Y - μ₀(x)]
        #
        # Ref: Butler et al. "DR-VIDAL." AMIA 2022, Section 3.3:
        #      Integrates doubly-robust block into adversarial ITE framework.
        #      Their DR block corrects generator predictions using propensity scores.
        #
        # Ref: Chernozhukov et al. "Double/debiased machine learning."
        #      Econometrica 2018. Section 4: Neyman-orthogonal scores provide
        #      first-order insensitivity to nuisance parameter perturbation.

        Args:
            pred_ite: (batch, 1) — predicted ITE from ITEPredictor
            gen_outputs: Dict — Generator's outcome predictions (as outcome model μ)
            propensity_scores: (batch, 1) — e(x) = P(W=1|X)
            observed_treatment: (batch, 1) — 0=IMV, 1=NIRS
            vfd_observed: (batch, 1) — observed VFD-28
            delta: (batch, 1) — 1=survived, 0=died
        Returns:
            loss: scalar — MSE between predicted ITE and DR pseudo-outcome
        """
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
        """
        Total Generator loss (Stage 1).

        [v5 ORIGINAL]: L_G = λ_adv·L_adv_G + λ_s·L_survival + λ_v·L_vfd + λ_g·L_gate
        [v6 UPDATE]:   L_G = λ_adv·L_adv_G + λ_s·L_survival + λ_v·L_vfd + λ_g·L_gate
                             + λ_ipm·L_mmd + L_propensity

        # Ref: Shalit et al. ICML 2017, Algorithm 1 — add IPM penalty to factual loss
        # Ref: Shi et al. NeurIPS 2019 — joint propensity + outcome training (Dragonnet)

        Args:
            p_real_fake: (batch, 1) — Discriminator's assessment of generated data
            gen_outputs: Dict from Generator
            observed_treatment: (batch, 1) — 0 or 1
            vfd_observed: (batch, 1) — observed VFD-28
            delta: (batch, 1) — 1=survived, 0=died
            gate: (batch, d_model) — survival gate values
            emb: (batch, d_model) — encoder embeddings [v6 NEW]
            propensity_logits: (batch, 1) — propensity head logits [v6 NEW]
        """
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
        """
        Total Predictor loss (Stage 2).

        [v5 ORIGINAL]: L_P = λ_c·L_consistency  (observed arm only)
        [v6 UPDATE]:   L_P = λ_c·L_consist_obs + λ_c·L_consist_cf + λ_dr·L_DR
        [v7 UPDATE]:   L_P = λ_c·(L_consist_obs + L_consist_cf) + λ_dr·L_DR_direct
                       where L_DR_direct targets the unconstrained direct ITE head

        [v6 — Phase 1c] Train on BOTH arms (observed + counterfactual)
        # Ref: Yoon et al. ICML 2018, Section 3.3, Eq. (7):
        #      GANITE trains predictor on Generator outputs for BOTH arms,
        #      not just the observed arm. Our v5 only used observed arm.

        [v7 — Direct ITE with DR pseudo-outcomes]
        # Ref: Kennedy. Ann Stat 2023. Theorem 1: DR pseudo-outcome
        #      Gamma_DR provides sqrt(n)-consistent CATE estimation.
        # Ref: Nie & Wager. Biometrika 2021. Theorem 1: R-learner achieves
        #      oracle rates by directly targeting tau(x) with orthogonal loss.
        # Ref: Foster & Syrgkanis. Ann Stat 2023. Section 3:
        #      orthogonal statistical learning for CATE estimation.
        # The direct ITE head is unconstrained and learns the full [-28,28]
        # range, while decomposed heads still train via consistency loss
        # for interpretability (survival vs VFD components).

        Args:
            gen_outputs: Dict from Generator (used as pseudo-labels)
            pred_outputs: Dict from ITEPredictor
            observed_treatment: (batch, 1)
            propensity_scores: (batch, 1) — e(x) [v6 NEW]
            vfd_observed: (batch, 1) [v6 NEW]
            delta: (batch, 1) [v6 NEW]
        """
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
