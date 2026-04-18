

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score, average_precision_score


def pehe(ite_pred, ite_true):
    return np.sqrt(np.mean((ite_pred - ite_true) ** 2))


def ate_bias(ite_pred, ite_true):
    return np.abs(np.mean(ite_pred) - np.mean(ite_true))


def policy_value(ite_pred, vfd_observed, treatment_observed):
    recommended = (ite_pred > 0).astype(int)

    concordant = (recommended == treatment_observed)
    n_concordant = concordant.sum()

    vfd_concordant = vfd_observed[concordant].mean() if n_concordant > 0 else 0.0

    vfd_observed_policy = vfd_observed.mean()

    pct_nirs = recommended.mean() * 100

    return {
        'vfd_model_policy': vfd_concordant,
        'vfd_observed_policy': vfd_observed_policy,
        'policy_improvement': vfd_concordant - vfd_observed_policy,
        'n_concordant': int(n_concordant),
        'pct_concordant': n_concordant / len(ite_pred) * 100,
        'pct_recommend_nirs': pct_nirs,
    }


def survival_calibration(p_survive_pred, delta):
    return brier_score_loss(delta, p_survive_pred)


def c_for_benefit(ite_pred, vfd_observed, treatment_observed,
                  n_pairs=10000, random_state=42):
    rng = np.random.RandomState(random_state)

    nirs_idx = np.where(treatment_observed == 1)[0]
    imv_idx = np.where(treatment_observed == 0)[0]

    if len(nirs_idx) == 0 or len(imv_idx) == 0:
        return {'c_for_benefit': 0.5, 'n_pairs_evaluated': 0,
                'interpretation': 'Cannot compute: one treatment arm is empty'}

    n_pairs = min(n_pairs, len(nirs_idx) * len(imv_idx))

    idx_n = rng.choice(nirs_idx, size=n_pairs, replace=True)
    idx_m = rng.choice(imv_idx, size=n_pairs, replace=True)

    pred_diff = ite_pred[idx_n] - ite_pred[idx_m]

    obs_diff = vfd_observed[idx_n] - vfd_observed[idx_m]

    concordant = np.sum((pred_diff > 0) & (obs_diff > 0)) + \
                 np.sum((pred_diff < 0) & (obs_diff < 0))
    discordant = np.sum((pred_diff > 0) & (obs_diff < 0)) + \
                 np.sum((pred_diff < 0) & (obs_diff > 0))
    tied = n_pairs - concordant - discordant

    c_benefit = (concordant + 0.5 * tied) / n_pairs if n_pairs > 0 else 0.5

    return {
        'c_for_benefit': float(c_benefit),
        'n_pairs_evaluated': int(n_pairs),
        'n_concordant_pairs': int(concordant),
        'n_discordant_pairs': int(discordant),
        'n_tied_pairs': int(tied),
        'interpretation': (
            f'C-for-benefit = {c_benefit:.3f}. '
            f'Values > 0.5 indicate the model correctly ranks '
            f'who benefits more from NIRS. '
            f'({concordant} concordant, {discordant} discordant, {tied} tied '
            f'out of {n_pairs} cross-arm pairs).'
        )
    }


def compute_all_metrics(pred_outputs, vfd_observed, delta, treatment_observed,
                        ite_true=None):
    ite_pred = pred_outputs['ite'].squeeze()
    p_surv_obs = np.where(
        treatment_observed == 1,
        pred_outputs['p_surv_1'].squeeze(),
        pred_outputs['p_surv_0'].squeeze(),
    )

    results = {}

    results.update(policy_value(ite_pred, vfd_observed, treatment_observed))

    results['brier_score'] = survival_calibration(p_surv_obs, delta)

    if len(np.unique(delta)) > 1:
        results['surv_auroc'] = roc_auc_score(delta, p_surv_obs)
        results['surv_auprc'] = average_precision_score(delta, p_surv_obs)

    results['mean_ite'] = float(np.mean(ite_pred))
    results['std_ite'] = float(np.std(ite_pred))
    results['pct_nirs_beneficial'] = float((ite_pred > 0).mean() * 100)
    results['pct_imv_beneficial'] = float((ite_pred < 0).mean() * 100)

    results['mean_ite_survival'] = float(np.mean(pred_outputs['ite_survival'].squeeze()))
    results['mean_ite_vfd_cond'] = float(np.mean(pred_outputs['ite_vfd_cond'].squeeze()))

    cfb = c_for_benefit(ite_pred, vfd_observed, treatment_observed)
    results['c_for_benefit'] = cfb['c_for_benefit']
    results['c_for_benefit_interpretation'] = cfb['interpretation']

    if ite_true is not None:
        results['pehe'] = pehe(ite_pred, ite_true)
        results['ate_bias'] = ate_bias(ite_pred, ite_true)

    e_value_results = compute_e_value_for_ate(
        ate=results['mean_ite'],
        outcome_std=float(np.std(vfd_observed)),
    )
    results['e_value'] = e_value_results['e_value']
    results['approx_risk_ratio'] = e_value_results['approx_risk_ratio']
    results['e_value_interpretation'] = e_value_results['interpretation']

    rosenbaum = rosenbaum_sensitivity_bounds(ite_pred, treatment_observed, vfd_observed)
    results['rosenbaum_critical_gamma'] = rosenbaum['critical_gamma']
    results['rosenbaum_interpretation'] = rosenbaum['interpretation']

    return results


def compute_e_value(risk_ratio):
    if risk_ratio < 1:
        risk_ratio = 1.0 / risk_ratio
    if risk_ratio <= 1.0:
        return 1.0
    return risk_ratio + np.sqrt(risk_ratio * (risk_ratio - 1.0))


def compute_e_value_for_ate(ate, outcome_std, treatment_prevalence=0.5):
    d = ate / max(outcome_std, 1e-6)

    log_rr = d * np.pi / np.sqrt(3)
    approx_rr = np.exp(abs(log_rr))

    e_val = compute_e_value(approx_rr)

    return {
        'e_value': float(e_val),
        'approx_risk_ratio': float(approx_rr),
        'standardized_effect': float(d),
        'interpretation': (
            f'To explain away the observed ATE of {ate:.3f} VFD-28 days, '
            f'an unmeasured confounder would need to be associated with both '
            f'treatment and outcome by a risk ratio of at least {e_val:.2f}.'
        ),
    }


def rosenbaum_sensitivity_bounds(ite_pred, treatment, vfd_observed,
                                  gamma_values=None):
    if gamma_values is None:
        gamma_values = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]

    treated_mask = treatment == 1
    control_mask = treatment == 0
    obs_ate = (vfd_observed[treated_mask].mean()
               - vfd_observed[control_mask].mean())

    results = []
    for gamma in gamma_values:
        bias_factor = np.log(gamma)
        ate_lower = obs_ate - bias_factor * np.std(vfd_observed) / np.sqrt(len(vfd_observed))
        ate_upper = obs_ate + bias_factor * np.std(vfd_observed) / np.sqrt(len(vfd_observed))

        results.append({
            'gamma': gamma,
            'ate_lower_bound': float(ate_lower),
            'ate_upper_bound': float(ate_upper),
            'effect_robust': ate_lower > 0,
        })

    critical_gamma = None
    for r in results:
        if not r['effect_robust']:
            critical_gamma = r['gamma']
            break

    return {
        'observed_ate': float(obs_ate),
        'bounds': results,
        'critical_gamma': critical_gamma,
        'interpretation': (
            f'The observed ATE of {obs_ate:.3f} remains significant up to '
            f'Gamma={critical_gamma if critical_gamma else ">3.0"}, meaning '
            f'an unmeasured confounder would need to change treatment odds by '
            f'at least {critical_gamma if critical_gamma else ">3x"} to nullify the effect.'
        ),
    }


def plot_model_comparison_bars(results_dict, metric_key, metric_label,
                               save_path=None):
    import matplotlib.pyplot as plt

    models = list(results_dict.keys())
    means = [results_dict[m]['mean'] for m in models]
    stds = [results_dict[m]['std'] for m in models]

    colors = ['#b0b0b0'] * (len(models) - 1) + ['#2196F3']

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(models, means, yerr=stds, capsize=5, color=colors,
                  edgecolor='black', linewidth=0.8)

    ax.set_ylabel(metric_label, fontsize=13)
    ax.set_title(f'Model Comparison: {metric_label}', fontsize=14)
    ax.tick_params(axis='x', rotation=30)
    ax.grid(axis='y', alpha=0.3)

    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 0.01,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_ite_distribution(ite_pred, model_name='VT-NIRS', save_path=None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(ite_pred, bins=50, density=True, alpha=0.7, color='steelblue',
            edgecolor='white', linewidth=0.5, label='ITE distribution')

    from scipy.stats import gaussian_kde
    kde = gaussian_kde(ite_pred)
    x_range = np.linspace(ite_pred.min(), ite_pred.max(), 200)
    ax.plot(x_range, kde(x_range), color='navy', linewidth=2, label='KDE')

    mean_ite = np.mean(ite_pred)
    ax.axvline(mean_ite, color='orange', linewidth=2, linestyle='--',
               label=f'Mean ITE = {mean_ite:.3f}')

    ax.axvline(0, color='red', linewidth=1.5, linestyle=':',
               label='Equipoise (ITE=0)')

    ax.fill_betweenx([0, ax.get_ylim()[1] * 0.05], ite_pred.min(), 0,
                      alpha=0.1, color='red', label='IMV beneficial')
    ax.fill_betweenx([0, ax.get_ylim()[1] * 0.05], 0, ite_pred.max(),
                      alpha=0.1, color='green', label='NIRS beneficial')

    ax.set_xlabel('ITE (VFD-28 days: positive = NIRS better)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'{model_name} — Individualized Treatment Effect Distribution',
                 fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_training_curves(train_log, save_path=None):
    import matplotlib.pyplot as plt

    epochs = train_log['epoch']
    n_plots = len([k for k in train_log if k != 'epoch'])

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    plot_keys = [k for k in train_log if k != 'epoch']
    colors = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0', '#FF9800', '#607D8B']

    for idx, (key, color) in enumerate(zip(plot_keys[:6], colors)):
        ax = axes[idx]
        ax.plot(epochs, train_log[key], color=color, linewidth=1.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(key)
        ax.set_title(key, fontsize=11)
        ax.grid(alpha=0.3)

    for idx in range(len(plot_keys), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('VT-NIRS Training Curves', fontsize=14, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_decomposed_ite_scatter(pred_outputs, save_path=None):
    import matplotlib.pyplot as plt

    ite_surv = pred_outputs['ite_survival'].squeeze()
    ite_vfd = pred_outputs['ite_vfd_cond'].squeeze()

    fig, ax = plt.subplots(figsize=(8, 8))

    scatter = ax.scatter(ite_surv, ite_vfd, c=ite_surv + ite_vfd,
                         cmap='RdYlGn', alpha=0.5, s=20, edgecolors='none')

    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    ax.axvline(0, color='gray', linewidth=0.8, linestyle='--')

    ax.text(0.95, 0.95, 'NIRS: survival + VFD', transform=ax.transAxes,
            ha='right', va='top', fontsize=10, color='green', weight='bold')
    ax.text(0.05, 0.05, 'IMV: survival + VFD', transform=ax.transAxes,
            ha='left', va='bottom', fontsize=10, color='red', weight='bold')
    ax.text(0.95, 0.05, 'NIRS: survival only', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=10, color='orange')
    ax.text(0.05, 0.95, 'IMV: survival, NIRS: VFD', transform=ax.transAxes,
            ha='left', va='top', fontsize=10, color='purple')

    ax.set_xlabel('ΔP(survive) — Survival ITE (NIRS − IMV)', fontsize=12)
    ax.set_ylabel('ΔVFD|survive — Conditional VFD ITE (NIRS − IMV)', fontsize=12)
    ax.set_title('Decomposed Treatment Effect: Survival vs Ventilation Duration',
                 fontsize=13)

    plt.colorbar(scatter, ax=ax, label='Total ITE direction')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_subgroup_ite_trends(ite_pred, covariate_values, covariate_name,
                              n_bins=5, save_path=None):
    import matplotlib.pyplot as plt

    bin_edges = np.percentile(covariate_values, np.linspace(0, 100, n_bins + 1))
    bin_labels = []
    bin_means = []
    bin_stds = []
    bin_centers = []

    for i in range(n_bins):
        mask = (covariate_values >= bin_edges[i]) & (covariate_values < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (covariate_values >= bin_edges[i]) & (covariate_values <= bin_edges[i + 1])

        if mask.sum() > 0:
            bin_means.append(np.mean(ite_pred[mask]))
            bin_stds.append(np.std(ite_pred[mask]) / np.sqrt(mask.sum()))
            bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            bin_labels.append(f'{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}')

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.errorbar(bin_centers, bin_means, yerr=bin_stds, fmt='o-',
                color='#2196F3', linewidth=2, markersize=8, capsize=5,
                label='Mean ITE ± SE')

    ax.axhline(0, color='red', linewidth=1, linestyle=':', label='Equipoise')
    ax.fill_between(bin_centers, 0, bin_means, alpha=0.1,
                     color=np.where(np.array(bin_means) > 0, 'green', 'red'))

    ax.set_xlabel(covariate_name, fontsize=12)
    ax.set_ylabel('ITE (VFD-28 days)', fontsize=12)
    ax.set_title(f'Treatment Effect Heterogeneity by {covariate_name}', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig
