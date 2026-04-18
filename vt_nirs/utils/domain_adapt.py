

import numpy as np
import torch
import torch.nn as nn


def recalibrate_propensity(model, target_loader, device='cuda'):
    model.eval()
    all_logits = []
    all_treatments = []

    with torch.no_grad():
        for batch in target_loader:
            x = batch['x'].to(device)
            pad_mask = batch['pad_mask'].to(device)
            treatment = batch['treatment']

            emb, _, _, _, _ = model.encoder(x, pad_mask)
            logits = model.propensity_head(emb)

            all_logits.append(logits.cpu())
            all_treatments.append(treatment)

    logits = torch.cat(all_logits, dim=0)
    treatments = torch.cat(all_treatments, dim=0)

    a = nn.Parameter(torch.ones(1))
    b = nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.LBFGS([a, b], lr=0.1, max_iter=100)

    def closure():
        optimizer.zero_grad()
        calibrated = torch.sigmoid(a * logits + b)
        loss = nn.BCELoss()(calibrated, treatments)
        loss.backward()
        return loss

    optimizer.step(closure)

    print(f'[Domain Adaptation] Platt scaling: a={a.item():.4f}, b={b.item():.4f}')
    return (a.item(), b.item())


def compute_importance_weights(source_features, target_features, method='kliep'):
    from sklearn.linear_model import LogisticRegression

    if method == 'logistic':
        n_s = source_features.shape[0]
        n_t = target_features.shape[0]

        X = np.vstack([source_features, target_features])
        y = np.concatenate([np.zeros(n_s), np.ones(n_t)])

        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(X, y)

        probs = clf.predict_proba(source_features)[:, 1]
        weights = probs / (1 - probs + 1e-8)

        weights = weights * n_s / weights.sum()

        return weights

    elif method == 'kliep':
        from sklearn.metrics.pairwise import rbf_kernel

        n_basis = min(100, target_features.shape[0])
        idx = np.random.choice(target_features.shape[0], n_basis, replace=False)
        basis = target_features[idx]

        gamma = 1.0 / (2 * np.median(
            np.linalg.norm(source_features[:100] - source_features[1:101], axis=1)) ** 2 + 1e-6)

        K_source = rbf_kernel(source_features, basis, gamma=gamma)
        K_target = rbf_kernel(target_features, basis, gamma=gamma)

        alpha = K_target.mean(axis=0)
        weights = K_source @ alpha
        weights = np.maximum(weights, 1e-6)
        weights = weights * len(weights) / weights.sum()

        return weights

    else:
        raise ValueError(f'Unknown method: {method}')


def apply_domain_adaptation(model, source_loader, target_loader,
                            config, method='propensity_recalib'):
    device = config.get('device', 'cuda')

    if method == 'propensity_recalib':
        params = recalibrate_propensity(model, target_loader, device)
        print(f'[Domain Adaptation] Propensity re-calibrated for target domain')
        return params

    elif method == 'fine_tune':
        print('[Domain Adaptation] Fine-tuning predictor head on target domain...')

        for p in model.encoder.parameters():
            p.requires_grad = False
        for p in model.generator.parameters():
            p.requires_grad = False
        for p in model.discriminator.parameters():
            p.requires_grad = False
        for p in model.propensity_head.parameters():
            p.requires_grad = False

        opt = torch.optim.Adam(model.predictor.parameters(), lr=1e-5)

        model.predictor.train()
        for epoch in range(10):
            for batch in target_loader:
                x = batch['x'].to(device)
                pad_mask = batch['pad_mask'].to(device)
                treatment = batch['treatment'].to(device)
                vfd = batch['vfd'].to(device)

                with torch.no_grad():
                    emb, _, _, _, _ = model.encoder(x, pad_mask)
                pred = model.predictor(emb)

                t = treatment.squeeze(-1)
                pred_vfd_obs = torch.where(
                    t.unsqueeze(-1) == 1, pred['vfd_1'], pred['vfd_0'])
                loss = nn.MSELoss()(pred_vfd_obs, vfd)

                opt.zero_grad()
                loss.backward()
                opt.step()

        for p in model.parameters():
            p.requires_grad = True

        print('[Domain Adaptation] Predictor fine-tuned on target domain')
        return model

    else:
        raise ValueError(f'Unknown DA method: {method}')
