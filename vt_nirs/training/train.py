"""
train.py — Two-Stage Adversarial Training Pipeline for VT-NIRS
================================================================
Implements the full training loop following GANITE's two-stage procedure,
adapted for our censoring-aware survival-decomposed architecture.

Training structure follows:
  # Ref: DT_ITE_Final.ipynb lines 2820-2870: GANITE training loop
  #      — Stage 1: Generator + Discriminator adversarial training
  #      — Stage 2: ITEPredictor training on Generator's pseudo-labels
  # Ref: graphspa training/01a_HiRID_baseline.ipynb:
  #      train/val/test per epoch, best model checkpoint, LR scheduler
  # Ref: mcem: PyTorch Lightning with multi-objective loss

Stage 1 (adversarial):
  For each batch:
    1. Encode patient sequences → embeddings
    2. Generator produces counterfactual outcomes (VFD-28 decomposed)
    3. Discriminator trained to distinguish real vs generated
    4. Generator trained to fool discriminator + match observed outcomes
    5. Survival and VFD losses enforce censoring-aware learning

Stage 2 (prediction):
  For each batch:
    1. Freeze Generator
    2. Generator produces pseudo-labels for counterfactual outcomes
    3. ITEPredictor trained to match Generator's outputs
    4. Consistency loss ensures predictor aligns with generator
"""

import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vt_nirs import VTNIRSModel
from utils.losses import CensoringAwareAdversarialLoss
from utils.metrics import compute_all_metrics


DEFAULT_CONFIG = {
    'n_covariates': 23,
    'd_model': 128,
    'n_heads': 4,
    'n_layers': 4,
    'd_ff': 256,
    'noise_dim': 8,
    'hidden_dim': 128,
    'dropout': 0.1,

    'epochs_stage1': 100,
    'lr_generator': 2e-4,
    'lr_discriminator': 2e-4,
    'lr_encoder': 2e-4,
    'weight_decay': 1e-4,

    'epochs_stage2': 50,
    'lr_predictor': 2e-4,

    'lambda_adv': 1.0,
    'lambda_surv': 1.0,
    'lambda_vfd': 1.0,
    'lambda_consist': 0.5,
    'lambda_gate': 0.1,
    'lambda_ipm': 1.0,
    'lambda_dr': 0.5,
    'lambda_gp': 10.0,

    'batch_size': 128,
    'random_state': 42,

    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'save_dir': 'checkpoints/',
    'patience': 10,
}


def train_stage1(model, train_loader, val_loader, loss_fn, config, save_dir):
    """
    Stage 1: Adversarial training of Encoder + Generator + Discriminator.

    # Ref: Yoon et al. ICML 2018, Section 3, Algorithm 1 —
    #      alternating Generator and Discriminator updates.
    # Ref: DT_ITE_Final.ipynb lines 2820-2870: training loop structure
    # Ref: graphspa training/01a_HiRID_baseline.ipynb:
    #      epoch loop with train/val phases, LR scheduler, checkpointing.
    """
    device = config['device']
    model = model.to(device)

    params_gen = (list(model.encoder.parameters())
                  + list(model.generator.parameters())
                  + list(model.propensity_head.parameters()))
    params_disc = list(model.discriminator.parameters())

    opt_gen = torch.optim.Adam(params_gen, lr=config['lr_generator'],
                                weight_decay=config['weight_decay'])
    opt_disc = torch.optim.Adam(params_disc, lr=config['lr_discriminator'],
                                 weight_decay=config['weight_decay'])

    scheduler_gen = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt_gen, mode='min', factor=0.5, patience=5, verbose=True)
    scheduler_disc = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt_disc, mode='min', factor=0.5, patience=5, verbose=True)

    train_log = defaultdict(list)
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, config['epochs_stage1'] + 1):
        model.train()
        epoch_losses = defaultdict(float)
        n_batches = 0

        for batch in train_loader:
            x = batch['x'].to(device)
            treatment = batch['treatment'].to(device)
            vfd = batch['vfd'].to(device)
            delta = batch['delta'].to(device)
            pad_mask = batch['pad_mask'].to(device)

            bs = x.size(0)
            noise = torch.randn(bs, config['noise_dim'], device=device)

            gen_outputs, enc_outputs = model.forward_generator(
                x, treatment, pad_mask, noise)
            emb, emb_surv, emb_vfd, gate, attn_out, prop_logits = enc_outputs

            opt_disc.zero_grad()

            real_outcomes = _build_real_outcomes(gen_outputs, treatment, vfd, delta)
            p_real = model.forward_discriminator(emb, real_outcomes)

            p_fake = model.forward_discriminator(emb, gen_outputs)

            loss_D = loss_fn.adversarial_loss_discriminator(p_real, p_fake)

            real_out_tensor = torch.cat([
                real_outcomes['p_surv_0'], real_outcomes['vfd_cond_0'],
                real_outcomes['vfd_0'], real_outcomes['p_surv_1'],
                real_outcomes['vfd_cond_1'], real_outcomes['vfd_1']
            ], dim=-1)
            fake_out_tensor = torch.cat([
                gen_outputs['p_surv_0'], gen_outputs['vfd_cond_0'],
                gen_outputs['vfd_0'], gen_outputs['p_surv_1'],
                gen_outputs['vfd_cond_1'], gen_outputs['vfd_1']
            ], dim=-1)
            gp = model.discriminator.gradient_penalty(
                emb, real_out_tensor.detach(), fake_out_tensor.detach(),
                lambda_gp=config.get('lambda_gp', 10.0))

            (loss_D + gp).backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(params_disc, max_norm=1.0)
            opt_disc.step()

            opt_gen.zero_grad()

            p_fake_for_G = model.forward_discriminator(emb, gen_outputs)

            loss_G, loss_dict = loss_fn.generator_loss(
                p_real_fake=p_fake_for_G,
                gen_outputs=gen_outputs,
                observed_treatment=treatment,
                vfd_observed=vfd,
                delta=delta,
                gate=gate,
                emb=emb,
                propensity_logits=prop_logits,
            )
            loss_G.backward()
            torch.nn.utils.clip_grad_norm_(params_gen, max_norm=1.0)
            opt_gen.step()

            epoch_losses['l_D'] += loss_D.item()
            epoch_losses['l_gp'] += gp.item()
            for k, v in loss_dict.items():
                epoch_losses[k] += v
            n_batches += 1

        for k in epoch_losses:
            epoch_losses[k] /= n_batches

        model.eval()
        val_losses = defaultdict(float)
        n_val = 0

        with torch.no_grad():
            for batch in val_loader:
                x = batch['x'].to(device)
                treatment = batch['treatment'].to(device)
                vfd = batch['vfd'].to(device)
                delta = batch['delta'].to(device)
                pad_mask = batch['pad_mask'].to(device)
                bs = x.size(0)
                noise = torch.randn(bs, config['noise_dim'], device=device)

                gen_outputs, enc_outputs = model.forward_generator(
                    x, treatment, pad_mask, noise)
                emb, _, _, gate, _, prop_logits = enc_outputs

                p_fake = model.forward_discriminator(emb, gen_outputs)

                _, loss_dict = loss_fn.generator_loss(
                    p_fake, gen_outputs, treatment, vfd, delta, gate,
                    emb=emb, propensity_logits=prop_logits)

                for k, v in loss_dict.items():
                    val_losses[k] += v
                n_val += 1

        for k in val_losses:
            val_losses[k] /= max(n_val, 1)

        train_log['epoch'].append(epoch)
        for k in epoch_losses:
            train_log[f'train_{k}'].append(epoch_losses[k])
        for k in val_losses:
            train_log[f'val_{k}'].append(val_losses[k])

        val_total = val_losses.get('l_total_G', float('inf'))
        scheduler_gen.step(val_total)
        scheduler_disc.step(val_total)

        if val_total < best_val_loss:
            best_val_loss = val_total
            patience_counter = 0
            torch.save(model.state_dict(),
                       os.path.join(save_dir, 'best_stage1.pth'))
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f'Epoch {epoch:03d}/{config["epochs_stage1"]}  |  '
                  f'G={epoch_losses["l_total_G"]:.4f}  '
                  f'D={epoch_losses["l_D"]:.4f}  '
                  f'surv={epoch_losses["l_surv"]:.4f}  '
                  f'vfd={epoch_losses["l_vfd"]:.4f}  '
                  f'mmd={epoch_losses.get("l_mmd", 0):.4f}  '
                  f'prop={epoch_losses.get("l_prop", 0):.4f}  |  '
                  f'val_G={val_total:.4f}')

        if patience_counter >= config['patience']:
            print(f'Early stopping at epoch {epoch}')
            break

    model.load_state_dict(
        torch.load(os.path.join(save_dir, 'best_stage1.pth'),
                    map_location=device))

    return model, dict(train_log)


def train_stage2(model, train_loader, val_loader, loss_fn, config, save_dir):
    """
    Stage 2: Train ITEPredictor on Generator's pseudo-labels.

    # Ref: Yoon et al. ICML 2018, Section 3.3:
    #      "Given a new sample x, we use the ITE generator [predictor]
    #       to directly estimate the ITE."
    # Ref: DT_ITE_Final.ipynb lines 2857-2868: predictor training loop
    """
    device = config['device']
    model = model.to(device)

    for p in model.encoder.parameters():
        p.requires_grad = False
    for p in model.generator.parameters():
        p.requires_grad = False
    for p in model.discriminator.parameters():
        p.requires_grad = False

    opt_pred = torch.optim.Adam(
        model.predictor.parameters(),
        lr=config['lr_predictor'],
        weight_decay=config['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt_pred, mode='min', factor=0.5, patience=5, verbose=True)

    train_log = defaultdict(list)
    best_val_loss = float('inf')

    for epoch in range(1, config['epochs_stage2'] + 1):
        model.predictor.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            x = batch['x'].to(device)
            treatment = batch['treatment'].to(device)
            pad_mask = batch['pad_mask'].to(device)
            vfd = batch['vfd'].to(device)
            delta = batch['delta'].to(device)
            bs = x.size(0)

            opt_pred.zero_grad()

            with torch.no_grad():
                noise = torch.randn(bs, config['noise_dim'], device=device)
                gen_outputs, enc_outputs = model.forward_generator(
                    x, treatment, pad_mask, noise)
                emb = enc_outputs[0]
                prop_logits = enc_outputs[5]
                prop_scores = torch.sigmoid(prop_logits)

            pred_outputs = model.predictor(emb)

            loss, _ = loss_fn.predictor_loss(
                gen_outputs, pred_outputs, treatment,
                propensity_scores=prop_scores,
                vfd_observed=vfd,
                delta=delta,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.predictor.parameters(), 1.0)
            opt_pred.step()

            epoch_loss += loss.item()
            n_batches += 1

        epoch_loss /= n_batches

        model.predictor.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch['x'].to(device)
                treatment = batch['treatment'].to(device)
                pad_mask = batch['pad_mask'].to(device)
                vfd = batch['vfd'].to(device)
                delta = batch['delta'].to(device)
                bs = x.size(0)

                noise = torch.randn(bs, config['noise_dim'], device=device)
                gen_outputs, enc_outputs = model.forward_generator(
                    x, treatment, pad_mask, noise)
                pred_outputs = model.predictor(enc_outputs[0])
                prop_scores = torch.sigmoid(enc_outputs[5])
                loss, _ = loss_fn.predictor_loss(
                    gen_outputs, pred_outputs, treatment,
                    propensity_scores=prop_scores,
                    vfd_observed=vfd, delta=delta)
                val_loss += loss.item()
                n_val += 1

        val_loss /= max(n_val, 1)
        scheduler.step(val_loss)

        train_log['epoch'].append(epoch)
        train_log['train_l_consist'].append(epoch_loss)
        train_log['val_l_consist'].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(),
                       os.path.join(save_dir, 'best_stage2.pth'))

        if epoch % 10 == 0 or epoch == 1:
            print(f'Stage 2 Epoch {epoch:03d}/{config["epochs_stage2"]}  |  '
                  f'train={epoch_loss:.4f}  val={val_loss:.4f}')

    model.load_state_dict(
        torch.load(os.path.join(save_dir, 'best_stage2.pth'),
                    map_location=device))
    for p in model.parameters():
        p.requires_grad = True

    return model, dict(train_log)


def evaluate(model, test_loader, config):
    """
    Full evaluation on test set.

    # Ref: graphspa training notebooks: validate function with
    #      metrics computation on held-out test set.
    """
    device = config['device']
    model = model.to(device)
    model.eval()

    all_preds = defaultdict(list)
    all_vfd = []
    all_delta = []
    all_treatment = []

    with torch.no_grad():
        for batch in test_loader:
            x = batch['x'].to(device)
            pad_mask = batch['pad_mask'].to(device)

            pred_outputs, _ = model.forward_predictor(x, pad_mask)

            for k, v in pred_outputs.items():
                all_preds[k].append(v.cpu().numpy())
            all_vfd.append(batch['vfd'].numpy())
            all_delta.append(batch['delta'].numpy())
            all_treatment.append(batch['treatment'].numpy())

    for k in all_preds:
        all_preds[k] = np.concatenate(all_preds[k], axis=0)
    vfd_obs = np.concatenate(all_vfd, axis=0).squeeze()
    delta = np.concatenate(all_delta, axis=0).squeeze()
    treatment = np.concatenate(all_treatment, axis=0).squeeze()

    metrics = compute_all_metrics(all_preds, vfd_obs, delta, treatment)

    return metrics, all_preds


def _build_real_outcomes(gen_outputs, treatment, vfd_observed, delta):
    """
    Build "real" outcome tensor for discriminator training.

    For the observed treatment arm: use actual observed values.
    For the counterfactual arm: use Generator's estimates (this is
    the key insight of GANITE — we don't have counterfactual ground truth).

    # Ref: Yoon et al. ICML 2018, Section 3.2:
    #      Discriminator sees (x, y_factual, y_counterfactual)
    #      where y_factual is real and y_counterfactual is from G.
    """
    t = treatment.squeeze(-1)

    real_p_surv = delta
    real_vfd_cond = vfd_observed
    real_vfd = delta * vfd_observed

    p_surv_0 = torch.where(t.unsqueeze(-1) == 0, real_p_surv, gen_outputs['p_surv_0'])
    vfd_cond_0 = torch.where(t.unsqueeze(-1) == 0, real_vfd_cond, gen_outputs['vfd_cond_0'])
    vfd_0 = torch.where(t.unsqueeze(-1) == 0, real_vfd, gen_outputs['vfd_0'])

    p_surv_1 = torch.where(t.unsqueeze(-1) == 1, real_p_surv, gen_outputs['p_surv_1'])
    vfd_cond_1 = torch.where(t.unsqueeze(-1) == 1, real_vfd_cond, gen_outputs['vfd_cond_1'])
    vfd_1 = torch.where(t.unsqueeze(-1) == 1, real_vfd, gen_outputs['vfd_1'])

    return {
        'p_surv_0': p_surv_0, 'vfd_cond_0': vfd_cond_0, 'vfd_0': vfd_0,
        'p_surv_1': p_surv_1, 'vfd_cond_1': vfd_cond_1, 'vfd_1': vfd_1,
    }


def run_full_pipeline(train_loader, val_loader, test_loader, config=None):
    """
    Run the complete VT-NIRS training and evaluation pipeline.

    # Ref: graphspa training notebooks: main_work() function
    #      orchestrating train → validate → test → save.

    Args:
        train_loader, val_loader, test_loader: DataLoaders
        config: Configuration dict (uses DEFAULT_CONFIG if None)
    Returns:
        model: Trained VTNIRSModel
        metrics: Test set evaluation results
        train_logs: Training history (Stage 1 + Stage 2)
    """
    if config is None:
        config = DEFAULT_CONFIG.copy()

    save_dir = config['save_dir']
    os.makedirs(save_dir, exist_ok=True)

    print('=' * 60)
    print('VT-NIRS: Virtual Twin for Non-Invasive Respiratory Support')
    print('=' * 60)
    print(f'Device: {config["device"]}')
    print(f'Architecture: d_model={config["d_model"]}, '
          f'n_layers={config["n_layers"]}, n_heads={config["n_heads"]}')
    print(f'Stage 1: {config["epochs_stage1"]} epochs | '
          f'Stage 2: {config["epochs_stage2"]} epochs')
    print()

    model = VTNIRSModel(
        n_covariates=config['n_covariates'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        d_ff=config['d_ff'],
        noise_dim=config['noise_dim'],
        hidden_dim=config['hidden_dim'],
        dropout=config['dropout'],
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total_params:,}')
    print()

    loss_fn = CensoringAwareAdversarialLoss(
        lambda_adv=config['lambda_adv'],
        lambda_surv=config['lambda_surv'],
        lambda_vfd=config['lambda_vfd'],
        lambda_consist=config['lambda_consist'],
        lambda_gate=config['lambda_gate'],
        lambda_ipm=config.get('lambda_ipm', 1.0),
        lambda_dr=config.get('lambda_dr', 0.5),
    )

    print('--- Stage 1: Adversarial Training ---')
    model, log1 = train_stage1(model, train_loader, val_loader,
                                loss_fn, config, save_dir)
    print()

    print('--- Stage 2: ITE Predictor Training ---')
    model, log2 = train_stage2(model, train_loader, val_loader,
                                loss_fn, config, save_dir)
    print()

    print('--- Evaluation ---')
    metrics, predictions = evaluate(model, test_loader, config)

    print(f'\nResults:')
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f'  {k}: {v:.4f}')
        else:
            print(f'  {k}: {v}')

    with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
        json.dump({k: float(v) if isinstance(v, (float, np.floating)) else v
                    for k, v in metrics.items()}, f, indent=2)

    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump({k: str(v) if not isinstance(v, (int, float, str, bool)) else v
                    for k, v in config.items()}, f, indent=2)

    train_logs = {'stage1': log1, 'stage2': log2}

    return model, metrics, train_logs
