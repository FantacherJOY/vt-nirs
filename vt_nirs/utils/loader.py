

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


COVARIATES_23 = [
    'age',
    'gender',
    'bmi',

    'heart_rate',
    'resp_rate',
    'spo2',
    'mbp',
    'temperature',
    'fio2',
    'peep',

    'pao2',
    'paco2',
    'ph',
    'pf_ratio',

    'lactate',
    'creatinine',
    'bilirubin',
    'platelets',
    'wbc',

    'sofa_score',
    'gcs_total',

    'hours_since_icu_admit',
    'vasopressor_flag',
]

MIMIC_ITEMIDS = {
    'heart_rate': [220045],
    'resp_rate': [220210, 224690],
    'spo2': [220277],
    'mbp': [220052, 220181, 225312],
    'temperature': [223761, 223762],
    'fio2': [223835],
    'peep': [220339, 224700],
    'pao2': [220224],
    'paco2': [220235],
    'ph': [220274, 220734],
    'lactate': [225668],
    'creatinine': [220615],
    'bilirubin': [225690],
    'platelets': [227457],
    'wbc': [220546],
    'gcs_total': [220739, 223900, 223901],
}


def compute_vfd28(vent_duration_days, survived_28d):
    """
    Compute Ventilator-Free Days at 28 days (VFD-28).

    Definition:
      - If patient survives to day 28: VFD-28 = max(0, 28 - vent_duration_days)
      - If patient dies before day 28: VFD-28 = 0

    # Ref: Schoenfeld DA et al. "Statistical Evaluation of Ventilator-Free Days
    #      as an Efficacy Measure in Clinical Trials of Treatments for Acute
    #      Respiratory Distress Syndrome." Crit Care Med 2002.
    #      — Original definition of VFD-28.
    # Ref: Yehya N et al. AJRCCM 2019 — discusses limitations of VFD composite.

    NOTE: This INCLUDES death patients (VFD=0). This is intentional:
    our survival-decomposed architecture handles the censoring directly,
    rather than excluding deaths like the AMIA binary mortality approach.

    Args:
        vent_duration_days: (N,) — total days on mechanical ventilation
        survived_28d: (N,) — 1 if survived to day 28, 0 if died
    Returns:
        vfd28: (N,) — VFD-28 values in [0, 28]
        delta: (N,) — survival indicator (same as survived_28d)
    """
    vfd28 = np.where(
        survived_28d == 1,
        np.maximum(0, 28 - vent_duration_days),
        0.0
    )
    return vfd28.astype(np.float32), survived_28d.astype(np.float32)


class NIRSTwinDataset(Dataset):
    """
    PyTorch Dataset for VT-NIRS training.

    Each sample: (time_series, treatment, vfd_observed, delta, static_covariates)

    Structure follows:
    # Ref: graphspa training/loader.py ICUVariableLengthDataset:
    #      returns (data, pad_mask, label) tuples from HDF5.
    # Ref: DT_ITE_Final.ipynb line 2343-2346: TensorDataset + DataLoader

    Args:
        sequences: (N, T, D) — time series of D covariates over T timesteps
        treatments: (N,) — 0=IMV, 1=NIRS
        vfd_observed: (N,) — observed VFD-28
        delta: (N,) — 1=survived, 0=died
        pad_masks: (N, T) — True where padded (optional)
    """

    def __init__(self, sequences, treatments, vfd_observed, delta, pad_masks=None):
        self.sequences = torch.FloatTensor(sequences)
        self.treatments = torch.FloatTensor(treatments).unsqueeze(-1)
        self.vfd_observed = torch.FloatTensor(vfd_observed).unsqueeze(-1)
        self.delta = torch.FloatTensor(delta).unsqueeze(-1)

        if pad_masks is not None:
            self.pad_masks = torch.BoolTensor(pad_masks)
        else:
            self.pad_masks = torch.zeros(len(sequences), sequences.shape[1],
                                         dtype=torch.bool)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {
            'x': self.sequences[idx],
            'treatment': self.treatments[idx],
            'vfd': self.vfd_observed[idx],
            'delta': self.delta[idx],
            'pad_mask': self.pad_masks[idx],
        }


def propensity_score_matching(df, treatment_col, covariate_cols,
                               caliper=0.05, random_state=42):
    """
    Propensity Score Matching with caliper.

    Reuses the PSM pipeline from the AMIA submission:
    # Ref: DT_ITE_Final.ipynb lines 1432-1460: LogisticRegression PSM
    #      with caliper-based nearest-neighbor matching.

    Args:
        df: DataFrame with covariates and treatment column
        treatment_col: Name of treatment column
        covariate_cols: List of covariate column names for PS model
        caliper: Maximum PS difference for matching (default: 0.05)
        random_state: Random seed
    Returns:
        df_matched: Matched DataFrame
        ps_model: Fitted propensity score model
    """
    X = df[covariate_cols].values
    W = df[treatment_col].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    ps_model = LogisticRegression(max_iter=1000, random_state=random_state)
    ps_model.fit(X_scaled, W)

    ps = ps_model.predict_proba(X_scaled)[:, 1]
    df = df.copy()
    df['propensity_score'] = ps

    treated = df[df[treatment_col] == 1].copy()
    control = df[df[treatment_col] == 0].copy()

    matched_indices = []
    used_controls = set()

    for idx, row in treated.iterrows():
        ps_treat = row['propensity_score']
        candidates = control[~control.index.isin(used_controls)]

        if len(candidates) == 0:
            continue

        distances = np.abs(candidates['propensity_score'] - ps_treat)
        best_idx = distances.idxmin()
        best_dist = distances.min()

        if best_dist <= caliper:
            matched_indices.append(idx)
            matched_indices.append(best_idx)
            used_controls.add(best_idx)

    df_matched = df.loc[matched_indices].copy()

    print(f'PSM: {len(treated)} treated, {len(control)} control → '
          f'{len(df_matched)//2} matched pairs ({len(df_matched)} total)')

    return df_matched, ps_model


def create_dataloaders(
    sequences,
    treatments,
    vfd_observed,
    delta,
    pad_masks=None,
    batch_size=128,
    val_fraction=0.15,
    test_fraction=0.15,
    random_state=42,
):
    """
    Create train/val/test DataLoaders.

    Split strategy: random patient-level split (following graphspa).
    # Ref: graphspa training/loader.py: 70:15:15 train/val/test splits
    # Ref: DT_ITE_Final.ipynb: uses full PSM cohort for training

    Args:
        sequences: (N, T, D) numpy array
        treatments: (N,) numpy array
        vfd_observed: (N,) numpy array
        delta: (N,) numpy array
        pad_masks: (N, T) numpy array (optional)
        batch_size: Batch size
        val_fraction, test_fraction: Split proportions
        random_state: Random seed
    Returns:
        train_loader, val_loader, test_loader
    """
    N = len(sequences)
    rng = np.random.RandomState(random_state)
    indices = rng.permutation(N)

    n_test = int(N * test_fraction)
    n_val = int(N * val_fraction)

    test_idx = indices[:n_test]
    val_idx = indices[n_test:n_test + n_val]
    train_idx = indices[n_test + n_val:]

    def make_loader(idx, shuffle):
        ds = NIRSTwinDataset(
            sequences=sequences[idx],
            treatments=treatments[idx],
            vfd_observed=vfd_observed[idx],
            delta=delta[idx],
            pad_masks=pad_masks[idx] if pad_masks is not None else None,
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=0, pin_memory=True)

    train_loader = make_loader(train_idx, shuffle=True)
    val_loader = make_loader(val_idx, shuffle=False)
    test_loader = make_loader(test_idx, shuffle=False)

    print(f'Data split: train={len(train_idx)}, val={len(val_idx)}, '
          f'test={len(test_idx)}')

    return train_loader, val_loader, test_loader
