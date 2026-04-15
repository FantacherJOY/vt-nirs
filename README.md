# VT-NIRS: Virtual Twin for Non-Invasive Respiratory Support

This repository contains the code for **VT-NIRS**, a causal deep learning framework for estimating individualized treatment effects (ITE) of non-invasive respiratory support (NIRS) versus invasive mechanical ventilation (IMV) in patients with acute respiratory failure. The primary outcome is ventilator-free days at 28 days (VFD-28).

**Paper:** *VT-NIRS: A Virtual Twin Framework for Individualized Treatment Effect Estimation of Non-Invasive Respiratory Support in Acute Respiratory Failure*
Published in Proceedings of Machine Learning Research (PMLR), Machine Learning for Healthcare (MLHC), 2026.

## Framework Overview

VT-NIRS consists of three stages:

1. **Survival-Aware Encoder** — A 4-layer Transformer encoder with an attention gate that decomposes learned representations into mortality (z_surv) and ventilation (z_vfd) components.
2. **Adversarial Pretraining** — A counterfactual generator produces potential outcomes under both treatment arms, trained adversarially against a discriminator (adapted from GANITE, Yoon et al. 2018).
3. **ITE Estimation** — A doubly-robust pseudo-outcome module (Kennedy, 2023) supervises a direct ITE prediction head, with MMD and propensity-based representation balancing (Shalit et al., 2017; Shi et al., 2019).

## Data

This project uses two publicly available ICU databases, both hosted on Google BigQuery via PhysioNet:

- **MIMIC-IV v3.1** ([PhysioNet](https://physionet.org/content/mimiciv/3.1/)) — Training and internal test set
- **eICU-CRD** ([PhysioNet](https://physionet.org/content/eicu-crd/)) — External validation set

Access requires a credentialed PhysioNet account and completion of the CITI training course. Once approved, set up [Google Cloud](https://cloud.google.com/run/docs/setup) with a billing project to query the databases.

## Directory Structure

```
vt_nirs_github/
|-- README.md
|-- VT_NIRS_Run_final.ipynb       # Main notebook (extraction, training, evaluation)
|-- vt_nirs/
    |-- models/
    |   |-- __init__.py
    |   |-- baselines.py           # Custom T-Learner and Causal Forest
    |   |-- discriminator.py       # Treatment discriminator with spectral norm
    |   |-- encoder.py             # Temporal Transformer encoder
    |   |-- encoder_sa.py          # Survival-aware encoder with attention gate
    |   |-- generator.py           # Counterfactual generator (survival + VFD heads)
    |   |-- ite_predictor.py       # Direct ITE prediction head
    |   |-- vt_nirs.py             # Full VT-NIRS model assembly
    |-- training/
    |   |-- __init__.py
    |   |-- train.py               # Two-stage training loop
    |-- utils/
        |-- __init__.py
        |-- domain_adapt.py        # eICU domain adaptation utilities
        |-- extraction.py          # BigQuery data extraction (MIMIC-IV + eICU)
        |-- loader.py              # Data loading, PSM, DataLoader construction
        |-- losses.py              # Adversarial + DR pseudo-outcome + MMD losses
        |-- metrics.py             # Evaluation metrics and visualization
```

## Prerequisites

1. Python 3.8+
2. Google Cloud account with BigQuery billing enabled
3. PhysioNet credentialed access to MIMIC-IV and eICU-CRD

## Dependencies

- torch
- numpy
- pandas
- scikit-learn
- google-cloud-bigquery
- matplotlib

## Setup

1. Clone this repository:
```bash
git clone https://github.com/YOUR_USERNAME/vt_nirs.git
cd vt_nirs
```

2. Set your Google Cloud billing project ID in `vt_nirs/utils/extraction.py`:
```python
BILLING_PROJECT = "YOUR_GCP_PROJECT_ID"
```

3. Authenticate with Google Cloud:
```python
from google.colab import auth
auth.authenticate_user()
```

## Usage

Open `VT_NIRS_Run_final.ipynb` and run cells sequentially. The notebook executes the full pipeline:

1. **Data Extraction** — Queries MIMIC-IV via BigQuery, identifies the ARF cohort, assigns treatment labels, computes VFD-28
2. **Propensity Score Matching** — Logistic regression on 23 baseline covariates with caliper matching
3. **Stage 1 Training** — Adversarial pretraining of encoder, generator, and discriminator
4. **Stage 2 Training** — Doubly-robust ITE refinement with frozen encoder
5. **Evaluation** — Policy value, C-for-benefit, subgroup analysis, baseline comparisons (T-Learner, Causal Forest)
6. **Ablation Study** — Gate removal ablation
7. **External Validation** — eICU-CRD cohort extraction and zero-shot transfer evaluation

## Citation

```bibtex
@inproceedings{vtnirs2026,
  title={VT-NIRS: A Virtual Twin Framework for Individualized Treatment Effect Estimation of Non-Invasive Respiratory Support in Acute Respiratory Failure},
  author={},
  booktitle={Proceedings of Machine Learning for Healthcare (MLHC)},
  year={2026},
  publisher={PMLR}
}
```

## License

This project is intended for research purposes only. The data used in this study is subject to PhysioNet's data use agreements.
