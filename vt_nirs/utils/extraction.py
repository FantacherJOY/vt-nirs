

import os, time, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .loader import MIMIC_ITEMIDS, compute_vfd28 as loader_compute_vfd28

warnings.filterwarnings("ignore", category=UserWarning, module='google.cloud.bigquery')
warnings.filterwarnings("ignore", category=FutureWarning)


BILLING_PROJECT = "YOUR_GCP_PROJECT_ID"
DATA_PROJECT    = "physionet-data"

MIMIC = {
    "icu":     "physionet-data.mimiciv_3_1_icu",
    "hosp":    "physionet-data.mimiciv_3_1_hosp",
    "derived": "physionet-data.mimiciv_3_1_derived",
}
EICU = {
    "main":    "physionet-data.eicu_crd",
    "derived": "physionet-data.eicu_crd_derived",
}

ITEMIDS = {
    "inv_proc": [225792],
    "niv_proc": [225794, 225949, 227578],
    "hfnc":     [226732],
    "bipap":    [227578, 227579],
    "cpap":     [227580],
    "peep":     [220339],
    "fio2":     [223835],
    "tv":       [224685, 224684],
}

VENT_MAP = {
    'InvasiveVent':       'Invasive',
    'Tracheostomy':       'Invasive',
    'NonInvasiveVent':    'NIV',
    'HFNC':               'NIV',
    'SupplementalOxygen': 'Oxygen',
    'None':               'None',
}
VENT_STATUS_NIRS = {"NonInvasiveVent", "HFNC"}
VENT_STATUS_IMV  = {"InvasiveVent", "Tracheostomy"}

FEATURE_COLS = [
    "age_X", "gender_X", "bmi_X",
    "sofa_X", "gcs_X", "sapsii_X",
    "hr_mean_X", "rr_mean_X", "spo2_mean_X", "mbp_mean_X", "tempc_mean_X",
    "pao2_X", "paco2_X", "ph_X", "fio2_X", "lactate_X", "bicarbonate_X",
    "pf_ratio_X", "rox_index_X",
    "copd_X", "chf_X", "immunosuppressed_X", "sepsis_X",
]

CONTINUOUS_COLS = [c for c in FEATURE_COLS if c not in
                   {"gender_X", "copd_X", "chf_X", "immunosuppressed_X", "sepsis_X"}]
BINARY_COLS = ["gender_X", "copd_X", "chf_X", "immunosuppressed_X", "sepsis_X"]

T0_WINDOW_H      = 24
MIN_LOS_DAYS     = 0.5
VFD_HORIZON_DAYS = 28
RANDOM_STATE     = 42


_client = None

def init_client():
    """Initialize BigQuery client (billing to your project)."""
    global _client
    from google.cloud import bigquery
    os.environ["GOOGLE_CLOUD_PROJECT"] = BILLING_PROJECT
    _client = bigquery.Client(project=BILLING_PROJECT)
    return _client


def run_bq(sql, verbose=True):
    """Execute BigQuery SQL and return DataFrame (from m0_config.py)."""
    global _client
    if _client is None:
        init_client()
    t0 = time.time()
    df = _client.query(sql).to_dataframe()
    if verbose:
        print(f"  -> {len(df):,} rows  ({time.time()-t0:.1f}s)")
    return df


def ids_str(id_array):
    """Convert array of IDs to comma-separated string for SQL IN clause."""
    return ",".join(str(int(x)) for x in id_array)


def safe_divide(a, b, fill=np.nan):
    """Element-wise division, filling division-by-zero with fill."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(b != 0, a / b, fill)
    return result


def extract_icu_stays():
    sql = f"""
    SELECT
        ie.subject_id,
        ie.hadm_id,
        ie.stay_id,
        ie.intime   AS icu_intime,
        ie.outtime  AS icu_outtime,
        DATETIME_DIFF(ie.outtime, ie.intime, HOUR) / 24.0 AS los_icu_days,
        pat.anchor_age AS age,
        pat.gender,
        adm.deathtime,
        adm.hospital_expire_flag,
        adm.admission_type
    FROM `{MIMIC['icu']}.icustays` ie
    JOIN `{MIMIC['hosp']}.patients`   pat ON ie.subject_id = pat.subject_id
    JOIN `{MIMIC['hosp']}.admissions` adm ON ie.hadm_id    = adm.hadm_id
    WHERE pat.anchor_age >= 18
      AND DATETIME_DIFF(ie.outtime, ie.intime, HOUR) / 24.0 >= {MIN_LOS_DAYS}
    """
    print("1.1  Extracting ICU stays...")
    df = run_bq(sql)
    print(f"     Eligible adult stays (LOS >= {MIN_LOS_DAYS}d): {df['stay_id'].nunique():,}")
    return df


def identify_arf(df_stays):
    """
    ARF = ICD-10 J96.0x PLUS physiological confirmation.
    # Ref: VT_ITHE_Code/m1_cohort.py lines 57-154 — identify_arf()
    """
    sql_icd_broad = f"""
    SELECT DISTINCT d.hadm_id
    FROM `{MIMIC['hosp']}.diagnoses_icd` d
    WHERE d.icd_version = 10
      AND (d.icd_code LIKE 'J960%'
           OR d.icd_code LIKE 'J80%'
           OR d.icd_code LIKE 'J96%')
    """
    print("1.2  Identifying ARF by ICD-10...")
    df_icd = run_bq(sql_icd_broad)

    sql_physio = f"""
    WITH physio_flags AS (
        SELECT
            ie.stay_id,
            ie.hadm_id,
            CASE WHEN v.spo2_min < 94 THEN 1 ELSE 0 END AS flag_spo2,
            CASE WHEN v.resp_rate_max > 25 THEN 1 ELSE 0 END AS flag_rr,
            CASE WHEN bg_agg.pao2_min < 60 THEN 1 ELSE 0 END AS flag_pao2,
            CASE WHEN bg_agg.paco2_max > 50 AND bg_agg.ph_min < 7.35
                 THEN 1 ELSE 0 END AS flag_hypercapnic
        FROM `{MIMIC['icu']}.icustays` ie
        JOIN `{MIMIC['hosp']}.patients` pat
            ON ie.subject_id = pat.subject_id
        LEFT JOIN `{MIMIC['derived']}.first_day_vitalsign` v
            ON ie.stay_id = v.stay_id
        LEFT JOIN (
            SELECT
                i2.stay_id,
                MIN(b.po2)     AS pao2_min,
                MAX(b.pco2)    AS paco2_max,
                MIN(b.ph)      AS ph_min
            FROM `{MIMIC['derived']}.bg` b
            JOIN `{MIMIC['derived']}.icustay_detail` i2
                ON b.subject_id = i2.subject_id
            WHERE b.charttime BETWEEN i2.icu_intime
                  AND DATETIME_ADD(i2.icu_intime, INTERVAL 24 HOUR)
              AND i2.los_icu >= {MIN_LOS_DAYS}
            GROUP BY i2.stay_id
        ) bg_agg ON ie.stay_id = bg_agg.stay_id
        WHERE pat.anchor_age >= 18
          AND DATETIME_DIFF(ie.outtime, ie.intime, HOUR) / 24.0 >= {MIN_LOS_DAYS}
    )
    SELECT stay_id, hadm_id,
           flag_spo2, flag_rr, flag_pao2, flag_hypercapnic,
           GREATEST(flag_spo2, flag_rr, flag_pao2, flag_hypercapnic) AS any_physio
    FROM physio_flags
    """
    print("     Querying physiological confirmation...")
    df_physio = run_bq(sql_physio)

    arf_hadm = set(df_icd["hadm_id"])
    df_physio_confirmed = df_physio[
        (df_physio["hadm_id"].isin(arf_hadm)) &
        (df_physio["any_physio"] == 1)
    ]
    arf_stay_ids = set(df_physio_confirmed["stay_id"])
    df_arf = df_stays[df_stays["stay_id"].isin(arf_stay_ids)].copy()
    print(f"     ARF stays (ICD + physiology): {len(df_arf):,}")
    return df_arf


def apply_exclusions(df_arf):
    print("1.3  Applying exclusions...")
    df_dnr = run_bq(sql_dnr)
    excl_dnr = set(df_dnr["stay_id"])

    sql_crash = f"""
    SELECT DISTINCT pe.stay_id
    FROM `{MIMIC['icu']}.procedureevents` pe
    JOIN `{MIMIC['icu']}.icustays` ie ON pe.stay_id = ie.stay_id
    WHERE pe.itemid = 225792
      AND pe.stay_id IN ({stay_ids})
      AND DATETIME_DIFF(pe.starttime, ie.intime, MINUTE) <= 60
    """
    df_crash = run_bq(sql_crash)
    excl_crash = set(df_crash["stay_id"])

    sql_chronic = f"""
    SELECT DISTINCT d.hadm_id
    FROM `{MIMIC['hosp']}.diagnoses_icd` d
    WHERE d.icd_version = 10
      AND (d.icd_code LIKE 'Z991%'
           OR d.icd_code LIKE 'J950%'
           OR d.icd_code LIKE 'Z930%')
    """
    df_chronic = run_bq(sql_chronic)
    excl_chronic_hadm = set(df_chronic["hadm_id"])
    excl_chronic = set(df_arf[df_arf["hadm_id"].isin(excl_chronic_hadm)]["stay_id"])

    all_excl = excl_dnr | excl_crash | excl_chronic
    df_clean = df_arf[~df_arf["stay_id"].isin(all_excl)].copy()

    print(f"     DNI/DNR excluded:        {len(excl_dnr):,}")
    print(f"     Crash intubation (<1h):  {len(excl_crash):,}")
    print(f"     Chronic vent/trach:      {len(excl_chronic):,}")
    print(f"     Remaining after excl:    {len(df_clean):,}  "
          f"(removed {n0 - len(df_clean):,})")
    return df_clean


def assign_treatment(df_cohort):
    stay_ids = ids_str(df_cohort["stay_id"])
    sql_vent = f"""
    SELECT v.stay_id, v.starttime, v.endtime, v.ventilation_status
    FROM `{MIMIC['derived']}.ventilation` v
    WHERE v.stay_id IN ({stay_ids})
    ORDER BY v.stay_id, v.starttime
    """

    niv_ids = ",".join(map(str, ITEMIDS["niv_proc"]))
    inv_ids = ",".join(map(str, ITEMIDS["inv_proc"]))
    sql_proc = f"""
    SELECT pe.stay_id, pe.starttime, pe.endtime, pe.itemid,
        CASE WHEN pe.itemid = 225792 THEN 'Invasive' ELSE 'NIV' END AS vent_type
    FROM `{MIMIC['icu']}.procedureevents` pe
    WHERE pe.stay_id IN ({stay_ids})
      AND pe.itemid IN ({niv_ids},{inv_ids})
    ORDER BY pe.stay_id, pe.starttime
    """

    print("1.4  Assigning treatment (NIRS vs IMV)...")
    df_vent = run_bq(sql_vent)
    df_proc = run_bq(sql_proc)

    priority_map = {'Invasive': 1, 'NIV': 2}

    df_merged = pd.merge(
        df_vent, df_proc,
        on='stay_id', suffixes=('_status', '_proc')
    )

    df_overlaps = df_merged.query(
        '(starttime_status <= endtime_proc and endtime_status >= starttime_proc) and '
        'ventilation_status in ["None", "SupplementalOxygen"]'
    ).copy()

    if len(df_overlaps) > 0:
        df_overlaps['priority'] = df_overlaps['vent_type'].map(priority_map)
        corrections = df_overlaps.groupby(
            ['stay_id', 'starttime_status', 'endtime_status', 'ventilation_status']
        )['priority'].min()
        corrections_map = {1: 'InvasiveVent', 2: 'NonInvasiveVent'}
        corrections = corrections.map(corrections_map).reset_index()
        corrections = corrections.rename(columns={
            'starttime_status': 'starttime',
            'endtime_status':   'endtime',
            'priority':         'corrected_status',
        })

        df_vent_corrected = pd.merge(
            df_vent, corrections,
            on=['stay_id', 'starttime', 'endtime', 'ventilation_status'],
            how='left'
        )
        df_vent_corrected['final_status'] = np.where(
            pd.notna(df_vent_corrected['corrected_status']),
            df_vent_corrected['corrected_status'],
            df_vent_corrected['ventilation_status']
        )
    else:
        df_vent_corrected = df_vent.copy()
        df_vent_corrected['final_status'] = df_vent_corrected['ventilation_status']

    df_vent_corrected['vent_type'] = df_vent_corrected['final_status'].map(VENT_MAP)
    df_vent_corrected = df_vent_corrected.sort_values(['stay_id', 'starttime'])

    n_before = df_vent[df_vent['ventilation_status'].isin(
        ['NonInvasiveVent', 'HFNC'])].shape[0]
    n_after = df_vent_corrected[df_vent_corrected['vent_type'] == 'NIV'].shape[0]
    print(f"     Ventilation correction: NIV episodes {n_before:,} → {n_after:,} "
          f"(+{n_after - n_before:,} from procedure overlap)")

    df_vent_events = df_vent_corrected[
        df_vent_corrected['vent_type'].isin(['Invasive', 'NIV'])
    ].copy()

    df_first_events = (
        df_vent_events
        .groupby(['stay_id', 'vent_type'])['starttime']
        .min()
        .unstack()
    )

    all_stay_ids = df_cohort['stay_id'].unique()
    df_first_events = df_first_events.reindex(all_stay_ids)

    def _categorize(row):
        inv_exists = pd.notna(row.get('Invasive'))
        niv_exists = pd.notna(row.get('NIV'))
        if inv_exists and not niv_exists:
            return 'IMV_only', 0
        if niv_exists and not inv_exists:
            return 'NIRS_only', 1
        if not inv_exists and not niv_exists:
            return 'None', np.nan
        if row['Invasive'] < row['NIV']:
            return 'IMV_then_NIRS', 0
        if row['NIV'] < row['Invasive']:
            return 'NIRS_then_IMV', 1
        return 'ambiguous', np.nan

    results = df_first_events.apply(_categorize, axis=1)
    df_tx = pd.DataFrame({
        'stay_id': df_first_events.index,
        'category': [r[0] for r in results],
        'Treatment_W': [r[1] for r in results],
    })

    df_tx = df_tx[df_tx['Treatment_W'].notna()].copy()
    df_tx['Treatment_W'] = df_tx['Treatment_W'].astype(int)

    for cat in ['NIRS_only', 'NIRS_then_IMV', 'IMV_only', 'IMV_then_NIRS', 'ambiguous']:
        n = (df_tx['category'] == cat).sum()
        if n > 0:
            print(f"     {cat}: {n:,}")

    df_treat = df_tx[['stay_id', 'Treatment_W', 'category']]
    df_final = df_cohort.merge(df_treat, on='stay_id', how='inner')

    print(f"\n     Treatment assignment results:")
    print(f"     Total assigned: {len(df_final):,}")
    print(f"     W=1 (NIRS first): {(df_final.Treatment_W==1).sum():,}")
    print(f"     W=0 (IMV first):  {(df_final.Treatment_W==0).sum():,}")
    return df_final


def build_cohort():
    print("=" * 65)
    print("  VT-NIRS COHORT EXTRACTION  (MIMIC-IV)")
    print("=" * 65)
    df_stays = extract_icu_stays()
    df_arf   = identify_arf(df_stays)
    df_clean = apply_exclusions(df_arf)
    df_final = assign_treatment(df_clean)
    print("=" * 65)
    print(f"  FINAL COHORT: {len(df_final):,} patients")
    print(f"    NIRS arm: {(df_final.Treatment_W==1).sum():,}")
    print(f"    IMV  arm: {(df_final.Treatment_W==0).sum():,}")
    print("=" * 65)
    return df_final


def compute_vfd28(df_cohort):
    stay_ids = ids_str(df_cohort["stay_id"])

    sql_vent_full = f"""
    WITH vent_with_icu AS (
        SELECT
            v.stay_id, v.starttime, v.endtime, v.ventilation_status,
            ie.intime AS icu_intime,
            GREATEST(v.starttime, ie.intime) AS eff_start,
            LEAST(v.endtime, DATETIME_ADD(ie.intime, INTERVAL 28 DAY)) AS eff_end
        FROM `{MIMIC['derived']}.ventilation` v
        JOIN `{MIMIC['icu']}.icustays` ie ON v.stay_id = ie.stay_id
        WHERE v.stay_id IN ({stay_ids})
          AND v.ventilation_status IN ('InvasiveVent', 'Tracheostomy')
          AND v.starttime < DATETIME_ADD(ie.intime, INTERVAL 28 DAY)
          AND v.endtime > ie.intime
    )
    SELECT stay_id,
           SUM(DATETIME_DIFF(eff_end, eff_start, HOUR)) AS total_imv_hours
    FROM vent_with_icu
    WHERE eff_end > eff_start
    GROUP BY stay_id
    """
    print("2.1  Computing VFD-28...")
    print("     Querying IMV duration within 28-day window...")
    df_imv = run_bq(sql_vent_full)

    sql_death = f"""
    SELECT
        ie.stay_id,
        adm.deathtime,
        adm.hospital_expire_flag,
        CASE
            WHEN adm.deathtime IS NOT NULL
                 AND DATETIME_DIFF(adm.deathtime, ie.intime, DAY) <= 28
            THEN 1
            WHEN adm.hospital_expire_flag = 1
            THEN 1
            ELSE 0
        END AS died_28d,
        CASE
            WHEN adm.deathtime IS NOT NULL
            THEN DATETIME_DIFF(adm.deathtime, ie.intime, DAY)
            ELSE NULL
        END AS days_to_death
    FROM `{MIMIC['icu']}.icustays` ie
    JOIN `{MIMIC['hosp']}.admissions` adm ON ie.hadm_id = adm.hadm_id
    WHERE ie.stay_id IN ({stay_ids})
    """
    print("     Querying 28-day mortality...")
    df_death = run_bq(sql_death)

    df_out = df_cohort[["stay_id", "Treatment_W"]].copy()
    df_out = df_out.merge(df_imv, on="stay_id", how="left")
    df_out = df_out.merge(
        df_death[["stay_id", "died_28d", "days_to_death"]],
        on="stay_id", how="left"
    )

    df_out["total_imv_hours"] = df_out["total_imv_hours"].fillna(0)
    df_out["died_28d"] = df_out["died_28d"].fillna(0).astype(int)
    df_out["total_imv_days"] = df_out["total_imv_hours"] / 24.0
    df_out["vfd28"] = np.where(
        df_out["died_28d"] == 1,
        0.0,
        np.clip(28.0 - df_out["total_imv_days"], 0, 28)
    )
    df_out["delta"] = 1 - df_out["died_28d"]

    print(f"\n     VFD-28 summary:")
    print(f"       Mean VFD-28:      {df_out['vfd28'].mean():.1f}")
    print(f"       Median VFD-28:    {df_out['vfd28'].median():.1f}")
    print(f"       28-day mortality: {df_out['died_28d'].mean()*100:.1f}%")
    for w in [0, 1]:
        arm = df_out[df_out["Treatment_W"] == w]
        lbl = "NIRS" if w == 1 else "IMV"
        print(f"       {lbl} arm: mean VFD-28 = {arm['vfd28'].mean():.1f}, "
              f"mort = {arm['died_28d'].mean()*100:.1f}%")
    return df_out


def extract_baseline_covariates(df_cohort):
    stay_ids = ids_str(df_cohort["stay_id"])

    sql_base = f"""
    SELECT
        ie.stay_id,
        pat.anchor_age AS age_X,
        CASE WHEN pat.gender = 'M' THEN 1 ELSE 0 END AS gender_X,
        fdw.weight_admit,
        fdh.height,
        CASE
            WHEN fdh.height > 0 AND fdw.weight_admit > 0
            THEN fdw.weight_admit / POWER(fdh.height / 100.0, 2)
            ELSE NULL
        END AS bmi_X,
        sf.sofa AS sofa_X,
        gcs.gcs_min AS gcs_X,
        v.heart_rate_mean AS hr_mean_X,
        v.resp_rate_mean AS rr_mean_X,
        v.spo2_mean AS spo2_mean_X,
        v.mbp_mean AS mbp_mean_X,
        v.temperature_mean AS tempc_mean_X
    FROM `{MIMIC['icu']}.icustays` ie
    JOIN `{MIMIC['hosp']}.patients` pat ON ie.subject_id = pat.subject_id
    LEFT JOIN `{MIMIC['derived']}.first_day_height` fdh ON ie.stay_id = fdh.stay_id
    LEFT JOIN `{MIMIC['derived']}.first_day_weight` fdw ON ie.stay_id = fdw.stay_id
    LEFT JOIN `{MIMIC['derived']}.first_day_sofa` sf ON ie.stay_id = sf.stay_id
    LEFT JOIN `{MIMIC['derived']}.first_day_gcs` gcs ON ie.stay_id = gcs.stay_id
    LEFT JOIN (
        SELECT stay_id, MAX(sapsii) AS sapsii
        FROM `{MIMIC['derived']}.sapsii`
        GROUP BY stay_id
    ) sp ON ie.stay_id = sp.stay_id
    LEFT JOIN `{MIMIC['derived']}.first_day_vitalsign` v ON ie.stay_id = v.stay_id
    WHERE ie.stay_id IN ({stay_ids})
    """
    sql_base = sql_base.replace(
        "v.temperature_mean AS tempc_mean_X",
        "v.temperature_mean AS tempc_mean_X,\n        sp.sapsii AS sapsii_X"
    )

    print("3.1  Extracting demographics + severity + vitals...")
    df_base = run_bq(sql_base)
    df_base.drop(columns=["weight_admit", "height"], inplace=True, errors="ignore")

    sql_abg = f"""
    SELECT
        i.stay_id,
        AVG(b.po2)  AS pao2_X,
        AVG(b.pco2) AS paco2_X,
        AVG(b.ph)   AS ph_X,
        AVG(CASE WHEN b.fio2 <= 1.0 THEN b.fio2 * 100.0 ELSE b.fio2 END) AS fio2_X,
        AVG(b.lactate)     AS lactate_X,
        AVG(b.bicarbonate) AS bicarbonate_X
    FROM `{MIMIC['derived']}.bg` b
    JOIN `{MIMIC['derived']}.icustay_detail` i ON b.subject_id = i.subject_id
    WHERE i.stay_id IN ({stay_ids})
      AND b.charttime BETWEEN i.icu_intime
          AND DATETIME_ADD(i.icu_intime, INTERVAL 24 HOUR)
    GROUP BY i.stay_id
    """
    print("3.2  Extracting ABG (first 24h)...")
    df_abg = run_bq(sql_abg)

    hadm_ids = ids_str(df_cohort["hadm_id"])
    sql_charlson = f"""
    SELECT
        ie.stay_id,
        COALESCE(c.chronic_pulmonary_disease, 0) AS copd_X,
        COALESCE(c.congestive_heart_failure, 0)  AS chf_X,
        CASE
            WHEN COALESCE(c.aids, 0) = 1
              OR COALESCE(c.metastatic_solid_tumor, 0) = 1
            THEN 1 ELSE 0
        END AS immunosuppressed_X
    FROM `{MIMIC['icu']}.icustays` ie
    LEFT JOIN `{MIMIC['derived']}.charlson` c ON ie.hadm_id = c.hadm_id
    WHERE ie.stay_id IN ({stay_ids})
    """
    sql_sepsis = f"""
    SELECT DISTINCT stay_id, 1 AS sepsis_X
    FROM `{MIMIC['derived']}.sepsis3`
    WHERE stay_id IN ({stay_ids})
    """
    print("3.3  Extracting comorbidities...")
    df_charlson = run_bq(sql_charlson)
    print("     Querying Sepsis-3...")
    df_sepsis = run_bq(sql_sepsis)

    df = df_base.merge(df_abg, on="stay_id", how="left")
    df = df.merge(df_charlson, on="stay_id", how="left")
    df = df.merge(df_sepsis, on="stay_id", how="left")
    df["sepsis_X"] = df["sepsis_X"].fillna(0).astype(int)

    df["pf_ratio_X"] = safe_divide(
        df["pao2_X"].values, df["fio2_X"].values / 100.0)
    df["pf_ratio_X"] = np.clip(df["pf_ratio_X"], 0, 700)

    df["rox_index_X"] = safe_divide(
        safe_divide(df["spo2_mean_X"].values, df["fio2_X"].values / 100.0),
        df["rr_mean_X"].values)
    df["rox_index_X"] = np.clip(df["rox_index_X"], 0, 30)

    df["bmi_X"] = np.clip(df["bmi_X"], 14.0, 70.0)

    for col in BINARY_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    print("\n     Imputation (median for continuous, 0 for binary):")
    for col in CONTINUOUS_COLS:
        if col in df.columns:
            n_miss = df[col].isna().sum()
            if n_miss > 0:
                med = df[col].median()
                df[col].fillna(med, inplace=True)
                print(f"       {col}: {n_miss:,} missing -> median={med:.2f}")

    present = [c for c in FEATURE_COLS if c in df.columns]
    print(f"\n     All {len(present)} covariates present")
    return df


def standardize_features(df, cols=None, reference_df=None):
    if cols is None:
        cols = CONTINUOUS_COLS
    ref = (reference_df if reference_df is not None else df).copy()
    stats = {}
    df_out = df.copy()
    for col in cols:
        if col in df_out.columns:
            df_out[col] = pd.to_numeric(df_out[col], errors='coerce')
        if col in ref.columns:
            ref[col] = pd.to_numeric(ref[col], errors='coerce')
    for col in cols:
        if col in df_out.columns:
            mu  = float(ref[col].mean())
            sig = float(ref[col].std())
            if sig < 1e-10:
                sig = 1.0
            df_out[col] = (df_out[col] - mu) / sig
            stats[col] = (mu, sig)
    return df_out, stats


def extract_temporal_chartevents(df_cohort, chunk_size=5000):
    stay_ids = df_cohort['stay_id'].tolist()
    chunks = [stay_ids[i:i + chunk_size]
              for i in range(0, len(stay_ids), chunk_size)]

    all_vital_ids = []
    for ids in MIMIC_ITEMIDS.values():
        all_vital_ids.extend(ids)
    vital_ids_str = ','.join(map(str, all_vital_ids))

    results = []
    for i, chunk in enumerate(chunks):
        chunk_str = ','.join(map(str, chunk))
        query = f"""
        SELECT
            ce.stay_id,
            ce.charttime,
            AVG(IF(itemid = 220045, valuenum, NULL))                         AS heart_rate,
            AVG(IF(itemid IN (220210, 224690), valuenum, NULL))              AS resp_rate,
            AVG(IF(itemid = 220277 AND valuenum <= 100, valuenum, NULL))     AS spo2,
            AVG(IF(itemid IN (220052, 220181, 225312), valuenum, NULL))      AS mbp,
            AVG(CASE
                    WHEN itemid = 223761 AND valuenum BETWEEN 70 AND 120
                         THEN (valuenum - 32) * 5.0/9.0
                    WHEN itemid = 223762 AND valuenum BETWEEN 10 AND 50
                         THEN valuenum
                    ELSE NULL END)                                           AS temperature,
            AVG(CASE
                    WHEN itemid = 223835 AND valuenum <= 1.0 THEN valuenum * 100
                    WHEN itemid = 223835 AND valuenum >  1.0 THEN valuenum
                    ELSE NULL END)                                           AS fio2,
            AVG(IF(itemid IN (220339, 224700), valuenum, NULL))              AS peep,
            AVG(IF(itemid = 220224, valuenum, NULL))                         AS pao2,
            AVG(IF(itemid = 220235, valuenum, NULL))                         AS paco2,
            AVG(IF(itemid IN (220274, 220734), valuenum, NULL))              AS ph,
            AVG(IF(itemid = 225668, valuenum, NULL))                         AS lactate,
            AVG(IF(itemid = 220615, valuenum, NULL))                         AS creatinine,
            AVG(IF(itemid = 225690, valuenum, NULL))                         AS bilirubin,
            AVG(IF(itemid = 227457, valuenum, NULL))                         AS platelets,
            AVG(IF(itemid = 220546, valuenum, NULL))                         AS wbc,
            AVG(IF(itemid IN (220739, 223900, 223901), valuenum, NULL))      AS gcs_component
        FROM `{MIMIC['icu']}.chartevents` ce
        WHERE ce.stay_id IN ({chunk_str})
          AND ce.itemid IN ({vital_ids_str})
          AND ce.valuenum IS NOT NULL
        GROUP BY ce.stay_id, ce.charttime
        ORDER BY ce.stay_id, ce.charttime
        """
        chunk_df = run_bq(query, verbose=False)
        results.append(chunk_df)
        print(f'  Chunk {i + 1}/{len(chunks)}: {len(chunk_df):,} rows')

    df_vitals = pd.concat(results, ignore_index=True)
    print(f'\nTotal vital sign rows: {len(df_vitals):,}')
    print(f'Unique stays: {df_vitals.stay_id.nunique():,}')
    return df_vitals


TEMPORAL_FEATURE_COLS = [
    'age', 'gender_num', 'bmi',
    'heart_rate', 'resp_rate', 'spo2', 'mbp', 'temperature', 'fio2', 'peep',
    'pao2', 'paco2', 'ph', 'pf_ratio',
    'lactate', 'creatinine', 'bilirubin', 'platelets', 'wbc',
    'sofa_score', 'gcs_total',
    'hours_since_admit', 'vasopressor_flag',
]
N_TEMPORAL_COVARIATES = len(TEMPORAL_FEATURE_COLS)


def build_temporal_sequences(df_vitals, df_cohort, seq_len=48):
    """
    Build padded time series tensors at 0.5h resolution for Transformer.

    # Ref: DT_ITE_Final.ipynb Cell 10 — build_tensors() + pad_sequences
    # Ref: graphspa training/loader.py — variable-length handling
    """
    def _pad_sequences(seqs, maxlen, dtype='float32', padding='pre', value=0.0):
        n_feat = seqs[0].shape[1] if len(seqs) > 0 else 0
        out = np.full((len(seqs), maxlen, n_feat), value, dtype=dtype)
        for i, s in enumerate(seqs):
            trunc = s[-maxlen:] if len(s) > maxlen else s
            if padding == 'pre':
                out[i, maxlen - len(trunc):, :] = trunc
            else:
                out[i, :len(trunc), :] = trunc
        return out

    df_merged = df_vitals.merge(
        df_cohort[['stay_id', 'age', 'gender', 'icu_intime',
                    'Treatment_W', 'vfd28', 'delta']].drop_duplicates('stay_id'),
        on='stay_id', how='inner'
    )

    if 'sofa_X' in df_cohort.columns:
        sofa_map = df_cohort.set_index('stay_id')['sofa_X'].to_dict()
        df_merged['sofa_score'] = df_merged['stay_id'].map(sofa_map)
    else:
        df_merged['sofa_score'] = np.nan

    df_merged['gender_num'] = (df_merged['gender'] == 'M').astype(float)
    df_merged['hours_since_admit'] = (
        pd.to_datetime(df_merged['charttime']) -
        pd.to_datetime(df_merged['icu_intime'])
    ).dt.total_seconds() / 3600

    df_merged['pf_ratio'] = np.where(
        (df_merged['fio2'] > 0) & df_merged['pao2'].notna(),
        df_merged['pao2'] / (df_merged['fio2'] / 100).clip(lower=0.21),
        np.nan
    )
    df_merged['bmi'] = np.nan
    df_merged['gcs_total'] = df_merged['gcs_component']
    df_merged['vasopressor_flag'] = 0

    sequences, treatments, vfd_list, delta_list, valid_ids = [], [], [], [], []

    for stay_id, group in df_merged.groupby('stay_id'):
        group = group.sort_values('charttime')
        group = group.set_index('charttime')
        group_resampled = group[TEMPORAL_FEATURE_COLS].resample('30min').mean()
        group_resampled = group_resampled.ffill().bfill()
        vals = group_resampled.values
        if len(vals) < 2:
            continue
        sequences.append(vals)
        treatments.append(group['Treatment_W'].iloc[0])
        vfd_list.append(group['vfd28'].iloc[0])
        delta_list.append(group['delta'].iloc[0])
        valid_ids.append(stay_id)

    X = _pad_sequences(sequences, maxlen=seq_len, dtype='float32',
                       padding='pre', value=0.0)
    W = np.array(treatments, dtype=np.float32)
    VFD = np.array(vfd_list, dtype=np.float32)
    D = np.array(delta_list, dtype=np.float32)

    print(f'Time series tensors built:')
    print(f'  X shape: {X.shape}  (patients, timesteps, covariates)')
    print(f'  NIRS (W=1): {(W == 1).sum():,}  |  IMV (W=0): {(W == 0).sum():,}')
    return X, W, VFD, D, valid_ids


def propensity_score_match(X_all, W_all, caliper_scale=0.2, random_state=42):
    X_baseline = X_all[:, -1, :]
    X_baseline_clean = np.nan_to_num(X_baseline, nan=0.0)

    scaler = StandardScaler()
    X_baseline_scaled = scaler.fit_transform(X_baseline_clean)

    ps_model = LogisticRegression(max_iter=1000, random_state=random_state)
    ps_model.fit(X_baseline_scaled, W_all)

    ps = ps_model.predict_proba(X_baseline_scaled)[:, 1]
    logit_ps = np.log(ps / (1 - ps + 1e-8))
    caliper = caliper_scale * logit_ps.std()

    nirs_idx = np.where(W_all == 1)[0]
    imv_idx = np.where(W_all == 0)[0]

    nn_matcher = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
    nn_matcher.fit(logit_ps[imv_idx].reshape(-1, 1))
    distances, matched_idx = nn_matcher.kneighbors(
        logit_ps[nirs_idx].reshape(-1, 1))

    within_caliper = distances.flatten() <= caliper
    matched_nirs = nirs_idx[within_caliper]
    matched_imv = imv_idx[matched_idx.flatten()[within_caliper]]

    psm_idx = np.concatenate([matched_nirs, matched_imv])
    np.random.RandomState(random_state).shuffle(psm_idx)

    print(f'PSM cohort: {len(psm_idx):,} patients '
          f'({len(matched_nirs):,} matched pairs)')
    print(f'  Caliper: {caliper:.4f}')
    return psm_idx, ps_model, ps


def normalize_and_mask(X_psm, n_covariates):
    X_flat = X_psm.reshape(-1, n_covariates)
    ts_scaler = StandardScaler()
    X_flat_scaled = ts_scaler.fit_transform(
        np.nan_to_num(X_flat, nan=0.0))
    X_scaled = X_flat_scaled.reshape(X_psm.shape)
    pad_masks = (X_scaled.sum(axis=-1) == 0)
    return X_scaled, pad_masks, ts_scaler


def extract_eicu_cohort():
    sql = f"""
    SELECT
        p.patientunitstayid   AS stay_id,
        p.uniquepid            AS subject_id,
        p.patienthealthsystemstayid AS hadm_id,
        p.hospitaladmitoffset,
        p.hospitaldischargeoffset,
        p.unitdischargeoffset,
        p.age,
        p.gender,
        p.unitdischargestatus,
        p.hospitaldischargestatus,
        p.unittype,
        p.unitdischargeoffset / (60.0 * 24.0) AS los_icu_days,
        0 AS icu_intime_offset
    FROM `{EICU['main']}.patient` p
    WHERE p.age != ''
      AND SAFE_CAST(REGEXP_REPLACE(p.age, '> ', '') AS INT64) >= 18
      AND p.unitdischargeoffset / (60.0 * 24.0) >= {MIN_LOS_DAYS}
      AND p.unitvisitNumber = 1
    """
    print("6.1  Extracting eICU cohort...")
    df = run_bq(sql)
    df["age_clean"] = df["age"].str.replace("> ", "", regex=False)
    df["age_clean"] = pd.to_numeric(df["age_clean"], errors="coerce")
    df = df[df["age_clean"] >= 18].copy()
    print(f"     eICU eligible stays: {len(df):,}")
    return df


def assign_eicu_treatment(df_eicu):
    _elig = f"""
        SELECT p.patientunitstayid
        FROM `{EICU['main']}.patient` p
        WHERE p.age != ''
          AND SAFE_CAST(REGEXP_REPLACE(p.age, '> ', '') AS INT64) >= 18
          AND p.unitdischargeoffset / (60.0 * 24.0) >= {MIN_LOS_DAYS}
          AND p.unitvisitNumber = 1
    """

    sql_vent_all = f"""
    WITH eligible AS ({_elig})
    SELECT
        ve.patientunitstayid AS stay_id,
        ve.event,
        ve.hrs
    FROM `{EICU['derived']}.ventilation_events` ve
    JOIN eligible e ON ve.patientunitstayid = e.patientunitstayid
    WHERE ve.event IN (
        'mechvent start', 'Trach', 'niv start',
        'mechvent end', 'niv end'
    )
    """

    print("6.2  Assigning eICU treatment (ventilation_events with start/end pairing)...")
    df_vent = run_bq(sql_vent_all)

    start_map = {
        'mechvent start': 'Invasive',
        'Trach': 'Invasive',
        'niv start': 'NIV',
    }
    end_map = {
        'mechvent end': 'Invasive',
        'niv end': 'NIV',
    }

    df_s = df_vent[df_vent['event'].isin(start_map.keys())].copy()
    df_s['vent_type'] = df_s['event'].map(start_map)
    df_s['action'] = 'start'

    df_e = df_vent[df_vent['event'].isin(end_map.keys())].copy()
    df_e['vent_type'] = df_e['event'].map(end_map)
    df_e['action'] = 'end'

    df_all_events = pd.concat([df_s, df_e]).sort_values(
        ['stay_id', 'vent_type', 'hrs'])

    def _get_valid_start(group):
        """Return earliest valid start time for episodes extending into ICU (hrs>0)."""
        valid_starts = []
        current_start = None
        for row in group.itertuples():
            if row.action == 'start':
                if current_start is None:
                    current_start = row.hrs
            elif row.action == 'end':
                if current_start is not None:
                    if row.hrs > 0:
                        valid_starts.append(current_start)
                    current_start = None
        if current_start is not None:
            valid_starts.append(current_start)
        return min(valid_starts) if valid_starts else np.nan

    df_filtered = (df_all_events
                   .groupby(['stay_id', 'vent_type'])
                   .apply(_get_valid_start)
                   .rename('first_hrs'))
    df_first_events = df_filtered.unstack(level='vent_type')

    def _categorize(row):
        first_inv = row.get('Invasive', np.nan)
        first_niv = row.get('NIV', np.nan)
        inv_exists = pd.notna(first_inv)
        niv_exists = pd.notna(first_niv)

        if inv_exists and not niv_exists:
            return 'IMV_only', 0.0
        if not inv_exists and niv_exists:
            return 'NIRS_only', 1.0
        if not inv_exists and not niv_exists:
            return 'exclude', np.nan
        if first_inv < first_niv:
            return 'IMV_then_NIRS', 0.0
        if first_niv < first_inv:
            return 'NIRS_then_IMV', 1.0
        return 'ambiguous', np.nan

    results = df_first_events.apply(_categorize, axis=1)
    df_tx = pd.DataFrame({
        'stay_id': df_first_events.index,
        'category': [r[0] for r in results],
        'Treatment_W': [r[1] for r in results],
    })
    df_tx = df_tx[df_tx['Treatment_W'].notna()].copy()
    df_tx['Treatment_W'] = df_tx['Treatment_W'].astype(int)

    for cat in ['NIRS_only', 'NIRS_then_IMV', 'IMV_only', 'IMV_then_NIRS', 'ambiguous']:
        n = (df_tx['category'] == cat).sum()
        if n > 0:
            print(f"     {cat}: {n:,}")

    df_treat = df_tx[['stay_id', 'Treatment_W', 'category']]
    df_out = df_eicu.merge(df_treat, on='stay_id', how='inner')

    print(f"     eICU assigned: {len(df_out):,}")
    print(f"     W=1 (NIRS): {(df_out.Treatment_W==1).sum():,}")
    print(f"     W=0 (IMV):  {(df_out.Treatment_W==0).sum():,}")
    return df_out


def extract_eicu_covariates(df_eicu):
    _elig = f"""
        SELECT p.patientunitstayid
        FROM `{EICU['main']}.patient` p
        WHERE p.age != ''
          AND SAFE_CAST(REGEXP_REPLACE(p.age, '> ', '') AS INT64) >= 18
          AND p.unitdischargeoffset / (60.0 * 24.0) >= {MIN_LOS_DAYS}
          AND p.unitvisitNumber = 1
    """

    sql_demo = f"""
    WITH eligible AS ({_elig})
    SELECT
        p.patientunitstayid AS stay_id,
        SAFE_CAST(REGEXP_REPLACE(p.age, '> ', '') AS INT64) AS age_X,
        CASE WHEN p.gender = 'Male' THEN 1 ELSE 0 END AS gender_X,
        CASE WHEN p.admissionheight > 0 AND p.admissionweight > 0
             THEN p.admissionweight / POW(p.admissionheight / 100.0, 2)
             ELSE NULL END AS bmi_X,
        COALESCE(apr.acutephysiologyscore, 0) AS apache_aps,
        (COALESCE(apv.eyes, 0) + COALESCE(apv.motor, 0)
         + COALESCE(apv.verbal, 0)) AS gcs_X
    FROM `{EICU['main']}.patient` p
    JOIN eligible e ON p.patientunitstayid = e.patientunitstayid
    LEFT JOIN (
        SELECT patientunitstayid, MAX(acutephysiologyscore) AS acutephysiologyscore
        FROM `{EICU['main']}.apachepatientresult`
        GROUP BY patientunitstayid
    ) apr ON p.patientunitstayid = apr.patientunitstayid
    LEFT JOIN `{EICU['main']}.apacheapsvar` apv
        ON p.patientunitstayid = apv.patientunitstayid
    """

    sql_vitals = f"""
    WITH eligible AS ({_elig})
    SELECT
        nc.patientunitstayid AS stay_id,
        AVG(CASE WHEN nc.nursingchartcelltypevalname = 'Heart Rate'
                 THEN SAFE_CAST(nc.nursingchartvalue AS FLOAT64) END) AS hr_mean_X,
        AVG(CASE WHEN nc.nursingchartcelltypevalname = 'Respiratory Rate'
                 THEN SAFE_CAST(nc.nursingchartvalue AS FLOAT64) END) AS rr_mean_X,
        AVG(CASE WHEN nc.nursingchartcelltypevalname = 'O2 Saturation'
                 THEN SAFE_CAST(nc.nursingchartvalue AS FLOAT64) END) AS spo2_mean_X,
        AVG(CASE WHEN nc.nursingchartcelltypevalname = 'Non-Invasive BP Mean'
                 THEN SAFE_CAST(nc.nursingchartvalue AS FLOAT64) END) AS mbp_mean_X,
        AVG(CASE WHEN nc.nursingchartcelltypevalname = 'Temperature (C)'
                 THEN SAFE_CAST(nc.nursingchartvalue AS FLOAT64) END) AS tempc_mean_X
    FROM `{EICU['main']}.nursecharting` nc
    JOIN eligible e ON nc.patientunitstayid = e.patientunitstayid
    WHERE nc.nursingchartoffset >= 0
      AND nc.nursingchartoffset <= {T0_WINDOW_H * 60}
    GROUP BY nc.patientunitstayid
    """

    sql_labs = f"""
    WITH eligible AS ({_elig})
    SELECT
        l.patientunitstayid AS stay_id,
        AVG(CASE WHEN LOWER(l.labname) = 'pao2' THEN l.labresult END) AS pao2_X,
        AVG(CASE WHEN LOWER(l.labname) = 'paco2' THEN l.labresult END) AS paco2_X,
        AVG(CASE WHEN LOWER(l.labname) = 'ph' THEN l.labresult END) AS ph_X,
        AVG(CASE WHEN LOWER(l.labname) IN ('fio2', 'fio2 (%)')
                 THEN l.labresult END) AS fio2_X,
        AVG(CASE WHEN LOWER(l.labname) = 'lactate' THEN l.labresult END) AS lactate_X,
        AVG(CASE WHEN LOWER(l.labname) IN ('bicarbonate', 'hco3')
                 THEN l.labresult END) AS bicarbonate_X
    FROM `{EICU['main']}.lab` l
    JOIN eligible e ON l.patientunitstayid = e.patientunitstayid
    WHERE l.labresultoffset >= 0
      AND l.labresultoffset <= {T0_WINDOW_H * 60}
    GROUP BY l.patientunitstayid
    """

    sql_comor = f"""
    WITH eligible AS ({_elig})
    SELECT
        p.patientunitstayid AS stay_id,
        MAX(CASE WHEN LOWER(ph.pasthistorypath) LIKE '%copd%'
                   OR LOWER(ph.pasthistorypath) LIKE '%chronic obstructive%'
            THEN 1 ELSE 0 END) AS copd_X,
        MAX(CASE WHEN LOWER(ph.pasthistorypath) LIKE '%chf%'
                   OR LOWER(ph.pasthistorypath) LIKE '%heart failure%'
                   OR LOWER(ph.pasthistorypath) LIKE '%congestive%'
            THEN 1 ELSE 0 END) AS chf_X,
        MAX(CASE WHEN LOWER(ph.pasthistorypath) LIKE '%immunosuppre%'
                   OR LOWER(ph.pasthistorypath) LIKE '%aids%'
                   OR LOWER(ph.pasthistorypath) LIKE '%transplant%'
                   OR LOWER(ph.pasthistorypath) LIKE '%chemotherapy%'
            THEN 1 ELSE 0 END) AS immunosuppressed_X
    FROM `{EICU['main']}.patient` p
    JOIN eligible e ON p.patientunitstayid = e.patientunitstayid
    LEFT JOIN `{EICU['main']}.pasthistory` ph
        ON p.patientunitstayid = ph.patientunitstayid
    GROUP BY p.patientunitstayid
    """

    sql_sepsis = f"""
    WITH eligible AS ({_elig})
    SELECT DISTINCT d.patientunitstayid AS stay_id, 1 AS sepsis_X
    FROM `{EICU['main']}.diagnosis` d
    JOIN eligible e ON d.patientunitstayid = e.patientunitstayid
    WHERE LOWER(d.diagnosisstring) LIKE '%sepsis%'
       OR LOWER(d.diagnosisstring) LIKE '%septic%'
    """

    print("6.3  Extracting eICU covariates...")
    df_demo   = run_bq(sql_demo)
    df_vitals = run_bq(sql_vitals)
    df_labs   = run_bq(sql_labs)
    df_comor  = run_bq(sql_comor)
    df_sepsis = run_bq(sql_sepsis)

    df = df_demo
    for dfi in [df_vitals, df_labs, df_comor, df_sepsis]:
        df = df.merge(dfi, on="stay_id", how="left")

    if "apache_aps" in df.columns:
        df["sofa_X"] = np.clip(df["apache_aps"] / 10.0, 0, 24)
    else:
        df["sofa_X"] = np.nan
    df["sapsii_X"] = df.get("apache_aps", np.nan)

    if "fio2_X" in df.columns:
        mask_frac = df["fio2_X"] <= 1.0
        df.loc[mask_frac, "fio2_X"] = df.loc[mask_frac, "fio2_X"] * 100.0

    df["pf_ratio_X"] = safe_divide(
        df["pao2_X"].values, df["fio2_X"].values / 100.0)
    df["pf_ratio_X"] = np.clip(df["pf_ratio_X"], 0, 700)

    df["rox_index_X"] = safe_divide(
        safe_divide(df["spo2_mean_X"].values, df["fio2_X"].values / 100.0),
        df["rr_mean_X"].values)
    df["rox_index_X"] = np.clip(df["rox_index_X"], 0, 30)

    for col in BINARY_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    df.drop(columns=["apache_aps"], inplace=True, errors="ignore")

    print(f"     eICU covariates: {len(df):,} stays x "
          f"{len([c for c in FEATURE_COLS if c in df.columns])} features")
    return df


def compute_eicu_vfd28(df_eicu):
    _elig = f"""
        SELECT p.patientunitstayid
        FROM `{EICU['main']}.patient` p
        WHERE p.age != ''
          AND SAFE_CAST(REGEXP_REPLACE(p.age, '> ', '') AS INT64) >= 18
          AND p.unitdischargeoffset / (60.0 * 24.0) >= {MIN_LOS_DAYS}
          AND p.unitvisitNumber = 1
    """

    sql_mechvent = f"""
    WITH eligible AS ({_elig})
    SELECT ve.patientunitstayid AS stay_id, ve.event, ve.hrs
    FROM `{EICU['derived']}.ventilation_events` ve
    JOIN eligible e ON ve.patientunitstayid = e.patientunitstayid
    WHERE ve.event IN ('mechvent start', 'mechvent end', 'Trach')
      AND ve.hrs <= {VFD_HORIZON_DAYS * 24}
    ORDER BY ve.patientunitstayid, ve.hrs
    """

    sql_death = f"""
    SELECT
        p.patientunitstayid AS stay_id,
        CASE
            WHEN p.hospitaldischargestatus = 'Expired' THEN 1
            WHEN p.unitdischargestatus = 'Expired' THEN 1
            ELSE 0
        END AS died_28d,
        p.hospitaldischargeoffset / (60.0 * 24.0) AS days_to_discharge
    FROM `{EICU['main']}.patient` p
    WHERE p.age != ''
      AND SAFE_CAST(REGEXP_REPLACE(p.age, '> ', '') AS INT64) >= 18
      AND p.unitdischargeoffset / (60.0 * 24.0) >= {MIN_LOS_DAYS}
      AND p.unitvisitNumber = 1
    """

    print("6.4  Computing eICU VFD-28...")
    df_mechvent = run_bq(sql_mechvent)
    df_death    = run_bq(sql_death)

    horizon_hrs = VFD_HORIZON_DAYS * 24

    def _imv_hours(grp):
        grp = grp.sort_values("hrs")
        total, on, t0 = 0.0, False, 0.0
        for _, row in grp.iterrows():
            if row["event"] in ("mechvent start", "Trach") and not on:
                on, t0 = True, max(row["hrs"], 0)
            elif row["event"] == "mechvent end" and on:
                total += row["hrs"] - t0
                on = False
        if on:
            total += horizon_hrs - t0
        return total

    if len(df_mechvent) > 0:
        imv_series = df_mechvent.groupby("stay_id").apply(_imv_hours)
        df_imv_hrs = imv_series.reset_index()
        df_imv_hrs.columns = ["stay_id", "total_imv_hours"]
    else:
        df_imv_hrs = pd.DataFrame(columns=["stay_id", "total_imv_hours"])

    df = df_eicu[["stay_id", "Treatment_W"]].copy()
    df = df.merge(df_imv_hrs, on="stay_id", how="left")
    df = df.merge(df_death, on="stay_id", how="left")

    df["total_imv_hours"] = df["total_imv_hours"].fillna(0)
    df["died_28d"] = df["died_28d"].fillna(0).astype(int)
    df["total_imv_days"] = df["total_imv_hours"] / 24.0
    df["vfd28"] = np.where(
        df["died_28d"] == 1, 0.0,
        np.clip(28.0 - df["total_imv_days"], 0, 28))
    df["delta"] = 1 - df["died_28d"]

    print(f"     eICU VFD-28: mean={df['vfd28'].mean():.1f}, "
          f"median={df['vfd28'].median():.1f}")
    return df


def propensity_score_match_baseline(X_baseline, W, feature_cols,
                                     caliper_scale=0.2, random_state=42):
    X_clean = np.nan_to_num(X_baseline, nan=0.0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    ps_model = LogisticRegression(max_iter=1000, random_state=random_state)
    ps_model.fit(X_scaled, W)

    ps = ps_model.predict_proba(X_scaled)[:, 1]
    logit_ps = np.log(ps / (1 - ps + 1e-8))
    caliper = caliper_scale * logit_ps.std()

    nirs_idx = np.where(W == 1)[0]
    imv_idx = np.where(W == 0)[0]

    nn_matcher = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
    nn_matcher.fit(logit_ps[imv_idx].reshape(-1, 1))
    distances, matched_idx = nn_matcher.kneighbors(
        logit_ps[nirs_idx].reshape(-1, 1))

    within_caliper = distances.flatten() <= caliper
    matched_nirs = nirs_idx[within_caliper]
    matched_imv = imv_idx[matched_idx.flatten()[within_caliper]]

    psm_idx = np.concatenate([matched_nirs, matched_imv])
    np.random.RandomState(random_state).shuffle(psm_idx)

    print(f'eICU PSM cohort: {len(psm_idx):,} patients '
          f'({len(matched_nirs):,} matched pairs)')
    print(f'  Caliper: {caliper:.4f}')
    print(f'  NIRS matched: {len(matched_nirs):,} / {len(nirs_idx):,}')
    print(f'  Unmatched NIRS dropped: {len(nirs_idx) - len(matched_nirs):,}')
    return psm_idx, ps_model, ps


def run_mimic_extraction(client=None, dataset_id=None, seq_len=48,
                         chunk_size=5000):
    if client is not None:
        global _client
        _client = client

    print("\n" + "=" * 65)
    print("  VT-NIRS FULL EXTRACTION PIPELINE")
    print("=" * 65)

    df_cohort = build_cohort()

    print("\n" + "=" * 65)
    print("  VFD-28 OUTCOME")
    print("=" * 65)
    df_vfd = compute_vfd28(df_cohort)

    df_cohort = df_cohort.merge(
        df_vfd[["stay_id", "vfd28", "delta", "died_28d"]],
        on="stay_id", how="inner")

    print("\n" + "=" * 65)
    print("  COVARIATE ENGINEERING")
    print("=" * 65)
    df_cov = extract_baseline_covariates(df_cohort)

    if 'sofa_X' in df_cov.columns:
        df_cohort = df_cohort.merge(
            df_cov[['stay_id', 'sofa_X']], on='stay_id', how='left')

    print("\n" + "=" * 65)
    print("  TEMPORAL COVARIATE EXTRACTION (chartevents)")
    print("=" * 65)
    df_vitals = extract_temporal_chartevents(df_cohort, chunk_size)

    print("\nBuilding time series tensors...")
    X, W, VFD, D, valid_ids = build_temporal_sequences(
        df_vitals, df_cohort, seq_len=seq_len)

    return {
        'X': X, 'W': W, 'VFD': VFD, 'D': D,
        'valid_ids': valid_ids,
        'df_cohort': df_cohort,
        'df_vfd': df_vfd,
        'df_cov': df_cov,
    }


def extract_eicu_temporal(df_eicu_tx, chunk_size=5000):
    stay_ids = df_eicu_tx['stay_id'].tolist()
    chunks = [stay_ids[i:i + chunk_size]
              for i in range(0, len(stay_ids), chunk_size)]

    print("\n7.1  Extracting eICU temporal vitals (nursecharting + vitalperiodic)...")

    results_nc = []
    for ci, chunk in enumerate(chunks):
        chunk_str = ','.join(map(str, chunk))
        sql_nc = f"""
        SELECT
            nc.patientunitstayid AS stay_id,
            CAST(FLOOR(nc.nursingchartoffset / 30.0) * 30 AS INT64) AS offset_bin,
            AVG(CASE WHEN nc.nursingchartcelltypevalname = 'Heart Rate'
                     THEN SAFE_CAST(nc.nursingchartvalue AS FLOAT64) END) AS heart_rate,
            AVG(CASE WHEN nc.nursingchartcelltypevalname = 'Respiratory Rate'
                     THEN SAFE_CAST(nc.nursingchartvalue AS FLOAT64) END) AS resp_rate,
            AVG(CASE WHEN nc.nursingchartcelltypevalname = 'O2 Saturation'
                     THEN SAFE_CAST(nc.nursingchartvalue AS FLOAT64) END) AS spo2,
            AVG(CASE WHEN nc.nursingchartcelltypevalname = 'Non-Invasive BP Mean'
                     THEN SAFE_CAST(nc.nursingchartvalue AS FLOAT64) END) AS mbp,
            AVG(CASE WHEN nc.nursingchartcelltypevalname = 'Temperature (C)'
                     THEN SAFE_CAST(nc.nursingchartvalue AS FLOAT64) END) AS temperature
        FROM `{EICU['main']}.nursecharting` nc
        WHERE nc.patientunitstayid IN ({chunk_str})
          AND nc.nursingchartoffset >= 0
          AND nc.nursingchartoffset <= 1440
        GROUP BY nc.patientunitstayid, offset_bin
        """
        chunk_df = run_bq(sql_nc, verbose=False)
        results_nc.append(chunk_df)
        print(f'  nursecharting chunk {ci+1}/{len(chunks)}: {len(chunk_df):,} rows')

    df_nc = pd.concat(results_nc, ignore_index=True) if results_nc else pd.DataFrame()

    print("7.2  Extracting eICU temporal vitals (vitalperiodic)...")
    results_vp = []
    for ci, chunk in enumerate(chunks):
        chunk_str = ','.join(map(str, chunk))
        sql_vp = f"""
        SELECT
            vp.patientunitstayid AS stay_id,
            CAST(FLOOR(vp.observationoffset / 30.0) * 30 AS INT64) AS offset_bin,
            AVG(vp.heartrate)       AS heart_rate_vp,
            AVG(vp.sao2)            AS spo2_vp,
            AVG(vp.systemicmean)    AS mbp_vp,
            AVG(vp.temperature)     AS temperature_vp
        FROM `{EICU['main']}.vitalperiodic` vp
        WHERE vp.patientunitstayid IN ({chunk_str})
          AND vp.observationoffset >= 0
          AND vp.observationoffset <= 1440
        GROUP BY vp.patientunitstayid, offset_bin
        """
        chunk_df = run_bq(sql_vp, verbose=False)
        results_vp.append(chunk_df)
        print(f'  vitalperiodic chunk {ci+1}/{len(chunks)}: {len(chunk_df):,} rows')

    df_vp = pd.concat(results_vp, ignore_index=True) if results_vp else pd.DataFrame()

    print("7.3  Extracting eICU temporal labs...")
    results_lab = []
    for ci, chunk in enumerate(chunks):
        chunk_str = ','.join(map(str, chunk))
        sql_lab = f"""
        SELECT
            l.patientunitstayid AS stay_id,
            CAST(FLOOR(l.labresultoffset / 30.0) * 30 AS INT64) AS offset_bin,
            AVG(CASE WHEN LOWER(l.labname) IN ('pao2', 'paO2')
                     THEN l.labresult END) AS pao2,
            AVG(CASE WHEN LOWER(l.labname) IN ('paco2', 'paCO2')
                     THEN l.labresult END) AS paco2,
            AVG(CASE WHEN LOWER(l.labname) IN ('ph', 'pH')
                     THEN l.labresult END) AS ph,
            AVG(CASE WHEN LOWER(l.labname) IN ('lactate', 'lactic acid')
                     THEN l.labresult END) AS lactate,
            AVG(CASE WHEN LOWER(l.labname) = 'creatinine'
                     THEN l.labresult END) AS creatinine,
            AVG(CASE WHEN LOWER(l.labname) IN ('total bilirubin', 'bilirubin')
                     THEN l.labresult END) AS bilirubin,
            AVG(CASE WHEN LOWER(l.labname) = 'platelets x 1000'
                     THEN l.labresult END) AS platelets,
            AVG(CASE WHEN LOWER(l.labname) IN ('wbc x 1000', 'wbc')
                     THEN l.labresult END) AS wbc
        FROM `{EICU['main']}.lab` l
        WHERE l.patientunitstayid IN ({chunk_str})
          AND l.labresultoffset >= 0
          AND l.labresultoffset <= 1440
        GROUP BY l.patientunitstayid, offset_bin
        """
        chunk_df = run_bq(sql_lab, verbose=False)
        results_lab.append(chunk_df)
        print(f'  lab chunk {ci+1}/{len(chunks)}: {len(chunk_df):,} rows')

    df_lab = pd.concat(results_lab, ignore_index=True) if results_lab else pd.DataFrame()

    print("7.4  Extracting eICU temporal respiratory (FiO2, PEEP)...")
    results_resp = []
    for ci, chunk in enumerate(chunks):
        chunk_str = ','.join(map(str, chunk))
        sql_resp = f"""
        SELECT
            rc.patientunitstayid AS stay_id,
            CAST(FLOOR(rc.respchartoffset / 30.0) * 30 AS INT64) AS offset_bin,
            AVG(CASE WHEN LOWER(rc.respchartvaluelabel) LIKE '%fio2%'
                     THEN SAFE_CAST(rc.respchartvalue AS FLOAT64) END) AS fio2,
            AVG(CASE WHEN LOWER(rc.respchartvaluelabel) LIKE '%peep%'
                     THEN SAFE_CAST(rc.respchartvalue AS FLOAT64) END) AS peep
        FROM `{EICU['main']}.respiratorycharting` rc
        WHERE rc.patientunitstayid IN ({chunk_str})
          AND rc.respchartoffset >= 0
          AND rc.respchartoffset <= 1440
          AND SAFE_CAST(rc.respchartvalue AS FLOAT64) IS NOT NULL
        GROUP BY rc.patientunitstayid, offset_bin
        """
        chunk_df = run_bq(sql_resp, verbose=False)
        results_resp.append(chunk_df)
        print(f'  respiratorycharting chunk {ci+1}/{len(chunks)}: {len(chunk_df):,} rows')

    df_resp = pd.concat(results_resp, ignore_index=True) if results_resp else pd.DataFrame()

    print("\n7.5  Merging temporal sources...")

    all_stay_ids = df_eicu_tx['stay_id'].unique()
    bins = np.arange(0, 1440, 30)
    idx = pd.MultiIndex.from_product([all_stay_ids, bins],
                                      names=['stay_id', 'offset_bin'])
    df_temporal = pd.DataFrame(index=idx).reset_index()

    if len(df_nc) > 0:
        df_temporal = df_temporal.merge(
            df_nc, on=['stay_id', 'offset_bin'], how='left')

    if len(df_vp) > 0:
        df_temporal = df_temporal.merge(
            df_vp, on=['stay_id', 'offset_bin'], how='left')
        for col, vp_col in [('heart_rate', 'heart_rate_vp'),
                             ('spo2', 'spo2_vp'),
                             ('mbp', 'mbp_vp'),
                             ('temperature', 'temperature_vp')]:
            if col in df_temporal.columns and vp_col in df_temporal.columns:
                df_temporal[col] = df_temporal[col].fillna(df_temporal[vp_col])
                df_temporal.drop(columns=[vp_col], inplace=True)

    if len(df_lab) > 0:
        df_temporal = df_temporal.merge(
            df_lab, on=['stay_id', 'offset_bin'], how='left')

    if len(df_resp) > 0:
        df_temporal = df_temporal.merge(
            df_resp, on=['stay_id', 'offset_bin'], how='left')

    for col in ['heart_rate', 'resp_rate', 'spo2', 'mbp', 'temperature',
                'fio2', 'peep', 'pao2', 'paco2', 'ph', 'lactate',
                'creatinine', 'bilirubin', 'platelets', 'wbc']:
        if col not in df_temporal.columns:
            df_temporal[col] = np.nan

    if 'fio2' in df_temporal.columns:
        mask_frac = df_temporal['fio2'] <= 1.0
        df_temporal.loc[mask_frac, 'fio2'] = df_temporal.loc[mask_frac, 'fio2'] * 100.0

    n_patients = df_temporal['stay_id'].nunique()
    n_rows = len(df_temporal)
    print(f'\n  eICU temporal data: {n_rows:,} rows, {n_patients:,} patients')
    for col in ['heart_rate', 'resp_rate', 'spo2', 'fio2', 'pao2', 'lactate']:
        pct = df_temporal[col].notna().mean() * 100
        print(f'    {col}: {pct:.1f}% non-null')

    return df_temporal


def build_eicu_temporal_sequences(df_temporal, df_eicu_tx, df_eicu_vfd,
                                   df_eicu_cov, seq_len=48):
    def _pad_sequences(seqs, maxlen, dtype='float32', padding='pre', value=0.0):
        n_feat = seqs[0].shape[1] if len(seqs) > 0 else 0
        out = np.full((len(seqs), maxlen, n_feat), value, dtype=dtype)
        for i, s in enumerate(seqs):
            trunc = s[-maxlen:] if len(s) > maxlen else s
            if padding == 'pre':
                out[i, maxlen - len(trunc):, :] = trunc
            else:
                out[i, :len(trunc), :] = trunc
        return out

    static = df_eicu_tx[['stay_id']].drop_duplicates().copy()
    static = static.merge(
        df_eicu_tx[['stay_id', 'age_clean', 'gender']].drop_duplicates('stay_id'),
        on='stay_id', how='left')
    static = static.merge(
        df_eicu_cov[['stay_id', 'bmi_X', 'sofa_X', 'gcs_X']].drop_duplicates('stay_id'),
        on='stay_id', how='left')
    static = static.merge(
        df_eicu_vfd[['stay_id', 'Treatment_W', 'vfd28', 'died_28d']],
        on='stay_id', how='left')
    static['gender_num'] = (static['gender'] == 'Male').astype(float)
    static_dict = static.set_index('stay_id').to_dict('index')

    sequences, treatments, vfd_list, delta_list, valid_ids = [], [], [], [], []

    for stay_id, group in df_temporal.groupby('stay_id'):
        if stay_id not in static_dict:
            continue
        s = static_dict[stay_id]
        if pd.isna(s.get('Treatment_W')) or pd.isna(s.get('vfd28')):
            continue

        group = group.sort_values('offset_bin')

        vital_cols = ['heart_rate', 'resp_rate', 'spo2', 'mbp', 'temperature',
                      'fio2', 'peep', 'pao2', 'paco2', 'ph', 'lactate',
                      'creatinine', 'bilirubin', 'platelets', 'wbc']
        for col in vital_cols:
            if col in group.columns:
                group[col] = group[col].ffill().bfill()

        hours_since_admit = group['offset_bin'].values / 60.0
        fio2_vals = group['fio2'].values if 'fio2' in group.columns else np.full(len(group), np.nan)
        pao2_vals = group['pao2'].values if 'pao2' in group.columns else np.full(len(group), np.nan)

        fio2_frac = np.where(fio2_vals > 0, fio2_vals / 100.0, np.nan)
        fio2_frac = np.clip(fio2_frac, 0.21, 1.0)
        pf_ratio = np.where(np.isfinite(pao2_vals) & np.isfinite(fio2_frac),
                            pao2_vals / fio2_frac, np.nan)

        n_rows = len(group)
        feat_matrix = np.column_stack([
            np.full(n_rows, s.get('age_clean', np.nan)),
            np.full(n_rows, s.get('gender_num', 0)),
            np.full(n_rows, s.get('bmi_X', np.nan)),
            group['heart_rate'].values if 'heart_rate' in group.columns else np.full(n_rows, np.nan),
            group['resp_rate'].values if 'resp_rate' in group.columns else np.full(n_rows, np.nan),
            group['spo2'].values if 'spo2' in group.columns else np.full(n_rows, np.nan),
            group['mbp'].values if 'mbp' in group.columns else np.full(n_rows, np.nan),
            group['temperature'].values if 'temperature' in group.columns else np.full(n_rows, np.nan),
            fio2_vals,
            group['peep'].values if 'peep' in group.columns else np.full(n_rows, np.nan),
            pao2_vals,
            group['paco2'].values if 'paco2' in group.columns else np.full(n_rows, np.nan),
            group['ph'].values if 'ph' in group.columns else np.full(n_rows, np.nan),
            pf_ratio,
            group['lactate'].values if 'lactate' in group.columns else np.full(n_rows, np.nan),
            group['creatinine'].values if 'creatinine' in group.columns else np.full(n_rows, np.nan),
            group['bilirubin'].values if 'bilirubin' in group.columns else np.full(n_rows, np.nan),
            group['platelets'].values if 'platelets' in group.columns else np.full(n_rows, np.nan),
            group['wbc'].values if 'wbc' in group.columns else np.full(n_rows, np.nan),
            np.full(n_rows, s.get('sofa_X', np.nan)),
            np.full(n_rows, s.get('gcs_X', np.nan)),
            hours_since_admit,
            np.zeros(n_rows),
        ])

        feat_matrix = np.nan_to_num(feat_matrix, nan=0.0)

        if feat_matrix.shape[0] < 2:
            continue

        sequences.append(feat_matrix)
        treatments.append(s['Treatment_W'])
        vfd_list.append(s['vfd28'])
        delta_list.append(1.0)
        valid_ids.append(stay_id)

    if len(sequences) == 0:
        print("WARNING: No valid eICU temporal sequences built!")
        return np.array([]), np.array([]), np.array([]), np.array([]), []

    X = _pad_sequences(sequences, maxlen=seq_len, dtype='float32',
                       padding='pre', value=0.0)
    W = np.array(treatments, dtype=np.float32)
    VFD = np.array(vfd_list, dtype=np.float32)
    D = np.array(delta_list, dtype=np.float32)

    print(f'\neICU temporal sequences built:')
    print(f'  X shape: {X.shape}  (patients, timesteps, covariates)')
    print(f'  NIRS (W=1): {(W == 1).sum():,}  |  IMV (W=0): {(W == 0).sum():,}')
    print(f'  Patients dropped (insufficient data): '
          f'{len(df_eicu_tx) - len(valid_ids):,}')
    return X, W, VFD, D, valid_ids
