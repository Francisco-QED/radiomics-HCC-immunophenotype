##### ------------ MRI-radiomics for prediction of immunophenotype in HCC -------------- #####
# Requirements:
# - Radiomics features are extracted for all patients and stored in a CSV file.
# - Inter-reader features for ICC calculation are available from patients segmented by two readers.
# - Clinical metadata and outcomes are available in CSV or Excel format, including the binary label.
#
# Pipeline overview (run one stage at a time via --stage):
#   --stage partition  : preselect sequences, partition dev/lock, global cleanup + ICC filtering,
#                         cross-validated consensus feature selection, save processed dev/lock artifacts.
#   --stage gridsearch  : load processed dev features, tune hyperparameters per model, pickle best estimators.
#   --stage train       : load processed dev/lock features (+ tuned estimators if available), evaluate
#                         every model on the lock set with bootstrap CIs at a fixed decision threshold.
#
# Example:
#   python main_pipeline.py --stage partition
#   python main_pipeline.py --stage gridsearch
#   python main_pipeline.py --stage train --bootstrap 2000 --threshold 0.35 --shap

import os
# Must be set before numpy/sklearn (libomp) and torch/tabpfn (libiomp) are imported -- having both
# OpenMP runtimes loaded at once is a known cause of silent hangs/crashes on macOS.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import pickle
import argparse
import textwrap
import warnings
from pathlib import Path

import shap
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt

from scipy.stats import mannwhitneyu, chi2_contingency
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from sklearn.svm import SVC
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve

from xgboost import XGBClassifier
import torch
from tabpfn import TabPFNClassifier

# --- Global, single-source-of-truth constants -----------------------------------------------------
# Centralizing these means a sensitivity run only requires changing one line, not hunting for
# every hardcoded literal scattered across the file.
RANDOM_STATE = 42        # All CV splits, model fitting, and bootstrap resampling.
PARTITION_SEED = 997     # Dev/lock split uses a distinct seed on purpose (separate stage, separate concern).
TARGET_THRESHOLD = 0.35  # Decision threshold applied to predict_proba for sensitivity/specificity + their CIs.

# Single source of truth for which models run through the nested-CV diagnostic, the gridsearch
# stage, and the train stage -- so adding/removing a model is a one-line change instead of three
# call sites silently drifting out of sync.
ALL_MODELS = ['logistic', 'svm_linear', 'svm_rbf', 'rf', 'extratrees', 'xgboost', 'gp', 'tabpfn']

# Stage to run when launching this script without a --stage flag (e.g. clicking "Run" in an IDE).
# Still overridable from the terminal with `--stage <name>`.
DEFAULT_STAGE = 'gridsearch'  # one of: 'partition', 'gridsearch', 'train'


# =====================================================================================================
# STAGE 1: Sequence preselection, partitioning, cleanup, and cross-validated feature selection
# =====================================================================================================

# The first step is to select the MRI sequences that will be included in the model.
# df: DataFrame containing radiomics features for all patients. Each column represents a
#     feature, and the column names include the sequence type (e.g., T1, T2, DWI).
# allowed_sequences: List of sequences to keep (e.g., ['T1', 'T2', 'DWI']). If empty, all sequences are kept.
def preselect_sequences(df, allowed_sequences, id_col='PatientID'):
    if not allowed_sequences:
        print("No sequence filter applied. Keeping all features.")
        return df

    # Match on the leading "{sequence}_" prefix segment, not a substring anywhere in the column
    # name -- a plain `in` check let 'ap' match "...original_shape_..." (since "shape" contains
    # "ap"), silently leaking every other sequence's shape features into any combo containing 'ap'.
    cols_to_keep = [id_col] if id_col in df.columns else []
    for col in df.columns:
        if any(col.lower().startswith(seq.lower() + '_') for seq in allowed_sequences):
            cols_to_keep.append(col)

    filtered_df = df[cols_to_keep]
    print(f"Preselection complete. Kept sequences: {allowed_sequences}")
    print(f"Features reduced from {df.shape[1]} to {filtered_df.shape[1]}")
    return filtered_df



def filter_feature_families(df, exclude_families=None, id_col='PatientID'):
    """Drop PyRadiomics filter families (e.g. wavelet) from the feature matrix.

    PyRadiomics feature names spell out their filter family explicitly (e.g.
    'pre_wavelet-LLL_glcm_Autocorrelation'), so a plain substring match is enough -- no other
    sequence, feature class, or feature name in this dataset contains 'wavelet' as a substring.
    Wavelet is excluded by default (see --exclude-feature-families): it accounts for ~61% of
    extracted columns and is the filter family most consistently flagged in the radiomics
    reproducibility literature as sensitive to acquisition/reconstruction variability, without a
    matching gain in candidate signal (most of its 8 subbands are highly correlated
    re-expressions of the same texture).
    """
    if not exclude_families:
        return df

    cols_to_keep = [
        col for col in df.columns
        if col == id_col or not any(fam.lower() in col.lower() for fam in exclude_families)
    ]
    filtered_df = df[cols_to_keep]
    print(f"Filtered out feature families {exclude_families}. "
          f"Features reduced from {df.shape[1]} to {filtered_df.shape[1]}")
    return filtered_df


# --- Clinical / MRI semantic features -----------------------------------------------------------
# Non-radiomics features from the clinical/imaging metadata sheet, engineered to be numeric and
# concatenated onto the radiomics feature matrix (see run_partition_stage) so LASSO can select from
# the combined pool exactly like any other feature. Prefixed so downstream code can identify them
# (e.g. to skip the radiomics-only ICC filter, which has no inter-reader analogue for these).
#
# Feature set is deliberately narrow: only baseline (pre-operative) variables that are both
# consistently coded and have real variance in this cohort. Excluded on purpose:
#   - Everything derived from the resection specimen (Tumor size (cm), Tumor grade, Tumor stage,
#     Histopathologic characteristics, Tumor_differentiation) -- not known at baseline, and Tumor
#     grade in particular is assigned on the same specimen the TILs outcome is scored from.
#   - Most "Qualitative MRI evaluation" columns (Morphology, Margin, Internal arteries,
#     Macrovascular invasion, Tumor necrosis, Fat content, Restricted diffusion, Enhancement
#     characteristics, LI-RADS, HBP signal) -- >85% single-category in this cohort (e.g. Morphology
#     is 145/147 "Nodular"), some also inconsistently coded (typos, stray "N/a" strings, LI-RADS
#     values like "FNH?" that aren't valid LI-RADS codes).
CLINICAL_FEATURE_PREFIX = 'clinical_'

# T2 signal free text, case-normalized: hypointense-leaning (Iso/Hypo) vs hyperintense-leaning
# (Mild/Mild Hyper/Hyper) -- "Mild Hyper" and "Mild hyper" are the same category with inconsistent
# capitalization in the source sheet.
_T2_ISOHYPO = {'iso', 'hypo'}
_T2_HYPER = {'mild', 'mild hyper', 'hyper'}


# Buckets the free-text, often multi-label "Etiology of liver disease" field into a single
# category per patient, checked in this priority order:
#   1. Viral (HBV or HCV present anywhere in the field)
#   2. Steatosis-related (MASH/NASH/MASLD/NAFLD/steatosis), only reached if not viral
#   3. Other (EtOH, Unknown/NaN, and incidental comorbidities like "Multiple Myeloma" that clearly
#      aren't a liver-disease etiology) -- each too rare on its own to be its own category
# The order matters: a patient listed as "HCV, MASH" returns 'Viral', never reaching the MASH
# check, since viral hepatitis is the dominant/primary driver when both are present.
def _classify_etiology(text):
    if pd.isna(text):
        return 'Other'
    t = str(text).lower()
    if 'hbv' in t or 'hcv' in t:
        return 'Viral'
    if any(k in t for k in ('mash', 'nash', 'masld', 'nafld', 'steatosis')):
        return 'MASLD/MASH/NASH/NAFLD'
    return 'Other'


# Loads and encodes the clinical/MRI semantic features from the clinical metadata sheet into a
# numeric, model-ready DataFrame indexed by PatientID.
def load_clinical_features(clinical_data_path, id_col='PatientID'):
    df = pd.read_excel(clinical_data_path).set_index(id_col)
    out = pd.DataFrame(index=df.index)

    out[f'{CLINICAL_FEATURE_PREFIX}Age'] = df['Age']
    out[f'{CLINICAL_FEATURE_PREFIX}BMI'] = df['BMI']
    out[f'{CLINICAL_FEATURE_PREFIX}Gender_M'] = (df['Gender'] == 'M').astype(float)
    out[f'{CLINICAL_FEATURE_PREFIX}Cirrhosis_Y'] = df['Cirrhosis'].map({'Y': 1.0, 'N': 0.0})
    # AFP is classically log-normally distributed (here spanning ~2 to ~13,900) -- log1p keeps a
    # handful of extreme values from dominating the later z-score/LASSO step.
    out[f'{CLINICAL_FEATURE_PREFIX}AFP_log'] = np.log1p(df['Serum AFP'])

    out[f'{CLINICAL_FEATURE_PREFIX}Capsule_PVP_Y'] = df['Capsule (on PVP yes/no)'].map({'Yes': 1.0, 'No': 0.0})
    out[f'{CLINICAL_FEATURE_PREFIX}Mosaic_Y'] = df['Mosaic appearance (yes/no)'].map({'Yes': 1.0, 'No': 0.0})
    out[f'{CLINICAL_FEATURE_PREFIX}TumorSize_PVP'] = df['Maximum tumor size on T1 portal venous phase (PVP)']

    t2 = df['T2 signal (Iso, hypo, Mild, marked Hyperintensity)'].str.strip().str.lower()
    out[f'{CLINICAL_FEATURE_PREFIX}T2_Hyper'] = np.select(
        [t2.isin(_T2_ISOHYPO), t2.isin(_T2_HYPER)], [0.0, 1.0], default=np.nan)

    etiology_group = df['Etiology of liver disease'].apply(_classify_etiology)
    etiology_dummies = pd.get_dummies(etiology_group, prefix=f'{CLINICAL_FEATURE_PREFIX}Etiology', dtype=float)
    out = out.join(etiology_dummies)

    print(f"Loaded {out.shape[1]} clinical features for {out.shape[0]} patients from: {clinical_data_path}")
    return out


# --- Partitioning with ICC and lock set constraints -------------------------------------------------
# Partition the data into development and lock sets. Ensure all inter-reader patients remain in the
# development set, while the remaining pool is stratified by outcome for the lock set.
#
# outcomes_path: Path to the excel file containing outcomes and labels.
# icc_path: Path to the CSV containing patients with double ROIs. Note: this is typically the same
#           file as path_r1 (reader 1's per-feature CSV), which is already restricted to the
#           inter-reader patient subset, so it doubles as the source of "must be in dev" patient IDs.
# clinical_data_path: Path to the clinical metadata file (Excel) containing patient clinical information.
# cutoff: Threshold for binarizing the outcome variable.
# test_size: Number of patients to include in the lock set.
def partition_with_constraints(df_filtered, outcomes_path, icc_path, clinical_data_path, label_col,
                                cutoff, test_size, output_dir, random_state=PARTITION_SEED,
                                clinical_output_dir=None):
    df_outcomes = pd.read_excel(outcomes_path)
    # NaN >= cutoff evaluates to False in pandas, which .astype(int) would silently turn into a
    # confirmed "0" (low) label -- patients with unscorable label_col (e.g. "No tumor in slide",
    # so TILs_resection was never assessed) would be miscoded as genuine negatives rather than
    # excluded. Drop them before computing y_bin, matching what compare_dev_test() already does
    # independently for its own descriptive table (that filter never touched X_dev/y_dev/X_lock/
    # y_lock, which are finalized here, before compare_dev_test ever runs).
    n_before = len(df_outcomes)
    df_outcomes = df_outcomes[df_outcomes[label_col].notna()].copy()
    if len(df_outcomes) < n_before:
        print(f"Dropped {n_before - len(df_outcomes)} patients with missing '{label_col}' "
              f"before computing y_bin (would otherwise be miscoded as class 0).")
    df_outcomes['y_bin'] = (df_outcomes[label_col] >= cutoff).astype(int)
    df_icc = pd.read_csv(icc_path)

    df_merged = df_filtered.reset_index().merge(df_outcomes[['PatientID', 'y_bin']], on='PatientID', how='inner').set_index('PatientID')

    # Sort by PatientID before any split: train_test_split's shuffling depends on row order even
    # under a fixed random_state, so without this, which patients land in dev vs lock would depend
    # on incidental row order in whichever features CSV was loaded -- not just on patient identity
    # + seed. Sorting pins the split to patient identity alone (see recurrence_pipeline.py's
    # partition_with_constraints_survival for the concrete failure mode this prevents).
    df_merged = df_merged.sort_index()

    icc_ids = df_icc['PatientID'].unique()
    # Dev-Must: Patients with double ROIs (inter-reader cases must be in the development set)
    df_dev_must = df_merged[df_merged.index.isin(icc_ids)]
    # Flexible: Everyone else
    df_rest = df_merged[~df_merged.index.isin(icc_ids)]

    X_rest = df_rest.drop(columns=['y_bin'])
    y_rest = df_rest['y_bin']

    X_dev_rest, X_lock, y_dev_rest, y_lock = train_test_split(
        X_rest, y_rest,
        test_size=test_size,
        stratify=y_rest,
        random_state=random_state
    )

    X_dev = pd.concat([df_dev_must.drop(columns=['y_bin']), X_dev_rest])
    y_dev = pd.concat([df_dev_must['y_bin'], y_dev_rest])

    df_clinical = pd.read_excel(clinical_data_path)
    # Outcome columns (TILs_resection, TLS_resection, ...) live in outcomes_path, not the clinical
    # metadata sheet -- merged in here so compare_dev_test can use them (e.g. as --compare-outcome-col)
    # without depending on a duplicate copy sitting inside the clinical sheet.
    outcome_cols = [c for c in df_outcomes.columns if c not in ('PatientID', 'y_bin')]
    df_clinical = df_clinical.merge(df_outcomes[['PatientID'] + outcome_cols], on='PatientID', how='left')
    df_clinical_dev = df_clinical[df_clinical['PatientID'].isin(X_dev.index)]
    df_clinical_lock = df_clinical[df_clinical['PatientID'].isin(X_lock.index)]

    # The dev/lock split (and everything derived from it here) depends only on cutoff/test_size/
    # random_state, never on which sequences are in df_filtered -- so a sweep over sequence combos
    # can point clinical_output_dir at one shared location instead of writing N identical copies
    # into N per-combo folders. Defaults to output_dir for a plain, non-sweep single run.
    _clin_dir = clinical_output_dir or output_dir
    Path(_clin_dir).mkdir(parents=True, exist_ok=True)
    df_clinical_dev.to_csv(f"{_clin_dir}/clinical_metadata_dev.csv", index=False)
    df_clinical_lock.to_csv(f"{_clin_dir}/clinical_metadata_lock.csv", index=False)

    print("Partitioning Complete.")
    print(f"Dev Set: {len(X_dev)} (Includes {len(df_dev_must)} ICC patients)")
    print(f"Lock Set: {len(X_lock)}")
    print(f"Sub-sheets created: Dev ({len(df_clinical_dev)}), Lock ({len(df_clinical_lock)})")

    return X_dev, y_dev, X_lock, y_lock, df_clinical_dev, df_clinical_lock


# Creates a summary table comparing Development and Lock sets, to confirm there are no significant
# differences in key clinical variables between the two partitions.
#   Continuous (Age, BMI): Median [IQR] + Mann-Whitney U
#   Categorical (Cirrhosis, Gender): n (%) for all levels + Chi-Square
def compare_dev_test(df_dev, df_lock, variables, output_dir, outcome_col):
    before_dev = len(df_dev)
    df_dev = df_dev[df_dev[outcome_col].notnull()].copy()
    if before_dev != len(df_dev):
        print(f"Dropped {before_dev - len(df_dev)} dev rows missing '{outcome_col}'")
    before_lock = len(df_lock)
    df_lock = df_lock[df_lock[outcome_col].notnull()].copy()
    if before_lock != len(df_lock):
        print(f"Dropped {before_lock - len(df_lock)} lock rows missing '{outcome_col}'")

    results = []

    for var in variables:
        is_categorical = df_dev[var].dtype == 'object' or df_dev[var].nunique() < 5

        if is_categorical:
            def get_cat_str(df, column):
                counts = df[column].value_counts()
                total = len(df[column].dropna())
                return ", ".join([f"{idx}: {val} ({(val/total)*100:.1f}%)"
                                 for idx, val in counts.items()])

            dev_str = get_cat_str(df_dev, var)
            lock_str = get_cat_str(df_lock, var)

            combined_series = pd.concat([df_dev[var], df_lock[var]])
            group_labels = pd.Series(['Dev'] * len(df_dev) + ['Lock'] * len(df_lock), index=combined_series.index)
            contingency = pd.crosstab(combined_series, group_labels)
            print(f"Contingency Table for {var}:\n{contingency}\n")
            try:
                _, p, _, _ = chi2_contingency(contingency)
            except ValueError:
                p = np.nan
            test_type = 'Chi-Square'

        else:
            q1_d, med_d, q3_d = df_dev[var].quantile([0.25, 0.5, 0.75])
            q1_l, med_l, q3_l = df_lock[var].quantile([0.25, 0.5, 0.75])

            dev_str = f"{med_d:.1f} [{q1_d:.1f} - {q3_d:.1f}]"
            lock_str = f"{med_l:.1f} [{q1_l:.1f} - {q3_l:.1f}]"

            _, p = mannwhitneyu(df_dev[var].dropna(), df_lock[var].dropna(), alternative='two-sided')
            test_type = 'Mann-Whitney U'

        results.append({
            'Variable': var,
            'Development Set (N={})'.format(len(df_dev)): dev_str,
            'Test Set (N={})'.format(len(df_lock)): lock_str,
            'p-value': round(p, 4) if not np.isnan(p) else "N/A",
            'Statistical Test': test_type
        })

    results = pd.DataFrame(results)
    results.to_excel(f"{output_dir}/dev_test_comparison.xlsx", index=False)
    return results


# This performs the cleanup that needs to happen inside the development set only: drop nans and
# drop near-zero variance features. We do this to avoid data leakage, then apply the same
# transformations to the lock set later.
# df: development set X_dev from "partition_with_constraints"
# cv_threshold: minimum coefficient of variation (std / |mean|) to keep a feature. CV is
#   scale-invariant, unlike a raw std cutoff, which is what we want across thousands of radiomics
#   features that live on wildly different scales (e.g. a firstorder_Energy in the millions next to
#   a glcm_Correlation in [-1, 1]) -- a fixed std threshold would systematically favor keeping
#   large-scale features and dropping small-scale ones regardless of their actual informativeness.
def global_cleanup(df, cv_threshold, id_col='PatientID'):
    if id_col in df.columns:
        df = df.set_index(id_col)
    df = df.select_dtypes(include=[np.number])
    df = df.replace([np.inf, -np.inf], np.nan)

    initial_rows = len(df)
    df = df.dropna(how='all')
    if len(df) < initial_rows:
        print(f"Dropped {initial_rows - len(df)} rows that were entirely empty.")

    thresh = len(df) * 0.9
    df = df.dropna(axis=1, thresh=thresh)

    std = df.std()
    mean_abs = df.mean().abs()
    # CV is undefined (or explodes) for naturally zero-centered features like glcm_ClusterShade or
    # firstorder_Skewness, where a near-zero mean isn't a sign of low variance. For those, fall back
    # to a tiny absolute floor that only catches genuinely-constant columns; the CV check otherwise
    # already requires std > 0, so a feature can't sneak through here by both being flat AND centered.
    cv = std / mean_abs.replace(0, np.nan)
    keep = (std > 1e-8) & ((cv > cv_threshold) | cv.isna())
    df = df.loc[:, keep]

    print(f"Cleanup finished. Columns remaining: {df.shape[1]}")
    return df


# Calculates the ICC for each feature across the two readers, keeping only those above threshold.
# df: cleaned-up X_dev from "global_cleanup"
def interobserver_stability(df, path_r1, path_r2, threshold, id_col='PatientID'):
    df_r1 = pd.read_csv(path_r1)
    df_r2 = pd.read_csv(path_r2)

    common_ids = np.intersect1d(df_r1[id_col], df_r2[id_col])
    r1 = df_r1.set_index(id_col).loc[common_ids].sort_index()
    r2 = df_r2.set_index(id_col).loc[common_ids].sort_index()

    features = [f for f in df.columns
                if f in r1.columns and f in r2.columns and r1[f].dtype != 'object']
    r1, r2 = r1[features], r2[features]
    n = len(common_ids)
    print(f"ICC: {len(features)} features across {n} patients (vectorized)...")

    r1_arr = r1.values.astype(float)
    r2_arr = r2.values.astype(float)
    valid   = ~(np.isnan(r1_arr).any(axis=0) | np.isnan(r2_arr).any(axis=0))
    r1_v, r2_v = r1_arr[:, valid], r2_arr[:, valid]
    valid_feats = [f for f, v in zip(features, valid) if v]

    gm     = (r1_v.mean(0) + r2_v.mean(0)) / 2
    sm     = (r1_v + r2_v) / 2
    SS_b   = 2 * ((sm - gm) ** 2).sum(0)
    SS_r   = n * ((r1_v.mean(0) - gm) ** 2 + (r2_v.mean(0) - gm) ** 2)
    SS_tot = ((r1_v - gm) ** 2).sum(0) + ((r2_v - gm) ** 2).sum(0)
    MS_b   = SS_b / (n - 1)
    MS_e   = (SS_tot - SS_b - SS_r) / (n - 1)
    denom  = MS_b + MS_e
    icc3   = np.where(denom > 0, (MS_b - MS_e) / denom, 0.0)

    stable = [f for f, c in zip(valid_feats, icc3) if c >= threshold]
    print(f"ICC ≥ {threshold}: {len(stable)} / {len(features)} features kept.")
    return df[stable]


# --- Fold-specific preprocessing and feature selection -----------------------------------------------
# The remaining feature removal and selection occurs inside each training fold to avoid data leakage:
# KNN imputation, z-scaling, univariate filtering, correlation filtering, and LASSO-based selection.
# Parameters (imputer, scaler, selector) are learned from the training fold only.

# Benjamini-Hochberg FDR correction. With thousands of features tested simultaneously, an
# uncorrected p < 0.05 cutoff lets through hundreds of features by chance alone (~5% of however many
# are tested); q-values control the expected false discovery rate among the features actually kept.
def benjamini_hochberg(pvals):
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty(n)
    out[order] = q
    return out


# Calculates univariate metrics for each feature (AUC, Mann-Whitney U p-value + FDR-adjusted q-value,
# odds ratio). X should be the training fold feature matrix.
def calculate_feature_metrics(X, y):
    metrics = []
    for col in X.columns:
        group0 = X[col][y == 0]
        group1 = X[col][y == 1]
        _, p = mannwhitneyu(group0, group1, alternative='two-sided')

        # Univariate AUC. Intentionally NOT flipped to be >= 0.5 here: run_fold_pruning's AUC filter
        # is two-sided (keeps AUC >= cutoff OR AUC <= 1 - cutoff), which depends on this being raw/unflipped.
        auc = roc_auc_score(y, X[col])

        model = LogisticRegression(solver='liblinear', random_state=RANDOM_STATE)
        model.fit(X[[col]], y)

        beta = model.coef_[0][0]
        odds_ratio = np.exp(beta)

        metrics.append({
            'Feature': col,
            'p_value': p,
            'AUC': auc,
            'Odds_Ratio': odds_ratio
        })

    metrics_df = pd.DataFrame(metrics).set_index('Feature')
    metrics_df['p_value_fdr'] = benjamini_hochberg(metrics_df['p_value'])
    return metrics_df


# Considers the spearman correlation between features and drops one of each correlated pair.
# The tie-breaker is the p-value (or AUC/OR) from above, keeping the one with the better score.
def spearman_filter(df, threshold, tie_breaker_series):
    # Vectorized greedy: process features best-tie-breaker-first, keep a feature unless it's
    # already too correlated with something already kept (a single numpy comparison against the
    # kept set, not a per-pair Python loop). This is a behavioral refinement over the original
    # pairwise-in-column-order loop, not just a speedup: the original's survivor set depended on
    # incidental column order (a feature only got compared against earlier-indexed columns first),
    # whereas this always lets the better-tie-breaker feature win regardless of column position --
    # matters once the candidate pool is in the thousands (wavelet/log-sigma restored) rather than
    # the ~100s a pure-Python O(n^2) loop was fine for.
    cols = list(df.columns)
    corr = df.corr(method='spearman').abs().values

    if tie_breaker_series is not None:
        order = sorted(range(len(cols)), key=lambda i: tie_breaker_series[cols[i]])
    else:
        order = list(range(len(cols)))

    kept_indices = []
    for idx in order:
        if kept_indices and (corr[idx, kept_indices] > threshold).any():
            continue
        kept_indices.append(idx)

    kept_cols = [cols[i] for i in sorted(kept_indices)]
    print(f"Pruning redundant features: kept {len(kept_cols)} of {len(cols)}")
    return df[kept_cols]


# Implements the fold-specific pruning steps: imputation, scaling, univariate filtering, correlation
# filtering, and LASSO-based selection. Learns all parameters from X_tr/y_tr only.
DEFAULT_L1_RATIOS = [0.1, 0.3, 0.5, 0.7, 0.9]


def run_fold_pruning(X_tr, y_tr, target_count, p_value_cutoff, metric, metric_cutoff,
                      decorrelation_threshold=0.85, penalty='l1', l1_ratios=None,
                      random_state=RANDOM_STATE):
    if l1_ratios is None:
        l1_ratios = DEFAULT_L1_RATIOS
    imputer = KNNImputer(n_neighbors=5)
    X_imp = pd.DataFrame(imputer.fit_transform(X_tr), columns=X_tr.columns, index=X_tr.index)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imp), columns=X_imp.columns, index=X_imp.index)

    scorecard = calculate_feature_metrics(X_scaled, y_tr)
    total_features = len(scorecard)

    if metric.lower() == 'auc':
        lower_limit = 1 - metric_cutoff
        mask_kept = (scorecard['AUC'] >= metric_cutoff) | (scorecard['AUC'] <= lower_limit)
        print(f"Filter: Keeping AUC >= {metric_cutoff} or <= {lower_limit:.2f}")
    elif metric.lower() == 'or':
        log_threshold = np.abs(np.log(metric_cutoff))
        mask_kept = np.abs(np.log(scorecard['Odds_Ratio'])) > log_threshold
        print(f"Filter: Keeping OR > {metric_cutoff} or < {1/metric_cutoff:.2f}")
    else:
        raise ValueError(f"Unknown univariate metric '{metric}'. Expected 'auc' or 'or'.")

    # FDR is computed only WITHIN the effect-size-screened pool, not across all total_features.
    # Applying Benjamini-Hochberg across the full multi-thousand-feature universe is needlessly
    # punishing here: most of those features are near-duplicates of each other (the same base
    # feature run through dozens of wavelet/log-sigma filters) and obviously flat, so correcting
    # against all of them buries any real signal under the correction factor (q for the single best
    # feature would need raw p ~ p_value_cutoff/total_features, an unreasonable bar). Screening on
    # effect size first and correcting only among the survivors is standard "independent filtering"
    # practice (cf. DESeq2 in RNA-seq) -- it keeps the FDR budget where it's actually useful instead
    # of diluting it across thousands of features nobody expected to be significant anyway.
    screened = scorecard[mask_kept].copy()
    screened['p_value_fdr_screened'] = benjamini_hochberg(screened['p_value'])
    mask_sig = screened['p_value_fdr_screened'] < p_value_cutoff
    kept_features_info = screened[mask_sig]

    print(f"Effect-size screen ({metric}): {len(screened)} of {total_features} pass.")
    print(f"FDR within screened pool: {mask_sig.sum()} of {len(screened)} pass q<{p_value_cutoff}.")
    print(f"Discarding {total_features - len(kept_features_info)} features.")

    if len(kept_features_info) == 0:
        raise RuntimeError(
            f"No features survived the univariate filter for this fold (0 of {total_features}). "
            f"{len(screened)} passed the effect-size filter, but none of those passed FDR "
            f"q<{p_value_cutoff} within the screened pool. Loosen --p-value-cutoff (FDR q threshold) "
            f"and/or --metric-cutoff and re-run."
        )

    X_filtered = X_scaled[kept_features_info.index]

    initial_n = X_filtered.shape[1]
    p_vals = scorecard.loc[X_filtered.columns, 'p_value']
    X_spearman = spearman_filter(X_filtered, threshold=decorrelation_threshold, tie_breaker_series=p_vals)
    kept_features = X_spearman.columns
    final_n = X_spearman.shape[1]
    print(f"Spearman: Dropped {initial_n - final_n} features (from {initial_n} to {final_n})")

    # C is cross-validated (inner CV, AUC-scored) rather than fixed, since an arbitrary, untuned
    # regularization strength is what let LASSO pick a near-random support set of features each outer
    # fold -- the single biggest driver of the fold-to-fold sign/selection instability we were seeing.
    # class_weight='balanced' added to match every other model in this file; it was previously the
    # only estimator here fit unweighted despite the outcome imbalance.
    #
    # penalty='elasticnet' is the textbook fix (Zou & Hastie) for pure L1's behavior on correlated
    # predictors: LASSO arbitrarily zeros all-but-one of a correlated group (the driver of the
    # vote-splitting that cluster-tallying addresses downstream), whereas elastic net's ridge
    # component gives correlated features SIMILAR coefficients -- the "grouping effect" -- so the
    # group survives or falls together. Note it necessarily selects MORE features per resample than
    # L1 at comparable C, by design. Requires solver='saga' (liblinear has no elasticnet support),
    # which is markedly slower and needs a higher max_iter to converge on this scaled data.
    inner_splits = min(5, y_tr.value_counts().min())
    inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=random_state)
    if penalty == 'elasticnet':
        lasso_cv = LogisticRegressionCV(
            Cs=np.logspace(-3, 1, 15), penalty='elasticnet', solver='saga', l1_ratios=l1_ratios,
            cv=inner_cv, scoring='roc_auc', class_weight='balanced', max_iter=5000,
            random_state=random_state, n_jobs=1,
        )
    else:
        lasso_cv = LogisticRegressionCV(
            Cs=np.logspace(-3, 1, 15), penalty='l1', solver='liblinear', cv=inner_cv,
            scoring='roc_auc', class_weight='balanced', max_iter=1000, random_state=random_state,
        )
    with warnings.catch_warnings():
        # saga routinely emits ConvergenceWarning on near-degenerate resamples; the resulting fit is
        # still usable (and a resample where it truly fails is caught by the caller's except block).
        warnings.simplefilter("ignore")
        lasso_cv.fit(X_spearman, y_tr)

    coefs = pd.Series(lasso_cv.coef_[0], index=X_spearman.columns)
    nonzero_coefs = coefs[coefs != 0]
    # Rank by magnitude, but keep the ORIGINAL signed coefficients in the output (fixes a bug where
    # .abs() was applied before slicing, silently discarding the direction of every selected feature).
    ranked_by_magnitude = nonzero_coefs.abs().sort_values(ascending=False)
    selected = nonzero_coefs.loc[ranked_by_magnitude.head(target_count).index]
    print(f"LASSO: Dropped {len(coefs) - len(selected)} features (from {len(coefs)} to {len(selected)})")
    return selected, imputer, scaler, kept_features, lasso_cv, scorecard


def stratified_resample(X, y, frac, rng, replace=False):
    """Draw a class-stratified resample: each outcome class resampled independently to `frac` of
    its own size, then concatenated -- guarantees exact class proportions by construction (each
    class resampled in isolation) rather than by solving a global partition-balance constraint,
    which is what a stratified K-way split has to do. Floored at 2 patients/class so a resample
    can never accidentally drop a whole class down to a single (degenerate, no-variance) member."""
    parts_X, parts_y = [], []
    for cls in y.unique():
        idx = y.index[y == cls]
        n = max(int(round(len(idx) * frac)), 2)
        n = min(n, len(idx)) if not replace else n
        drawn = rng.choice(idx, size=n, replace=replace)
        parts_X.append(X.loc[drawn])
        parts_y.append(y.loc[drawn])
    return pd.concat(parts_X), pd.concat(parts_y)


# Groups features into correlation clusters (connected components of |Spearman rho| > threshold,
# same threshold convention as spearman_filter's pairwise pruning), computed ONCE from the whole
# X passed in -- independent of any resampling. Fixes vote-splitting in stability selection: two
# correlated near-duplicate columns (e.g. an original-filter feature and a slightly-differently-
# processed cousin) can each individually "win" the per-resample decorrelation/LASSO race about
# half the time, making BOTH look unreliable when tallied separately even if the underlying
# effect they both represent is reliably picked up every single resample. Tallying at the cluster
# level instead credits the resample to whichever cluster was hit, regardless of which specific
# member won that particular resample's tie-break.
# Returns (cluster_of, orientation): cluster_of maps each feature to its cluster's root feature
# name; orientation maps each feature to +1/-1, the sign of its direct correlation with that
# root -- used to normalize LASSO coefficient signs onto a common cluster-level direction before
# checking sign stability (two anti-correlated cluster-mates pushing the SAME underlying effect
# would otherwise look sign-unstable purely because of which side of zero their raw values sit on).
# Simplification: orientation is computed against the cluster's root directly, not propagated
# transitively through the union chain -- fine at this threshold, where clusters are small and
# tightly correlated, but a long weak transitive chain could rarely misorient a member.
def compute_correlation_clusters(X, threshold=0.85):
    # Vectorized (scipy connected_components) instead of a pure-Python O(n^2) pairwise loop --
    # matters once wavelet/log-sigma are back in the candidate pool (thousands of columns instead
    # of ~100s), where a Python-level double loop over every pair would be impractically slow.
    cols = list(X.columns)
    corr = X.corr(method='spearman').values
    n = len(cols)
    adj = np.abs(corr) > threshold
    np.fill_diagonal(adj, False)
    n_components, labels = connected_components(csgraph=csr_matrix(adj), directed=False)

    root_of_label = {}
    for i, c in enumerate(cols):
        lbl = labels[i]
        if lbl not in root_of_label:
            root_of_label[lbl] = c  # first-encountered member of each cluster is its root
    cluster_of = {c: root_of_label[labels[i]] for i, c in enumerate(cols)}

    corr_df = pd.DataFrame(corr, index=cols, columns=cols)
    orientation = {}
    for c in cols:
        root = cluster_of[c]
        if c == root:
            orientation[c] = 1
        else:
            r = corr_df.loc[c, root]
            orientation[c] = -1 if (pd.notna(r) and r < 0) else 1
    return cluster_of, orientation


# Meinshausen-Buhlmann-style stability selection. Replaces the single, full-dev-set feature
# selection pass that currently produces consensus_features.csv (via run_cross_validation's
# 5-fold Selection_Frequency -- only 5 noisy trials) with many more (n_resamples) independent,
# class-stratified subsamples of the whole dev set, each re-running the full screen -> FDR ->
# decorrelation -> LASSO pipeline (run_fold_pruning, unchanged) from scratch via
# stratified_resample. Tallying happens at the correlation-CLUSTER level (see
# compute_correlation_clusters), not per individual column -- a cluster's stability frequency is
# a far less noisy, less vote-split estimate of "how reliably does this signal get picked" than a
# 5-trial per-column count, which directly targets both the winner's-curse-style optimism baked
# into a single full-dev selection pass AND the vote-splitting that per-column tallying suffers
# from when correlated near-duplicates compete for individual credit. Does NOT replace
# run_cross_validation's nested-CV OOF diagnostic -- that keeps reselecting fresh per outer fold
# exactly as before, and stays the honest reference number to compare this method's lock-set
# result against.
def run_stability_selection(X, y_dev, target_count, p_value_cutoff, metric, metric_cutoff,
                             n_resamples=200, resample_frac=0.75, decorrelation_threshold=0.85,
                             cluster_threshold=None, penalty='l1', l1_ratios=None,
                             random_state=RANDOM_STATE):
    # cluster_threshold controls ONLY the post-hoc tallying grouping (compute_correlation_clusters,
    # run once on the full X). decorrelation_threshold controls ONLY the per-resample pairwise
    # pruning inside run_fold_pruning (what the penalized model sees). These are DELIBERATELY
    # independent -- e.g. testing "let elastic net see correlated features directly" means setting
    # decorrelation_threshold near/at 1.0 while explicitly KEEPING cluster_threshold at its normal
    # value (0.85), so stability is still tallied at the correlation-cluster level even though the
    # model itself no longer has features pre-pruned. Conflating the two (as an earlier version of
    # this function did, by reusing decorrelation_threshold for both) silently disables clustering
    # itself whenever decorrelation is relaxed -- exactly the wrong thing when testing that scenario.
    if cluster_threshold is None:
        cluster_threshold = decorrelation_threshold
    rng = np.random.default_rng(random_state)
    cluster_of, orientation = compute_correlation_clusters(X, threshold=cluster_threshold)
    cluster_members = {}
    for feat, c in cluster_of.items():
        cluster_members.setdefault(c, []).append(feat)

    cluster_selection_counts = pd.Series(0, index=sorted(cluster_members), dtype=int)
    cluster_sign_records = {c: [] for c in cluster_members}
    member_selection_counts = pd.Series(0, index=X.columns, dtype=int)
    n_valid = 0

    for _ in tqdm(range(n_resamples), desc="Stability selection resamples"):
        X_rs, y_rs = stratified_resample(X, y_dev, resample_frac, rng, replace=False)
        try:
            selected_coefs, *_ = run_fold_pruning(
                X_rs, y_rs, target_count, p_value_cutoff, metric, metric_cutoff=metric_cutoff,
                decorrelation_threshold=decorrelation_threshold, penalty=penalty,
                l1_ratios=l1_ratios, random_state=random_state,
            )
        except RuntimeError:
            # This resample's screen found nothing -- a valid outcome (a "not selected" vote for
            # every cluster), not a crash worth stopping the whole procedure over.
            continue
        n_valid += 1
        nonzero = selected_coefs[selected_coefs != 0]
        member_selection_counts.loc[nonzero.index] += 1
        # A cluster counts as "selected" at most once per resample, even on the rare occasion
        # more than one of its members survives that resample's own pairwise decorrelation step.
        hit_clusters = set()
        for feat, coef in nonzero.items():
            c = cluster_of[feat]
            if c not in hit_clusters:
                cluster_selection_counts[c] += 1
                hit_clusters.add(c)
            cluster_sign_records[c].append(np.sign(coef) * orientation[feat])

    if n_valid == 0:
        raise RuntimeError(
            "Stability selection: every resample failed the univariate screen -- no signal "
            "survives even under resampling. Loosen --metric-cutoff/--p-value-cutoff and re-run."
        )

    def _sign_stable(c):
        signs = cluster_sign_records[c]
        return bool(signs) and (all(s > 0 for s in signs) or all(s < 0 for s in signs))

    rows = []
    for c, members in cluster_members.items():
        if cluster_selection_counts[c] == 0:
            continue
        # Representative: whichever member was individually selected most often across resamples
        # -- the concrete column that actually goes into the model for this cluster.
        representative = member_selection_counts.loc[members].idxmax()
        rows.append({
            'Cluster': c,
            'Representative_Feature': representative,
            'Cluster_Members': ';'.join(members),
            'N_Members': len(members),
            'Selection_Frequency': cluster_selection_counts[c] / n_valid,
            'Selection_Count': int(cluster_selection_counts[c]),
            'N_Resamples_Valid': n_valid,
            'Sign_Stable': _sign_stable(c),
        })
    stability_df = pd.DataFrame(rows).set_index('Cluster')
    stability_df = stability_df.sort_values('Selection_Frequency', ascending=False)
    return stability_df


# --- Cross-validation wrapper --------------------------------------------------------------------------
# Runs the main K-fold loop, applying fold-specific pruning to identify consensus features. Also tunes
# (via tune_model, same PARAM_GRIDS as the real gridsearch stage) and fits NESTED_CV_MODELS on each
# fold's own fold-specific feature selection, predicting on the held-out fold to produce out-of-fold
# predictions pooled across all of X. Unlike the lock-set evaluation (which judges models tuned/fit on
# the *final* consensus_features and hyperparameters, both chosen by looking at the whole dev set),
# this reruns selection AND tuning per fold using only that fold's train data -- a true nested CV,
# giving an honest, leakage-free read on how the whole pipeline generalizes, using all of X_dev instead
# of just the 30-patient lock set.
NESTED_CV_MODELS = ALL_MODELS


def run_cross_validation(X, y_dev, target_count, p_value_cutoff, metric, metric_cutoff,
                          model_names=None, tuning_cv=5, decorrelation_threshold=0.85,
                          penalty='l1', l1_ratios=None, random_state=RANDOM_STATE):
    if model_names is None:
        model_names = NESTED_CV_MODELS

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    fold_coefficients = pd.DataFrame(0.0, index=X.columns, columns=[f'Fold_{i+1}' for i in range(5)])
    oof_predictions = pd.DataFrame(np.nan, index=X.index, columns=model_names)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_dev)):
        X_tr, y_tr = X.iloc[train_idx], y_dev.iloc[train_idx]
        X_val = X.iloc[val_idx]

        try:
            selected_coefs, imputer, scaler, kept_features, lasso, scorecard = run_fold_pruning(
                X_tr, y_tr, target_count, p_value_cutoff, metric, metric_cutoff=metric_cutoff,
                decorrelation_threshold=decorrelation_threshold, penalty=penalty,
                l1_ratios=l1_ratios, random_state=random_state,
            )
        except RuntimeError as exc:
            # This fold's own screen found 0 candidate features -- a valid outcome for a small
            # candidate pool (e.g. a single-sequence combo), not a crash worth losing the whole
            # OOF diagnostic over. Contributes no predictions/coefficients for this fold; the
            # other folds still pool into an honest (if noisier) OOF estimate. Matches how
            # run_stability_selection already treats the same exception per resample.
            print(f"Fold {fold+1}: 0 candidate features passed the screen ({exc}) -- skipping this fold.")
            continue

        fold_coefficients.loc[selected_coefs.index, f'Fold_{fold+1}'] = selected_coefs.values

        active_count = (selected_coefs != 0).sum()
        print(f"Fold {fold+1} complete. Features selected by LASSO: {active_count}")

        fold_features = selected_coefs.index
        if len(fold_features) == 0:
            continue

        X_tr_scaled = pd.DataFrame(scaler.transform(imputer.transform(X_tr)), columns=X_tr.columns, index=X_tr.index)
        X_val_scaled = pd.DataFrame(scaler.transform(imputer.transform(X_val)), columns=X_val.columns, index=X_val.index)
        X_tr_final = X_tr_scaled[fold_features]
        X_val_final = X_val_scaled[fold_features]

        for model_name in model_names:
            model = tune_model(X_tr_final, y_tr, model_name, cv=tuning_cv, random_state=random_state)
            oof_predictions.loc[X_val.index, model_name] = model.predict_proba(X_val_final)[:, 1]

    fold_cols = [f'Fold_{i+1}' for i in range(5)]
    fold_coefficients['Selection_Frequency'] = (fold_coefficients[fold_cols] != 0).sum(axis=1)
    fold_coefficients['Mean_Absolute_Coefficient'] = fold_coefficients[fold_cols].abs().mean(axis=1)

    # A feature whose LASSO coefficient flips sign across folds isn't a stable effect -- it's noise
    # that happened to correlate with the outcome in one direction in some folds and the other
    # direction in others. Only require agreement among the folds where it was actually selected
    # (zero entries are "not selected that fold", not "selected with coefficient 0").
    def _sign_stable(row):
        nonzero = row[fold_cols][row[fold_cols] != 0]
        if len(nonzero) == 0:
            return False
        return bool((nonzero > 0).all() or (nonzero < 0).all())
    fold_coefficients['Sign_Stable'] = fold_coefficients.apply(_sign_stable, axis=1)

    fold_coefficients = fold_coefficients[fold_coefficients['Selection_Frequency'] > 0]
    fold_coefficients = fold_coefficients.sort_values(by=['Selection_Frequency', 'Mean_Absolute_Coefficient'], ascending=False)

    # Clean global scorecard from the full dev set, for downstream reporting only (not used to filter
    # consensus_features, which is driven entirely by the per-fold selection above).
    dev_imputer = KNNImputer(n_neighbors=5)
    X_dev_imp = pd.DataFrame(dev_imputer.fit_transform(X), columns=X.columns, index=X.index)
    dev_scaler = StandardScaler()
    X_dev_scaled = pd.DataFrame(dev_scaler.fit_transform(X_dev_imp), columns=X.columns, index=X.index)
    global_scorecard = calculate_feature_metrics(X_dev_scaled, y_dev)

    return fold_coefficients, global_scorecard, oof_predictions


# Summarizes the out-of-fold predictions from run_cross_validation into a per-model honest AUC
# (with bootstrap CI, using the same bootstrap_ci machinery as the lock-set table so the two are
# directly comparable) plus the Youden's-J-optimal decision threshold for that model.
def summarize_nested_cv(oof_predictions, y_dev, n_bootstrap=1000, random_state=RANDOM_STATE):
    rows = []
    for model_name in oof_predictions.columns:
        proba = oof_predictions[model_name]
        valid_mask = proba.notna()
        y_valid = y_dev[valid_mask]
        proba_valid = proba[valid_mask]

        auc = roc_auc_score(y_valid, proba_valid)
        auc_ci = bootstrap_ci(y_valid, proba_valid, roc_auc_score, n_bootstrap=n_bootstrap, seed=random_state)

        fpr, tpr, thresholds = roc_curve(y_valid, proba_valid)
        youden_threshold = thresholds[np.argmax(tpr - fpr)]

        rows.append({
            'model': model_name,
            'oof_auc': auc,
            'auc_ci_lower': auc_ci['ci_lower'],
            'auc_ci_upper': auc_ci['ci_upper'],
            'youden_threshold': youden_threshold,
            'n_dev': int(valid_mask.sum()),
        })

    return pd.DataFrame(rows)


# Prints the heatmap + side-by-side univariate metric table comparing dev vs. lock for the consensus
# features. Diagnostic only — does not feed back into modeling.
def run_feature_diagnostics(X_dev_clean_icc, y_dev_filtered, X_lock, y_lock, consensus_features, output_dir):
    print("\n" + "=" * 85)
    print("RUNNING FEATURE DIAGNOSTICS (DEV vs. LOCK)")
    print("=" * 85)

    X_dev_diag = X_dev_clean_icc[consensus_features]
    plt.figure(figsize=(8, 6))
    corr_matrix = X_dev_diag.corr(method='spearman')
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1, square=True)
    plt.title(f"Spearman Correlation (Dev Set): Top {len(consensus_features)} Features")
    plt.tight_layout()
    heatmap_path = f"{output_dir}/consensus_features_correlation.png"
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    print(f"-> Saved correlation heatmap to: {heatmap_path}")

    X_lock_numeric = X_lock.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)

    print(f"\n{'Feature Name':<40} | {'Dev MW p':<10} | {'Dev AUC':<8} | {'Lock MW p':<10} | {'Lock AUC':<8}")
    print("-" * 85)

    for feat in consensus_features:
        try:
            dev_data = pd.concat([X_dev_diag[feat], y_dev_filtered], axis=1).dropna()
            dev_feat_clean = dev_data[feat]
            dev_y_clean = dev_data[y_dev_filtered.name]
            d0 = dev_feat_clean[dev_y_clean == 0]
            d1 = dev_feat_clean[dev_y_clean == 1]
            _, p_dev = mannwhitneyu(d0, d1, alternative='two-sided')
            # AUC intentionally unflipped here too, to match calculate_feature_metrics' convention
            # used in univariate_metrics.csv -- previously these two tables disagreed for AUC < 0.5.
            auc_dev = roc_auc_score(dev_y_clean, dev_feat_clean)

            lock_data = pd.concat([X_lock_numeric[feat], y_lock], axis=1).dropna()
            lock_feat_clean = lock_data[feat]
            lock_y_clean = lock_data[y_lock.name]
            l0 = lock_feat_clean[lock_y_clean == 0]
            l1 = lock_feat_clean[lock_y_clean == 1]
            _, p_lock = mannwhitneyu(l0, l1, alternative='two-sided')
            auc_lock = roc_auc_score(lock_y_clean, lock_feat_clean)

            print(f"{feat:<40} | {p_dev:<10.3f} | {auc_dev:<8.3f} | {p_lock:<10.3f} | {auc_lock:<8.3f}")
        except Exception as e:
            print(f"{feat:<40} | Error: {e}")
    print("=" * 85)


# =====================================================================================================
# Shared model factory (single definition used by every stage)
# =====================================================================================================

# scale_pos_weight is left None by default so XGBoost falls back to its own default (1) when the
# caller hasn't computed a class ratio; both the gridsearch and train stages compute and pass it.
def get_model(model_name, random_state=RANDOM_STATE, class_weight='balanced', scale_pos_weight=None):
    if model_name == 'logistic':
        return LogisticRegression(penalty='l2', C=0.01, solver='lbfgs', max_iter=1000,
                                   class_weight=class_weight, random_state=random_state)
    elif model_name == 'svm_linear':
        return SVC(kernel='linear', probability=True, class_weight=class_weight, random_state=random_state)
    elif model_name == 'svm_rbf':
        return SVC(kernel='rbf', C=10, gamma=0.01, probability=True,
                   class_weight=class_weight, random_state=random_state)
    elif model_name == 'rf':
        # n_estimators fixed (not in PARAM_GRIDS): RF test AUC is ~flat past a couple hundred trees
        # on this dev set (500 vs 200 trees differed by <0.003 AUC, within noise) -- tuning it just
        # burns grid combinations for no real gain. max_depth/min_samples_leaf are what actually
        # control overfitting, so the grid search budget goes there instead.
        return RandomForestClassifier(n_estimators=300, max_depth=3,
                                       class_weight=class_weight, random_state=random_state)
    elif model_name == 'extratrees':
        # Same rationale as rf for fixing n_estimators. Extremely randomized splits give a different
        # bias/variance trade-off than rf, useful as a second tree-ensemble opinion.
        return ExtraTreesClassifier(n_estimators=300, max_depth=3,
                                     class_weight=class_weight, random_state=random_state)
    elif model_name == 'xgboost':
        # Low depth to prevent overfitting on a small N. n_jobs=1: XGBoost defaults to all cores
        # per fit, which oversubscribes when GridSearchCV also parallelizes across fits.
        return XGBClassifier(n_estimators=100, max_depth=2, learning_rate=0.01, subsample=0.6,
                              eval_metric='logloss', scale_pos_weight=scale_pos_weight, random_state=random_state,
                              n_jobs=1)
    elif model_name == 'gp':
        # No PARAM_GRIDS entry by design: the kernel's length scale is fit internally via marginal
        # likelihood maximization at .fit() time, so external grid search would be redundant. Also no
        # class_weight/sample_weight support in sklearn's GaussianProcessClassifier -- left unweighted.
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
        return GaussianProcessClassifier(kernel=kernel, random_state=random_state)
    elif model_name == 'tabpfn':
        # torch defaults to one thread per physical core for its internal ops; combined with the
        # native thread pools sklearn/scipy/xgboost already spawn in this same process, that caused
        # multi-minute hangs in practice. Capping torch to 1 thread fixed it outright (fit went from
        # hanging indefinitely to <1s) -- this dataset is far too small for that parallelism to matter.
        torch.set_num_threads(1)
        return TabPFNClassifier(device='cpu', random_state=random_state)
    else:
        raise ValueError(f"Model {model_name} not recognized.")


def _positive_class_weight(y):
    """Ratio of negatives to positives, for XGBoost's scale_pos_weight."""
    n_pos = max((y == 1).sum(), 1)
    return (y == 0).sum() / n_pos


# Single source of truth for hyperparameter search spaces, shared by perform_grid_search (tunes once
# on the full dev set) and tune_model (tunes inside each nested-CV outer fold) -- so the nested CV
# honest-estimate reflects the SAME tuning process the real pipeline uses, not untuned defaults.
# Models absent here (gp, tabpfn) are intentionally untuned -- see their get_model() comments.
#
# Ranges below were trimmed against actual cv_results_<model>.csv curves from this dev set, not
# guessed: svm_linear/logistic plateaued at C<=0.001 (svm_linear's two lowest C ranks were bit-for-bit
# identical -- the decision boundary direction had stopped changing), and svm_rbf's C<=0.1/gamma=1
# region was uniformly dominated (ranked >=13 of 36). rf/extratrees n_estimators is fixed in
# get_model() instead of grid-searched for the same reason (500 vs 200 trees differed by <0.003 AUC).

PARAM_GRIDS = {
    'svm_linear': {'C': [0.001, 0.01, 0.1, 1, 10]},
    'svm_rbf': {'C': [1, 10, 100, 1000], 'gamma': ['scale', 'auto', 0.01]},
    'rf': {'max_depth': [2, 3, 4, 5, 6], 'min_samples_leaf': [1, 2, 5, 10]},
    'extratrees': {'max_depth': [2, 3, 4, 5, 6], 'min_samples_leaf': [1, 2, 5, 10]},
    'xgboost': {'n_estimators': [25, 50, 100, 200, 300], 'max_depth': [1, 2, 3],
                'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3], 'subsample': [0.6, 0.8, 1.0]},
    'logistic': {'C': [0.001, 0.01, 0.1, 1, 10], 'penalty': ['l2'], 'solver': ['lbfgs']},
}


# Tunes model_name via GridSearchCV over PARAM_GRIDS and returns the best fitted estimator. Models
# without an entry in PARAM_GRIDS (e.g. 'tabpfn') just fit get_model()'s plain defaults on X/y.
# cv is clamped to the smaller class's count so this stays safe on small nested-CV fold sizes, where
# a fixed cv=5 could exceed the number of available minority-class samples.
def tune_model(X, y, model_name, cv=5, scoring='roc_auc', random_state=RANDOM_STATE, n_jobs=6):
    pos_weight = _positive_class_weight(y)
    estimator = get_model(model_name, random_state=random_state, class_weight='balanced',
                           scale_pos_weight=pos_weight)
    if model_name not in PARAM_GRIDS:
        estimator.fit(X, y)
        return estimator

    safe_cv = min(cv, y.value_counts().min())
    cv_splitter = StratifiedKFold(n_splits=safe_cv, shuffle=True, random_state=random_state)
    gs = GridSearchCV(estimator=estimator, param_grid=PARAM_GRIDS[model_name], scoring=scoring,
                       cv=cv_splitter, n_jobs=n_jobs, refit=True)
    gs.fit(X, y)
    return gs.best_estimator_


# =====================================================================================================
# STAGE 2: Hyperparameter search
# =====================================================================================================

# Loads the artifacts saved by the partition stage. include_lock=False is used by the gridsearch
# stage, which must never touch X_lock/y_lock.
def load_preprocessed_data(in_dir, include_lock=True):
    print(f"\nLoading preprocessed data from: {in_dir}")

    X_dev = pd.read_csv(f"{in_dir}/X_dev_processed.csv", index_col=0)
    y_dev = pd.read_csv(f"{in_dir}/y_dev.csv", index_col=0).squeeze()
    print(f"Loaded X_dev: {X_dev.shape}")
    print(f"Loaded y_dev: {len(y_dev)} samples")

    if not include_lock:
        return X_dev, y_dev

    X_lock = pd.read_csv(f"{in_dir}/X_lock_processed.csv", index_col=0)
    y_lock = pd.read_csv(f"{in_dir}/y_lock.csv", index_col=0).squeeze()
    print(f"Loaded X_lock: {X_lock.shape}")
    print(f"Loaded y_lock: {len(y_lock)} samples")
    return X_dev, X_lock, y_dev, y_lock


def load_calibrated_thresholds(in_dir):
    """Load the per-model Youden's-J thresholds written by the partition stage's nested CV
    diagnostics, if present. Returns {} for older artifacts that predate this file."""
    path = os.path.join(in_dir, 'calibrated_thresholds.csv')
    if not os.path.exists(path):
        return {}
    thresholds = pd.read_csv(path)
    return dict(zip(thresholds['model'], thresholds['youden_threshold']))


def load_saved_estimator(model_name, input_dir):
    """Load a pickled best estimator for the given model from the input directory, if it exists."""
    if input_dir is None:
        return None

    estimator_path = os.path.join(input_dir, f"best_estimator_{model_name}.pkl")
    if os.path.exists(estimator_path):
        print(f"Loading saved estimator for {model_name} from: {estimator_path}")
        with open(estimator_path, 'rb') as f:
            return pickle.load(f)
    return None


# Run GridSearchCV for a set of models and save best params/estimators. The base estimator for each
# model comes from get_model(), so class_weight/scale_pos_weight policy stays identical between the
# gridsearch stage and the plain fallback path used in the train stage.
def perform_grid_search(X, y, model_names=None, out_dir=None, cv=5, scoring=None, n_jobs=6,
                         random_state=RANDOM_STATE, save=True):
    if model_names is None:
        model_names = ALL_MODELS
    if scoring is None:
        scoring = 'roc_auc'

    os.makedirs(out_dir, exist_ok=True)
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    pos_weight = _positive_class_weight(y)

    results = []
    for m in model_names:
        if m not in PARAM_GRIDS:
            print(f"Skipping unknown/untuned model: {m}")
            continue

        print(f"\nRunning grid search for: {m}")
        estimator = get_model(m, random_state=random_state, class_weight='balanced', scale_pos_weight=pos_weight)

        gs = GridSearchCV(estimator=estimator, param_grid=PARAM_GRIDS[m], scoring=scoring,
                           cv=cv_splitter, n_jobs=n_jobs, refit=True)
        gs.fit(X, y)
        print(f"Best {scoring} for {m}: {gs.best_score_:.4f} with params: {gs.best_params_}")

        results.append({'model': m, 'best_score': gs.best_score_, 'best_params': json.dumps(gs.best_params_)})

        if save:
            with open(os.path.join(out_dir, f"best_estimator_{m}.pkl"), 'wb') as f:
                pickle.dump(gs.best_estimator_, f)

            # Full param->score curve, not just the argmax, so edge-of-grid best_params can be told
            # apart from a genuine plateau (more tuning won't help) vs a still-rising curve (widen further).
            cv_results_df = pd.DataFrame(gs.cv_results_).sort_values('rank_test_score')
            cv_results_df.to_csv(os.path.join(out_dir, f"cv_results_{m}.csv"), index=False)

    results_df = pd.DataFrame(results)
    if save:
        csvp = os.path.join(out_dir, 'grid_search_results.csv')
        results_df.to_csv(csvp, index=False)
        print(f"Saved grid search summary to: {csvp}")

    return results_df


# =====================================================================================================
# STAGE 3: Bootstrap-validated evaluation on the lock set
# =====================================================================================================

# Compute bootstrap confidence intervals for a binary classification metric.
def bootstrap_ci(y_true, y_pred_proba, metric_fn, n_bootstrap=1000, ci=95, seed=RANDOM_STATE):
    # A local Generator instead of np.random.seed(): the old version mutated global numpy random
    # state as a side effect, which could change behavior depending on call order elsewhere.
    rng = np.random.default_rng(seed)

    y_true_arr = y_true.values if hasattr(y_true, 'values') else np.asarray(y_true)
    y_pred_arr = y_pred_proba.values if hasattr(y_pred_proba, 'values') else np.asarray(y_pred_proba)
    n_samples = len(y_true_arr)

    bootstrap_scores = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, n_samples, size=n_samples)
        y_true_boot = y_true_arr[indices]
        y_pred_boot = y_pred_arr[indices]

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                score = metric_fn(y_true_boot, y_pred_boot)
                if not np.isnan(score):
                    bootstrap_scores.append(score)
        except Exception:
            continue

    bootstrap_scores = np.array(bootstrap_scores)
    n_valid = len(bootstrap_scores)

    if n_valid > 0:
        point_estimate = metric_fn(y_true_arr, y_pred_arr)

        if n_valid < n_bootstrap * 0.5:
            print(f"    Warning: Only {n_valid}/{n_bootstrap} bootstrap samples were valid.")
            print("      (Small test set may have bootstrap samples with only one class)")

        alpha = (100 - ci) / 2
        ci_lower = np.percentile(bootstrap_scores, alpha)
        ci_upper = np.percentile(bootstrap_scores, 100 - alpha)
    else:
        point_estimate = np.nan
        ci_lower = np.nan
        ci_upper = np.nan

    return {
        'point_estimate': point_estimate,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'bootstrap_scores': bootstrap_scores,
        'n_valid': n_valid
    }


# Compute sensitivity/specificity for binary classification at a fixed decision threshold.
# Degenerate resamples (no positives, or no negatives) return NaN -- NOT 0 -- so bootstrap_ci's
# existing NaN filter actually excludes them instead of silently injecting a biased zero.
def compute_sensitivity(y_true, y_pred_proba, threshold=TARGET_THRESHOLD):
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_pred_proba) >= threshold).astype(int)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp / (tp + fn) if (tp + fn) > 0 else np.nan


def compute_specificity(y_true, y_pred_proba, threshold=TARGET_THRESHOLD):
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_pred_proba) >= threshold).astype(int)
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tn / (tn + fp) if (tn + fp) > 0 else np.nan


def create_shap_explainer(model, X, model_name=None):
    model_type = (model_name or type(model).__name__).lower()
    if any(keyword in model_type for keyword in
           ['xgb', 'xgboost', 'booster', 'rf', 'randomforest', 'extratrees', 'extra_trees']):
        return shap.TreeExplainer(model)
    if 'logistic' in model_type or ('linear' in model_type and hasattr(model, 'coef_')):
        return shap.LinearExplainer(model, X.values if hasattr(X, 'values') else X)
    if 'svm' in model_type or 'svc' in model_type:
        if hasattr(model, 'coef_') and getattr(model, 'kernel', None) == 'linear':
            return shap.LinearExplainer(model, X.values if hasattr(X, 'values') else X)
        bg_sample = shap.sample(X, min(50, len(X)))
        bg_data = bg_sample.values if hasattr(bg_sample, 'values') else bg_sample
        return shap.KernelExplainer(model.predict_proba, bg_data)
    if any(keyword in model_type for keyword in ['gp', 'gaussianprocess', 'tabpfn']):
        # Neither has a SHAP-native explainer; KernelExplainer works against any predict_proba.
        bg_sample = shap.sample(X, min(50, len(X)))
        bg_data = bg_sample.values if hasattr(bg_sample, 'values') else bg_sample
        return shap.KernelExplainer(model.predict_proba, bg_data)
    return shap.Explainer(model, X)


def plot_shap_summary(model, X, out_dir, model_name, plot_type='bar'):
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    explainer = create_shap_explainer(model, X, model_name=model_name)
    shap_values = explainer(X)

    if isinstance(X, np.ndarray):
        X_plot = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    else:
        X_plot = X

    fig = plt.figure(figsize=(14, 10))
    shap.summary_plot(shap_values, X_plot, plot_type=plot_type, show=False)
    ax = plt.gca()
    ax.set_xlabel("SHAP value", fontsize=12, labelpad=12)

    labels = [item.get_text() for item in ax.get_yticklabels()]
    wrapped_labels = []
    for label in labels:
        if "_3D_" in label:
            wrapped_labels.append(label.replace("_3D_", "_\n3D_"))
        elif "wavelet-" in label:
            wrapped_labels.append(label.replace("wavelet-", "wavelet-\n"))
        elif "original_" in label:
            wrapped_labels.append(label.replace("original_", "original_\n"))
        else:
            wrapped_labels.append('\n'.join(textwrap.wrap(label, width=28)))

    ax.set_yticklabels(wrapped_labels, fontsize=10, ha='right', va='center')
    plt.subplots_adjust(left=0.45, right=0.95, top=0.95, bottom=0.15)

    filename = f'{model_name}_shap_summary_{plot_type}.png'
    plot_path = os.path.join(out_dir, filename)
    plt.savefig(plot_path, dpi=300)
    plt.close(fig)
    return plot_path


def create_roc_comparison_plot(results_list, out_dir, filename='model_roc_comparison.png'):
    plt.figure(figsize=(10, 8))
    sns.set_style('whitegrid')

    plotted = 0
    for result in results_list:
        if result.get('status') != 'success':
            continue
        fpr = result.get('fpr')
        tpr = result.get('tpr')
        auc = result.get('lock_auc')
        model_name = result.get('model_name', 'model').upper()

        if fpr is None or tpr is None or np.isnan(auc):
            continue

        plt.plot(fpr, tpr, linewidth=2.5, label=f"{model_name} (AUC={auc:.3f})")
        plotted += 1

    if plotted == 0:
        print("No valid ROC curves to plot")
        return

    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1.5, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Comparison of Models on Lock Set', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.tight_layout()

    plot_path = f"{out_dir}/{filename}"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"ROC comparison plot saved to: {plot_path}")
    return plot_path


# Train (or load a tuned estimator for) a single model and return comprehensive lock-set metrics
# with bootstrap CIs, all computed at the SAME explicit threshold.
def train_and_evaluate_model(X_dev, y_dev, X_lock, y_lock, model_name, out_dir, param_dir,
                              random_state=RANDOM_STATE, n_bootstrap=1000, threshold=TARGET_THRESHOLD,
                              youden_threshold=None, generate_shap=False):
    print(f"\n{'='*60}")
    print(f"Training: {model_name.upper()}")
    print(f"{'='*60}")

    try:
        model = load_saved_estimator(model_name, param_dir)
        if model is not None:
            model_source = 'gridsearch_pickle'
            print(f"Using pickled model for {model_name} from param_dir.")
        else:
            # Recorded explicitly in the results so the comparison table shows, per model, whether it
            # was the grid-search-tuned/class-weighted estimator or this hardcoded fallback -- previously
            # this distinction was invisible, and the two paths could differ wildly in class weighting.
            model_source = 'default_fallback'
            model = get_model(model_name, random_state=random_state, class_weight='balanced',
                               scale_pos_weight=_positive_class_weight(y_dev))
            model.fit(X_dev, y_dev)

        y_pred_proba = model.predict_proba(X_lock)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)

        auc = roc_auc_score(y_lock, y_pred_proba)
        fpr, tpr, _ = roc_curve(y_lock, y_pred_proba)

        sensitivity = compute_sensitivity(y_lock, y_pred_proba, threshold=threshold)
        specificity = compute_specificity(y_lock, y_pred_proba, threshold=threshold)

        print(f"  Computing bootstrap CIs ({n_bootstrap} iterations)...")
        auc_ci = bootstrap_ci(y_lock, y_pred_proba, roc_auc_score, n_bootstrap=n_bootstrap, seed=random_state)
        # Lambda wrappers force sensitivity/specificity bootstrap iterations to use the exact same
        # threshold as the point estimates above -- point estimate and CI must never silently diverge.
        sens_ci = bootstrap_ci(
            y_lock, y_pred_proba,
            lambda y_true, y_prob: compute_sensitivity(y_true, y_prob, threshold=threshold),
            n_bootstrap=n_bootstrap, seed=random_state,
        )
        spec_ci = bootstrap_ci(
            y_lock, y_pred_proba,
            lambda y_true, y_prob: compute_specificity(y_true, y_prob, threshold=threshold),
            n_bootstrap=n_bootstrap, seed=random_state,
        )

        # Optional second read at this model's own Youden's-J-optimal threshold (from dev-only
        # nested CV, see summarize_nested_cv), since a single fixed threshold across differently
        # calibrated models can make sens/spec look nonsensical for some of them.
        youden_sensitivity = youden_specificity = np.nan
        youden_sens_ci = youden_spec_ci = {'ci_lower': np.nan, 'ci_upper': np.nan}
        if youden_threshold is not None:
            youden_sensitivity = compute_sensitivity(y_lock, y_pred_proba, threshold=youden_threshold)
            youden_specificity = compute_specificity(y_lock, y_pred_proba, threshold=youden_threshold)
            youden_sens_ci = bootstrap_ci(
                y_lock, y_pred_proba,
                lambda y_true, y_prob: compute_sensitivity(y_true, y_prob, threshold=youden_threshold),
                n_bootstrap=n_bootstrap, seed=random_state,
            )
            youden_spec_ci = bootstrap_ci(
                y_lock, y_pred_proba,
                lambda y_true, y_prob: compute_specificity(y_true, y_prob, threshold=youden_threshold),
                n_bootstrap=n_bootstrap, seed=random_state,
            )

        def format_metric(val, ci_lower, ci_upper):
            if np.isnan(val) or np.isnan(ci_lower) or np.isnan(ci_upper):
                return f"{val:.2f}", "(95% CI: N/A)"
            return f"{val:.2f}", f"(95% CI: {ci_lower:.2f} - {ci_upper:.2f})"

        auc_str, auc_ci_str = format_metric(auc, auc_ci['ci_lower'], auc_ci['ci_upper'])
        sens_str, sens_ci_str = format_metric(sensitivity, sens_ci['ci_lower'], sens_ci['ci_upper'])
        spec_str, spec_ci_str = format_metric(specificity, spec_ci['ci_lower'], spec_ci['ci_upper'])

        results = {
            'model_name': model_name,
            'model': model,
            'model_source': model_source,
            'lock_auc': auc,
            'auc_ci_lower': auc_ci['ci_lower'],
            'auc_ci_upper': auc_ci['ci_upper'],
            'sensitivity': sensitivity,
            'sens_ci_lower': sens_ci['ci_lower'],
            'sens_ci_upper': sens_ci['ci_upper'],
            'specificity': specificity,
            'spec_ci_lower': spec_ci['ci_lower'],
            'spec_ci_upper': spec_ci['ci_upper'],
            'youden_threshold': youden_threshold,
            'youden_sensitivity': youden_sensitivity,
            'youden_sens_ci_lower': youden_sens_ci['ci_lower'],
            'youden_sens_ci_upper': youden_sens_ci['ci_upper'],
            'youden_specificity': youden_specificity,
            'youden_spec_ci_lower': youden_spec_ci['ci_lower'],
            'youden_spec_ci_upper': youden_spec_ci['ci_upper'],
            'y_pred_proba': y_pred_proba,
            'y_pred': y_pred,
            'fpr': fpr,
            'tpr': tpr,
            'n_dev_samples': len(X_dev),
            'n_lock_samples': len(X_lock),
            'status': 'success'
        }

        print(f"Lock Set AUC:       {auc_str} {auc_ci_str}")
        print(f"Sensitivity:        {sens_str} {sens_ci_str}  [threshold={threshold}]")
        print(f"Specificity:        {spec_str} {spec_ci_str}  [threshold={threshold}]")
        if youden_threshold is not None:
            youden_sens_str, youden_sens_ci_str = format_metric(
                youden_sensitivity, youden_sens_ci['ci_lower'], youden_sens_ci['ci_upper'])
            youden_spec_str, youden_spec_ci_str = format_metric(
                youden_specificity, youden_spec_ci['ci_lower'], youden_spec_ci['ci_upper'])
            print(f"Sensitivity:        {youden_sens_str} {youden_sens_ci_str}  [threshold={youden_threshold:.3f} (Youden)]")
            print(f"Specificity:        {youden_spec_str} {youden_spec_ci_str}  [threshold={youden_threshold:.3f} (Youden)]")
        print(f"  Dev Samples:        {len(X_dev)}")
        print(f"  Lock Samples:       {len(X_lock)}")
        print(f"  Model source:       {model_source}")

        if generate_shap:
            try:
                shap_plot_path = plot_shap_summary(model, X_dev, out_dir, model_name, plot_type='bar')
                results['shap_plot'] = shap_plot_path
                print(f"SHAP summary plot saved to: {shap_plot_path}")
            except Exception as exc:
                print(f"SHAP plot skipped: {exc}")
        return results

    except Exception as e:
        print(f"Error training {model_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'model_name': model_name,
            'status': 'failed',
            'error': str(e)
        }


# Train multiple models on preprocessed data and return a comparison table with bootstrap CIs.
def train_multiple_models(in_dir, param_dir, out_dir, model_list=None, n_bootstrap=1000,
                           threshold=TARGET_THRESHOLD, random_state=RANDOM_STATE, generate_shap=True):
    if model_list is None:
        model_list = ALL_MODELS

    X_dev, X_lock, y_dev, y_lock = load_preprocessed_data(in_dir, include_lock=True)
    calibrated_thresholds = load_calibrated_thresholds(in_dir)

    results_list = []
    formatted_results = []
    model_objects = {}

    for model_name in model_list:
        result = train_and_evaluate_model(
            X_dev, y_dev, X_lock, y_lock, model_name, out_dir,
            param_dir=param_dir, random_state=random_state,
            n_bootstrap=n_bootstrap, threshold=threshold,
            youden_threshold=calibrated_thresholds.get(model_name), generate_shap=generate_shap,
        )
        results_list.append(result)

        if result['status'] == 'success':
            row = {
                'Model': model_name,
                'Model Source': result['model_source'],
                'AUC': f"{result['lock_auc']:.2f}",
                'AUC 95% CI': f"({result['auc_ci_lower']:.2f}, {result['auc_ci_upper']:.2f})",
                'Sensitivity': f"{result['sensitivity']:.2f}",
                'Sens 95% CI': f"({result['sens_ci_lower']:.2f}, {result['sens_ci_upper']:.2f})",
                'Specificity': f"{result['specificity']:.2f}",
                'Spec 95% CI': f"({result['spec_ci_lower']:.2f}, {result['spec_ci_upper']:.2f})",
                'N Dev': result['n_dev_samples'],
                'N Lock': result['n_lock_samples'],
            }
            if result['youden_threshold'] is not None:
                row.update({
                    'Youden Threshold': f"{result['youden_threshold']:.3f}",
                    'Sensitivity (Youden)': f"{result['youden_sensitivity']:.2f}",
                    'Sens (Youden) 95% CI': f"({result['youden_sens_ci_lower']:.2f}, {result['youden_sens_ci_upper']:.2f})",
                    'Specificity (Youden)': f"{result['youden_specificity']:.2f}",
                    'Spec (Youden) 95% CI': f"({result['youden_spec_ci_lower']:.2f}, {result['youden_spec_ci_upper']:.2f})",
                })
            formatted_results.append(row)
            model_objects[model_name] = result['model']
        else:
            formatted_results.append({
                'Model': model_name,
                'Model Source': '-',
                'AUC': '-', 'AUC 95% CI': '-',
                'Sensitivity': '-', 'Sens 95% CI': '-',
                'Specificity': '-', 'Spec 95% CI': '-',
                'Error': result.get('error', 'Unknown'),
                'N Dev': '-', 'N Lock': '-',
            })

    comparison_df = pd.DataFrame(formatted_results)
    create_roc_comparison_plot(results_list, out_dir)
    # svm_linear consistently underperforms the rest of the roster (Section 7 reports/presentation
    # material) -- a second plot with it excluded avoids one bad curve compressing the others.
    results_no_svm_linear = [r for r in results_list if r.get('model_name') != 'svm_linear']
    create_roc_comparison_plot(results_no_svm_linear, out_dir, filename='model_roc_comparison_no_svm_linear.png')

    print(f"\n{'='*100}")
    print(f"MODEL COMPARISON SUMMARY (threshold={threshold}, with 95% Bootstrap Confidence Intervals)")
    print(f"{'='*100}")
    print(comparison_df.to_string(index=False))
    print(f"{'='*100}\n")

    comparison_df.to_csv(f"{out_dir}/model_comparison.csv", index=False)
    print(f"Comparison saved to: {out_dir}/model_comparison.csv")

    # Best model is now selected on the raw float AUC carried in results_list, not by re-parsing the
    # 2-decimal-rounded string from the comparison table (the old version broke ties by list order).
    successful = [r for r in results_list if r.get('status') == 'success']
    if successful:
        best = max(successful, key=lambda r: r['lock_auc'])
        best_model = model_objects[best['model_name']]

        with open(f"{out_dir}/best_model.pkl", 'wb') as f:
            pickle.dump(best_model, f)

        print(f"Best model ({best['model_name']}) saved: {best['lock_auc']:.3f} AUC "
              f"[source={best['model_source']}]\n")

    return comparison_df, model_objects


# =====================================================================================================
# Stage runners
# =====================================================================================================

def run_partition_stage(args):
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    df_all_seq = pd.read_csv(args.features_path, index_col=0)
    df_filtered = preselect_sequences(df_all_seq, allowed_sequences=args.sequences)
    df_filtered = filter_feature_families(df_filtered, exclude_families=args.exclude_feature_families)

    if args.feature_set == 'both':
        # Joined AFTER sequence preselection: preselect_sequences keeps any column whose name
        # contains a sequence string, and several clinical column names (e.g. "Capsule on PVP")
        # would otherwise collide with sequence substrings like "pvp".
        df_clinical_feats = load_clinical_features(args.clinical_data_path)
        df_filtered = df_filtered.join(df_clinical_feats, how='left')
        print(f"Joined clinical features. Feature matrix now has {df_filtered.shape[1]} columns.")

    X_dev, y_dev, X_lock, y_lock, df_clinical_dev, df_clinical_lock = partition_with_constraints(
        df_filtered,
        args.outcomes_path,
        args.icc_path,
        clinical_data_path=args.clinical_data_path,
        label_col=args.label_col,
        cutoff=args.cutoff,
        test_size=args.test_size,
        output_dir=args.output_dir,
        clinical_output_dir=args.clinical_output_dir,
    )

    _clin_dir = args.clinical_output_dir or args.output_dir
    compare_dev_test(df_clinical_dev, df_clinical_lock, args.compare_variables,
                      output_dir=_clin_dir, outcome_col=args.compare_outcome_col)

    X_dev_clean = global_cleanup(X_dev, id_col='PatientID', cv_threshold=args.cv_threshold)

    clinical_cols = [c for c in X_dev_clean.columns if c.startswith(CLINICAL_FEATURE_PREFIX)]
    radiomics_cols = [c for c in X_dev_clean.columns if c not in clinical_cols]
    X_dev_clean_icc = interobserver_stability(X_dev_clean[radiomics_cols], args.path_r1, args.path_r2,
                                               id_col='PatientID', threshold=args.icc_threshold)
    if clinical_cols:
        # Clinical/semantic features have no double-reader segmentation, so ICC doesn't apply to
        # them -- they pass straight through to the same fold-specific selection as everything else.
        X_dev_clean_icc = pd.concat([X_dev_clean_icc, X_dev_clean[clinical_cols]], axis=1)
        print(f"Clinical features bypass ICC filter, added back: {len(clinical_cols)}")

    y_dev_filtered = y_dev.loc[X_dev_clean_icc.index]
    print(f"After filtering: X_dev has {X_dev_clean_icc.shape[0]} samples, "
          f"y_dev has {len(y_dev_filtered)} samples")

    # Nested-CV OOF diagnostic: unaffected by --feature-selection-method -- always reselects fresh
    # per outer fold, so it stays the honest reference number to compare either method's final
    # (fold-consensus or stability-selection) lock-set result against.
    fold_coefficients, global_scorecard, oof_predictions = run_cross_validation(
        X_dev_clean_icc, y_dev_filtered, target_count=args.lasso_target_count,
        p_value_cutoff=args.p_value_cutoff, metric=args.univariate_metric,
        metric_cutoff=args.metric_cutoff, decorrelation_threshold=args.decorrelation_threshold,
        penalty=args.penalty, l1_ratios=args.l1_ratios, random_state=RANDOM_STATE,
    )
    fold_coefficients.to_csv(f"{args.output_dir}/lasso_coeffs_fold.csv", index=True)

    if args.feature_selection_method == 'stability':
        # Cluster-tallied schema (Cluster index, Representative_Feature column) is incompatible
        # with the older per-column stability_selection.csv -- a distinct filename avoids silently
        # misreading a stale cache left over from before this tallying change.
        stability_cache_path = f"{args.output_dir}/stability_selection_clustered.csv"
        if os.path.exists(stability_cache_path) and not args.stability_force_recompute:
            # Re-running just to try a different --stability-threshold doesn't need to redo the
            # (slow, ~200-resample) selection-frequency computation -- only the frequency cutoff
            # applied to it changed, not the resamples themselves. Pass --stability-force-recompute
            # to actually rerun (e.g. after changing --stability-n-resamples/--stability-resample-frac).
            print(f"Reusing cached stability selection frequencies from: {stability_cache_path}")
            stability_df = pd.read_csv(stability_cache_path, index_col=0)
        else:
            stability_df = run_stability_selection(
                X_dev_clean_icc, y_dev_filtered, target_count=args.lasso_target_count,
                p_value_cutoff=args.p_value_cutoff, metric=args.univariate_metric,
                metric_cutoff=args.metric_cutoff, n_resamples=args.stability_n_resamples,
                resample_frac=args.stability_resample_frac,
                decorrelation_threshold=args.decorrelation_threshold,
                cluster_threshold=args.cluster_threshold, penalty=args.penalty,
                l1_ratios=args.l1_ratios, random_state=RANDOM_STATE,
            )
            stability_df.to_csv(stability_cache_path, index=True)
        frequent_clusters = stability_df[
            (stability_df['Selection_Frequency'] >= args.stability_threshold) & stability_df['Sign_Stable']
        ]
        consensus_features = frequent_clusters.head(args.target_feature_count)['Representative_Feature'].tolist()
        selection_error = (
            f"No consensus features survived stability selection: 0 clusters were selected in "
            f">= {args.stability_threshold:.0%} of {args.stability_n_resamples} resamples with a "
            f"stable coefficient sign (see stability_selection_clustered.csv). Loosen "
            f"--stability-threshold and/or re-run."
        )
    else:
        frequent_features = fold_coefficients[
            (fold_coefficients['Selection_Frequency'] >= args.min_fold_n) & fold_coefficients['Sign_Stable']
        ]
        consensus_features = frequent_features.head(args.target_feature_count).index.tolist()
        selection_error = (
            f"No consensus features survived selection: 0 features were selected by LASSO in "
            f">= --min-fold-n={args.min_fold_n} of the 5 outer folds with a stable coefficient sign. "
            f"Each fold's own LASSO selection was non-empty (see lasso_coeffs_fold.csv), they just "
            f"didn't overlap enough across folds -- a fold-to-fold selection instability symptom, "
            f"not a data error. Loosen --min-fold-n and/or re-run."
        )

    if len(consensus_features) == 0:
        raise RuntimeError(selection_error)

    # Honest, dev-only nested CV estimate: how well does the whole selection+fit pipeline
    # generalize, pooled across all of X_dev rather than the much smaller lock set. Never touches
    # X_lock/y_lock, so it can be looked at as many times as needed without eroding the lock set.
    nested_cv_summary = summarize_nested_cv(oof_predictions, y_dev_filtered, random_state=RANDOM_STATE)
    nested_cv_summary.to_csv(f"{args.output_dir}/nested_cv_diagnostics.csv", index=False)
    nested_cv_summary[['model', 'youden_threshold']].to_csv(f"{args.output_dir}/calibrated_thresholds.csv", index=False)

    print(f"\n{'='*100}")
    print("DEV-ONLY NESTED CV DIAGNOSTICS (honest, leakage-free estimate -- does not touch the lock set)")
    print(f"{'='*100}")
    print(nested_cv_summary.to_string(index=False))
    print(f"{'='*100}\n")

    print(f"Consensus features identified: {len(consensus_features)}")
    print(consensus_features)
    global_scorecard.to_csv(f"{args.output_dir}/univariate_metrics.csv")

    run_feature_diagnostics(X_dev_clean_icc, y_dev_filtered, X_lock, y_lock,
                             consensus_features, args.output_dir)

    # Final dev/lock preprocessing: fit imputer/scaler on dev ONLY, transform lock.
    imputer = KNNImputer(n_neighbors=5)
    scaler = StandardScaler()

    X_dev_subset = X_dev_clean_icc[consensus_features]
    X_lock_subset = X_lock[consensus_features]

    X_dev_impute = pd.DataFrame(imputer.fit_transform(X_dev_subset), columns=consensus_features, index=X_dev_subset.index)
    X_dev_final = pd.DataFrame(scaler.fit_transform(X_dev_impute), columns=consensus_features, index=X_dev_subset.index)

    X_lock_impute = pd.DataFrame(imputer.transform(X_lock_subset), columns=consensus_features, index=X_lock.index)
    X_lock_final = pd.DataFrame(scaler.transform(X_lock_impute), columns=consensus_features, index=X_lock.index)

    print("\nSaving preprocessing items for reference and reproducibility")
    X_dev_final.to_csv(f"{args.output_dir}/X_dev_processed.csv")
    X_lock_final.to_csv(f"{args.output_dir}/X_lock_processed.csv")
    pd.Series(y_dev_filtered, name='y_dev').to_csv(f"{args.output_dir}/y_dev.csv")
    pd.Series(y_lock, name='y_lock').to_csv(f"{args.output_dir}/y_lock.csv")
    pd.Series(consensus_features, name='feature').to_csv(f"{args.output_dir}/consensus_features.csv", index=False)
    pd.DataFrame({'n_features_in': [imputer.n_features_in_]}).to_csv(f"{args.output_dir}/imputer_n_features.csv", index=False)

    with open(f"{args.output_dir}/imputer.pkl", 'wb') as f:
        pickle.dump(imputer, f)
    with open(f"{args.output_dir}/scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)

    print(f"Artifacts saved to {args.output_dir}")
    print(f"Processed Development Set: {X_dev_final.shape}")
    print(f"Processed Lock Set: {X_lock_final.shape}")
    print(f"Consensus Features: {len(consensus_features)}")


def run_gridsearch_stage(args):
    os.makedirs(args.output_dir, exist_ok=True)
    X_dev, y_dev = load_preprocessed_data(args.input_dir, include_lock=False)
    results_df = perform_grid_search(
        X_dev, y_dev, model_names=args.models, out_dir=args.output_dir,
        cv=args.cv, scoring=args.scoring, n_jobs=args.n_jobs,
        random_state=RANDOM_STATE, save=True,
    )
    print(results_df)


def run_train_stage(args):
    os.makedirs(args.output_dir, exist_ok=True)
    comparison_df, _model_objects = train_multiple_models(
        in_dir=args.input_data_dir, param_dir=args.input_params_dir, out_dir=args.output_dir,
        model_list=args.models, n_bootstrap=args.bootstrap, threshold=args.threshold,
        random_state=RANDOM_STATE, generate_shap=args.shap,
    )
    print(comparison_df)


# =====================================================================================================
# CLI
# =====================================================================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Radiomics ML pipeline: partition data, select features, tune hyperparameters, '
                    'and train/evaluate models.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--stage', choices=['partition', 'gridsearch', 'train'], default=DEFAULT_STAGE,
                         help="Which pipeline stage to run. Defaults to DEFAULT_STAGE at the top of this file.")

    # --- partition stage ---
    parser.add_argument('--sequences', nargs='+', default=['pvp', 'tp', 'hbp', 'dwi_b1', 'adc'])
    parser.add_argument('--exclude-feature-families', nargs='*', default=['wavelet'],
        help="PyRadiomics filter families to drop before feature selection (prefix match on "
             "the '{sequence}_{family}_...' column segment, e.g. 'wavelet' drops all 8 "
             "wavelet-* subbands). Pass --exclude-feature-families with no values to keep all.")
    parser.add_argument('--features-path', type=str,
        default='/Volumes/T7 Shield/Restrepo/HCC_Retrospective_radiomics/Results/Radiomics_sheets/SA1_2/R-SA1_2_I-sitkBSpline_S-1mm_W-perseq.csv')
    parser.add_argument('--path-r1', type=str,
        default='/Volumes/T7 Shield/Restrepo/HCC_Retrospective_radiomics/Results/Radiomics_sheets/inter-reader/Reader1/R-Reader1_I-sitkBSpline_S-1mm_W-perseq.csv')
    parser.add_argument('--path-r2', type=str,
        default='/Volumes/T7 Shield/Restrepo/HCC_Retrospective_radiomics/Results/Radiomics_sheets/inter-reader/Reader2/R-Reader2_I-sitkBSpline_S-1mm_W-perseq.csv')
    parser.add_argument('--outcomes-path', type=str,
        default='/Volumes/T7 Shield/Restrepo/HCC_Retrospective_radiomics/databases/immunophenotype_outcomes.xlsx')
    parser.add_argument('--clinical-data-path', type=str,
        default='/Volumes/T7 Shield/Restrepo/HCC_Retrospective_radiomics/databases/SA1-2_clinical_imaging.xlsx')
    parser.add_argument('--icc-path', type=str, default=None,
        help="Defaults to --path-r1, since that file is already restricted to inter-reader patients.")
    parser.add_argument('--label-col', type=str, default='TILs_resection')
    parser.add_argument('--cutoff', type=float, default=3)
    parser.add_argument('--test-size', type=int, default=30)
    parser.add_argument('--cv-threshold', type=float, default=0.1,
        help="Minimum coefficient of variation (std / |mean|) to keep a feature in global_cleanup. "
             "Scale-invariant replacement for the old raw-std threshold.")
    parser.add_argument('--icc-threshold', type=float, default=0.85)
    parser.add_argument('--lasso-target-count', type=int, default=100)
    parser.add_argument('--p-value-cutoff', type=float, default=0.2,
        help="FDR-adjusted (Benjamini-Hochberg) q-value threshold, applied AFTER the effect-size "
             "(--metric-cutoff) screen and computed only within that screened subset -- not across "
             "all candidate features -- so correction isn't diluted by thousands of features nobody "
             "expected to be significant. Still a q-value, not a raw p-value (0.2 is a common "
             "screening-stage choice).")
    parser.add_argument('--univariate-metric', type=str, default='auc')
    parser.add_argument('--penalty', choices=['l1', 'elasticnet'], default='l1',
        help="Penalty for the per-fold/per-resample feature-selection model. 'l1' (default, "
             "unchanged): pure LASSO via liblinear. 'elasticnet': LASSO+ridge via saga, whose "
             "grouping effect gives correlated features similar coefficients instead of "
             "arbitrarily zeroing all but one -- the standard fix for L1's instability on "
             "correlated predictors. Selects more features per resample by design, and is "
             "substantially slower (saga + an l1_ratios grid on top of the Cs grid).")
    parser.add_argument('--l1-ratios', nargs='+', type=float, default=DEFAULT_L1_RATIOS,
        help="Elastic-net mixing grid, cross-validated alongside C. 1.0 == pure L1, 0.0 == pure "
             "ridge. Only used when --penalty elasticnet.")
    parser.add_argument('--cluster-threshold', type=float, default=None,
        help="Correlation threshold used ONLY for stability-selection's post-hoc tallying "
             "(compute_correlation_clusters) -- independent of --decorrelation-threshold, which "
             "controls the per-resample pairwise pruning the penalized model actually sees. "
             "Defaults to --decorrelation-threshold when not set. Set this explicitly (e.g. to "
             "0.85) when relaxing/disabling --decorrelation-threshold (e.g. to 1.0, to test "
             "elastic net on undecorrelated features) -- otherwise clustering gets silently "
             "disabled too, exactly when it's needed most to keep tallying interpretable.")
    parser.add_argument('--decorrelation-threshold', type=float, default=0.85,
        help="Absolute Spearman correlation above which spearman_filter drops one of a pair of "
             "features (keeping the one with the better univariate p-value). Lower = more "
             "aggressive pruning of near-duplicate columns (e.g. an original-filter feature and "
             "its own LoG/wavelet variants), which directly reduces the vote-splitting across "
             "resamples that stability selection is sensitive to. Applies to both the nested-CV "
             "OOF diagnostic and whichever --feature-selection-method builds consensus_features, "
             "for a consistent comparison.")
    parser.add_argument('--metric-cutoff', type=float, default=0.65)
    parser.add_argument('--min-fold-n', type=int, default=3)
    parser.add_argument('--target-feature-count', type=int, default=10)
    parser.add_argument('--feature-selection-method', choices=['fold-consensus', 'stability'],
        default='fold-consensus',
        help="'fold-consensus' (default, unchanged): consensus_features chosen from the same "
             "5-fold split as the nested-CV OOF diagnostic (--min-fold-n of 5 trials). "
             "'stability': Meinshausen-Buhlmann stability selection -- many more (see "
             "--stability-n-resamples), class-stratified resamples of the whole dev set, each "
             "rerunning the full screen/FDR/decorrelation/LASSO from scratch; keeps features "
             "selected in >= --stability-threshold of resamples with a stable sign. Targets the "
             "winner's-curse-style optimism in a single full-dev selection pass -- does not "
             "change the nested-CV OOF diagnostic, which always reselects fresh per fold either way.")
    parser.add_argument('--stability-n-resamples', type=int, default=200)
    parser.add_argument('--stability-resample-frac', type=float, default=0.75,
        help="Fraction of each outcome class drawn (without replacement) per stability-selection resample.")
    parser.add_argument('--stability-threshold', type=float, default=0.7,
        help="Minimum selection frequency across stability-selection resamples to keep a feature.")
    parser.add_argument('--stability-force-recompute', action='store_true',
        help="Recompute stability_selection.csv even if one already exists in --output-dir "
             "(otherwise a cached file is reused -- see run_partition_stage).")
    parser.add_argument('--compare-variables', nargs='+', default=[
        'Age', 'Gender', 'BMI', 'Cirrhosis', 'Capsule (on PVP yes/no)', 'Mosaic appearance (yes/no)',
        'Macrovascular invasion (yes/no)', 'Maximum tumor size on T1 portal venous phase (PVP)',
        'TILs_resection', 'TLS_resection',
    ])
    parser.add_argument('--compare-outcome-col', type=str, default='TILs_resection')
    parser.add_argument('--clinical-output-dir', type=str, default=None,
        help="Where clinical_metadata_dev/lock.csv and dev_test_comparison.xlsx get written. "
             "Defaults to --output-dir. Sweeps over sequence combos should point this at one shared "
             "location, since the dev/lock split doesn't depend on which sequences are used.")
    parser.add_argument('--feature-set', choices=['radiomics', 'both'], default='radiomics',
        help="'radiomics' (default): current behavior, unaffected. 'both': encoded clinical/MRI "
             "semantic features (age, gender, BMI, AFP, cirrhosis, etiology, capsule on PVP, mosaic "
             "appearance, T2 signal, tumor size on PVP) appended to the radiomics matrix, treated "
             "like any other feature for LASSO to select from. The clinical-only benchmark lives in "
             "its own script (clinical_only_pipeline.py), not here.")

    # --- gridsearch stage ---
    parser.add_argument('--input-dir', type=str, default=None,
        help="Partition-stage output dir to read from. Defaults to the same cutoff/sequence-derived "
             "path as --output-dir when not provided.")
    parser.add_argument('--cv', type=int, default=5)
    parser.add_argument('--scoring', type=str, default='roc_auc')
    parser.add_argument('--n-jobs', type=int, default=6)

    # --- train stage ---
    parser.add_argument('--input-data-dir', type=str, default=None,
        help="Partition-stage output dir for processed features. Defaults to same cutoff/sequence path.")
    parser.add_argument('--input-params-dir', type=str, default=None,
        help="Gridsearch-stage output dir for pickled estimators. Defaults to same cutoff/sequence path.")
    parser.add_argument('--bootstrap', type=int, default=1000)
    parser.add_argument('--threshold', type=float, default=TARGET_THRESHOLD)
    parser.add_argument('--shap', action='store_true')

    # --- shared ---
    parser.add_argument('--models', nargs='+', default=None)
    parser.add_argument('--output-dir', type=str, default=None,
        help="Defaults depend on --stage if not provided.")

    args = parser.parse_args()

    if args.icc_path is None:
        args.icc_path = args.path_r1

    cutoff_val = int(args.cutoff) if args.cutoff == int(args.cutoff) else args.cutoff
    neg_max = int(args.cutoff) - 1
    cutoff_label = f"tils_{cutoff_val}_vs_0-{neg_max}"
    feature_set_label = 'clinical+radiomics' if args.feature_set == 'both' else 'radiomics'
    # Outputs go to OneDrive (synced/shared); source data (radiomics CSVs, clinical/outcomes
    # spreadsheets) stays on T7 Shield -- see --features-path/--clinical-data-path/etc. defaults.
    _base = (f'/Users/Francisco/Library/CloudStorage/OneDrive-TheMountSinaiHospital/'
             f'Regeneron retrospective radiomics/Results/main_pipeline/immunophenotype/'
             f'{cutoff_label}/' + feature_set_label)

    if args.output_dir is None:
        if args.stage == 'partition':
            args.output_dir = f"{_base}/clean_partition_select"
        elif args.stage == 'gridsearch':
            args.output_dir = f"{_base}/optimized_params"
        elif args.stage == 'train':
            args.output_dir = f"{_base}/trained_model_outputs"

    if args.input_dir is None:
        args.input_dir = f"{_base}/clean_partition_select"
    if args.input_data_dir is None:
        args.input_data_dir = f"{_base}/clean_partition_select"
    if args.input_params_dir is None:
        args.input_params_dir = f"{_base}/optimized_params"

    return args


if __name__ == "__main__":
    args = parse_arguments()

    print("\n" + "=" * 100)
    print(f"RADIOMICS PIPELINE - STAGE: {args.stage.upper()}")
    print(f"Output directory: {args.output_dir}")
    if args.models:
        print(f"Models: {', '.join(args.models)}")
    print("=" * 100)

    if args.stage == 'partition':
        run_partition_stage(args)
    elif args.stage == 'gridsearch':
        run_gridsearch_stage(args)
    elif args.stage == 'train':
        run_train_stage(args)

    print(f"{'='*100}")
    print(f"STAGE '{args.stage}' COMPLETE")
    print(f"{'='*100}\n")
