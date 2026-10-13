##### ------------ MRI-radiomics regression pipeline for TILs immunophenotype (ordinal-as-continuous) ---- #####
# Sibling of main_pipeline.py (binary classification at a chosen cutoff) and recurrence_pipeline.py
# (survival). This one treats TILs_resection (0/1/2/3) as a continuous/ordinal regression target
# instead of binarizing it at a cutoff -- see project discussion: binarizing throws away information
# and forces two separate, differently-imbalanced cutoff analyses (2-vs-0/1 at 82/18%, 3-vs-0/1/2 at
# 68/32%) where a single regression model uses every patient once. This is the "plain regression"
# option (equal-interval-spacing assumption across the 4 levels), not a full proportional-odds/
# cumulative-link ordinal model -- see the option-A/option-B discussion in project memory for why
# that's the right place to start given the sample size here.
#
# Same architecture as main_pipeline.py end to end (extraction -> QC filters -> per-fold univariate
# screen -> FDR -> decorrelation -> penalized-model consensus voting -> nested-CV diagnostics ->
# gridsearch -> lock-set testing); every outcome-type-specific piece has a regression counterpart:
#   Mann-Whitney U + AUC          -> Spearman correlation (rho, p-value)
#   LogisticRegressionCV (L1)     -> LassoCV (regression L1), same fold-count/target-count mechanics
#   predict_proba + ROC-AUC       -> predict + Spearman rho / R^2 / MAE (rho is the primary ranking
#                                    metric, playing AUC's role -- it's threshold-free and monotone,
#                                    matching what "predicts ordinal TILs level" actually means)
#   sensitivity/specificity       -> dropped entirely (no decision threshold in a regression model)
#   ROC comparison plot           -> predicted-vs-actual scatter grid, one panel per model
# Stratification (dev/lock split, inner/outer CV folds) uses the label's own 4 discrete values
# (TILs_resection is exactly {0,1,2,3} even though modeled as continuous) instead of a synthetic
# quantile-binned proxy -- more information-preserving than the classification pipeline's binary
# stratification, and free since the levels already exist.
#
# Pipeline overview (run one stage at a time via --stage):
#   --stage partition  : preselect sequences, partition dev/lock (stratified on the 4 TILs levels),
#                         global cleanup + ICC filtering, cross-validated consensus feature selection.
#   --stage gridsearch  : load processed dev features, tune hyperparameters per model, pickle best estimators.
#   --stage train       : load processed dev/lock features (+ tuned estimators if available), evaluate
#                         every model on the lock set (Spearman rho / R^2 / MAE + bootstrap CIs).
#
# Example:
#   python tils_regression_pipeline.py --stage partition --sequences pre ap pvp tp hbp
#   python tils_regression_pipeline.py --stage gridsearch --sequences pre ap pvp tp hbp
#   python tils_regression_pipeline.py --stage train --sequences pre ap pvp tp hbp --shap

import os
# Must be set before numpy/sklearn (libomp) and torch/tabpfn (libiomp) are imported -- having both
# OpenMP runtimes loaded at once is a known cause of silent hangs/crashes on macOS (see also the
# harder segfault-on-unpickle variant documented in project memory for scripts that import torch
# before ever touching a pickled xgboost model -- this file avoids that by importing xgboost, THEN
# torch/tabpfn, at the top, exactly like main_pipeline.py, which has run this exposure-order safely
# across dozens of combos).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import pickle
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import mannwhitneyu, chi2_contingency, spearmanr

from sklearn.svm import SVR
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, LassoCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, make_scorer

from xgboost import XGBRegressor
import torch
from tabpfn import TabPFNRegressor
import shap

# --- Global, single-source-of-truth constants -----------------------------------------------------
RANDOM_STATE = 42        # All CV splits, model fitting, and bootstrap resampling.
PARTITION_SEED = 997     # Dev/lock split -- same seed as main_pipeline.py/recurrence_pipeline.py by
                          # convention, but NOT the same patient split: this pipeline's dev/lock
                          # membership is stratified on the 4-level TILs_resection score, not y_bin,
                          # so the same seed does not guarantee the same patients land in lock here
                          # as in the classification pipeline (documented divergence, same as
                          # classification-vs-recurrence already is).

ALL_MODELS = ['linear', 'svm_linear', 'svm_rbf', 'rf', 'extratrees', 'xgboost', 'gp', 'tabpfn']
NESTED_CV_MODELS = ALL_MODELS
CLINICAL_FEATURE_PREFIX = 'clinical__'

DEFAULT_STAGE = 'gridsearch'  # one of: 'partition', 'gridsearch', 'train'

_spearman_rho = lambda y_true, y_pred: spearmanr(y_true, y_pred).correlation
SPEARMAN_SCORER = make_scorer(_spearman_rho, greater_is_better=True)


# =====================================================================================================
# Sequence / feature-family selection -- unchanged from main_pipeline.py (outcome-agnostic)
# =====================================================================================================

def preselect_sequences(df, allowed_sequences, id_col='PatientID'):
    if not allowed_sequences:
        return df
    keep_cols = [id_col] if id_col in df.columns else []
    for col in df.columns:
        if col == id_col:
            continue
        if any(seq == col.split('_')[0] or col.startswith(seq + '_') for seq in allowed_sequences):
            keep_cols.append(col)
    return df[keep_cols]


def filter_feature_families(df, exclude_families=None, id_col='PatientID'):
    if not exclude_families:
        return df
    keep_cols = [c for c in df.columns
                 if c == id_col or not any(fam.lower() in c.lower() for fam in exclude_families)]
    print(f"Excluded families {exclude_families}: {df.shape[1]} -> {len(keep_cols)} columns")
    return df[keep_cols]


# --- Clinical / MRI semantic features (unchanged from main_pipeline.py) ----------------------------

def _classify_etiology(text):
    if pd.isna(text):
        return np.nan
    t = str(text).lower()
    if 'hbv' in t or 'hepatitis b' in t:
        return 'HBV'
    if 'hcv' in t or 'hepatitis c' in t:
        return 'HCV'
    if 'mash' in t or 'nash' in t or 'fatty' in t:
        return 'MASH'
    if 'alcohol' in t or 'etoh' in t:
        return 'Alcohol'
    return 'Other'


def load_clinical_features(clinical_data_path, id_col='PatientID'):
    df = pd.read_excel(clinical_data_path)
    out = pd.DataFrame(index=df[id_col])

    def add_numeric(col, name):
        if col in df.columns:
            out[f"{CLINICAL_FEATURE_PREFIX}{name}"] = pd.to_numeric(df[col], errors='coerce').values

    add_numeric('Age', 'age')
    add_numeric('BMI', 'bmi')
    add_numeric('Maximum tumor size on T1 portal venous phase (PVP)', 'tumor_size_pvp')

    def add_binary(col, name, pos_values):
        if col in df.columns:
            out[f"{CLINICAL_FEATURE_PREFIX}{name}"] = df[col].astype(str).str.strip().str.lower().isin(
                [v.lower() for v in pos_values]).astype(float)

    add_binary('Gender', 'gender_male', ['M', 'Male'])
    add_binary('Cirrhosis', 'cirrhosis', ['Y', 'Yes'])
    add_binary('Capsule (on PVP yes/no)', 'capsule_pvp', ['Y', 'Yes'])
    add_binary('Mosaic appearance (yes/no)', 'mosaic', ['Y', 'Yes'])

    if 'Etiology of liver disease' in df.columns:
        etiology = df['Etiology of liver disease'].apply(_classify_etiology)
        dummies = pd.get_dummies(etiology, prefix=f"{CLINICAL_FEATURE_PREFIX}etiology")
        out = out.join(dummies.set_axis(df[id_col]), how='left')

    out.index.name = id_col
    print(f"Loaded {out.shape[1]} clinical features for {out.shape[0]} patients from: {clinical_data_path}")
    return out


# =====================================================================================================
# STAGE 1: Sequence preselection, partitioning, cleanup, and cross-validated feature selection
# =====================================================================================================

# Partitions into dev/lock, stratified on TILs_resection's own 4 discrete values (0/1/2/3) rather than
# a binarized y_bin -- no cutoff involved, and this preserves more of the outcome's information in the
# split than the classification pipeline's binary stratification does.
def partition_with_constraints(df_filtered, outcomes_path, icc_path, clinical_data_path, label_col,
                                test_size, output_dir, random_state=PARTITION_SEED,
                                clinical_output_dir=None):
    df_outcomes = pd.read_excel(outcomes_path)
    n_before = len(df_outcomes)
    df_outcomes = df_outcomes[df_outcomes[label_col].notna()].copy()
    if len(df_outcomes) < n_before:
        print(f"Dropped {n_before - len(df_outcomes)} patients with missing '{label_col}'.")
    df_outcomes['y_reg'] = df_outcomes[label_col].astype(float)
    df_icc = pd.read_csv(icc_path)

    df_merged = df_filtered.reset_index().merge(
        df_outcomes[['PatientID', 'y_reg']], on='PatientID', how='inner').set_index('PatientID')
    # Sort by PatientID before any split -- see main_pipeline.py's identical comment; pins the split
    # to patient identity + seed alone, not incidental row order in whichever CSV was loaded.
    df_merged = df_merged.sort_index()

    icc_ids = df_icc['PatientID'].unique()
    df_dev_must = df_merged[df_merged.index.isin(icc_ids)]
    df_rest = df_merged[~df_merged.index.isin(icc_ids)]

    X_rest = df_rest.drop(columns=['y_reg'])
    y_rest = df_rest['y_reg']
    # Stratify on the rounded integer level (0/1/2/3), not the continuous value itself -- sklearn's
    # stratify= needs discrete groups, and this label already only takes 4 distinct values.
    strata = y_rest.round().astype(int)

    X_dev_rest, X_lock, y_dev_rest, y_lock = train_test_split(
        X_rest, y_rest, test_size=test_size, stratify=strata, random_state=random_state
    )

    X_dev = pd.concat([df_dev_must.drop(columns=['y_reg']), X_dev_rest])
    y_dev = pd.concat([df_dev_must['y_reg'], y_dev_rest])

    df_clinical = pd.read_excel(clinical_data_path)
    outcome_cols = [c for c in df_outcomes.columns if c not in ('PatientID', 'y_reg')]
    df_clinical = df_clinical.merge(df_outcomes[['PatientID'] + outcome_cols], on='PatientID', how='left')
    df_clinical_dev = df_clinical[df_clinical['PatientID'].isin(X_dev.index)]
    df_clinical_lock = df_clinical[df_clinical['PatientID'].isin(X_lock.index)]

    _clin_dir = clinical_output_dir or output_dir
    Path(_clin_dir).mkdir(parents=True, exist_ok=True)
    df_clinical_dev.to_csv(f"{_clin_dir}/clinical_metadata_dev.csv", index=False)
    df_clinical_lock.to_csv(f"{_clin_dir}/clinical_metadata_lock.csv", index=False)

    print("Partitioning Complete.")
    print(f"Dev Set: {len(X_dev)} (Includes {len(df_dev_must)} ICC patients)")
    print(f"Lock Set: {len(X_lock)}")
    print(f"TILs level counts -- dev: {y_dev.round().astype(int).value_counts().sort_index().to_dict()}, "
          f"lock: {y_lock.round().astype(int).value_counts().sort_index().to_dict()}")

    return X_dev, y_dev, X_lock, y_lock, df_clinical_dev, df_clinical_lock


# Unchanged from main_pipeline.py -- outcome-agnostic descriptive comparison.
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
                return ", ".join([f"{idx}: {val} ({(val/total)*100:.1f}%)" for idx, val in counts.items()])
            dev_str = get_cat_str(df_dev, var)
            lock_str = get_cat_str(df_lock, var)
            combined_series = pd.concat([df_dev[var], df_lock[var]])
            group_labels = pd.Series(['Dev'] * len(df_dev) + ['Lock'] * len(df_lock), index=combined_series.index)
            contingency = pd.crosstab(combined_series, group_labels)
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
            'Statistical Test': test_type,
        })
    results = pd.DataFrame(results)
    results.to_excel(f"{output_dir}/dev_test_comparison.xlsx", index=False)
    return results


# Unchanged from main_pipeline.py -- outcome-agnostic.
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
    cv = std / mean_abs.replace(0, np.nan)
    keep = (std > 1e-8) & ((cv > cv_threshold) | cv.isna())
    df = df.loc[:, keep]
    print(f"Cleanup finished. Columns remaining: {df.shape[1]}")
    return df


# Unchanged from main_pipeline.py -- outcome-agnostic.
def interobserver_stability(df, path_r1, path_r2, threshold, id_col='PatientID'):
    df_r1 = pd.read_csv(path_r1)
    df_r2 = pd.read_csv(path_r2)
    common_ids = np.intersect1d(df_r1[id_col], df_r2[id_col])
    r1 = df_r1.set_index(id_col).loc[common_ids].sort_index()
    r2 = df_r2.set_index(id_col).loc[common_ids].sort_index()
    features = [f for f in df.columns if f in r1.columns and f in r2.columns and r1[f].dtype != 'object']
    r1, r2 = r1[features], r2[features]
    n = len(common_ids)
    print(f"ICC: {len(features)} features across {n} patients (vectorized)...")
    r1_arr = r1.values.astype(float)
    r2_arr = r2.values.astype(float)
    valid = ~(np.isnan(r1_arr).any(axis=0) | np.isnan(r2_arr).any(axis=0))
    r1_v, r2_v = r1_arr[:, valid], r2_arr[:, valid]
    valid_feats = [f for f, v in zip(features, valid) if v]
    gm = (r1_v.mean(0) + r2_v.mean(0)) / 2
    sm = (r1_v + r2_v) / 2
    SS_b = 2 * ((sm - gm) ** 2).sum(0)
    SS_r = n * ((r1_v.mean(0) - gm) ** 2 + (r2_v.mean(0) - gm) ** 2)
    SS_tot = ((r1_v - gm) ** 2).sum(0) + ((r2_v - gm) ** 2).sum(0)
    MS_b = SS_b / (n - 1)
    MS_e = (SS_tot - SS_b - SS_r) / (n - 1)
    denom = MS_b + MS_e
    icc3 = np.where(denom > 0, (MS_b - MS_e) / denom, 0.0)
    stable = [f for f, c in zip(valid_feats, icc3) if c >= threshold]
    print(f"ICC >= {threshold}: {len(stable)} / {len(features)} features kept.")
    return df[stable]


# --- Fold-specific preprocessing and feature selection -----------------------------------------------

# Unchanged from main_pipeline.py -- outcome-agnostic.
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


# Regression counterpart of calculate_feature_metrics: Spearman correlation instead of Mann-Whitney
# U + AUC. Spearman (not Pearson) to stay consistent with the rest of this pipeline's correlation
# machinery (decorrelation is already Spearman-based) and because it doesn't assume a linear
# relationship or normally-distributed radiomics features.
def calculate_feature_metrics(X, y):
    metrics = []
    for col in X.columns:
        rho, p = spearmanr(X[col], y)
        metrics.append({'Feature': col, 'p_value': p, 'Spearman_rho': rho})
    metrics_df = pd.DataFrame(metrics).set_index('Feature')
    metrics_df['p_value_fdr'] = benjamini_hochberg(metrics_df['p_value'])
    return metrics_df


# Unchanged from main_pipeline.py -- outcome-agnostic (works on any numeric DataFrame).
def spearman_filter(df, threshold, tie_breaker_series):
    corr_matrix = df.corr(method='spearman').abs()
    to_drop = set()
    cols = df.columns
    print("Pruning redundant features...")
    for i in tqdm(range(len(cols)), desc="Spearman Filter"):
        for j in range(i + 1, len(cols)):
            f1, f2 = cols[i], cols[j]
            if f1 not in to_drop and f2 not in to_drop:
                if corr_matrix.loc[f1, f2] > threshold:
                    if tie_breaker_series is not None:
                        if tie_breaker_series[f1] <= tie_breaker_series[f2]:
                            to_drop.add(f2)
                        else:
                            to_drop.add(f1)
                    else:
                        to_drop.add(f2)
    return df.drop(columns=list(to_drop))


def _safe_stratified_splits(X, y, n_splits, random_state):
    """StratifiedKFold splits on y's rounded integer level, floored at 2 splits/2 samples-per-class
    the way main_pipeline.py's tune_model clamps to the minority class count -- but TILs=0 can be as
    few as a handful of patients dev-wide, so this floors more defensively than a plain min() would."""
    y_int = y.round().astype(int)
    min_class = y_int.value_counts().min()
    safe_n = max(min(n_splits, min_class), 2)
    skf = StratifiedKFold(n_splits=safe_n, shuffle=True, random_state=random_state)
    return list(skf.split(X, y_int))


# Regression counterpart of run_fold_pruning: Spearman effect-size screen -> FDR within the screened
# pool -> Spearman decorrelation -> LassoCV consensus selection (in place of LogisticRegressionCV L1).
def run_fold_pruning(X_tr, y_tr, target_count, p_value_cutoff, metric_cutoff,
                      decorrelation_threshold=0.85, random_state=RANDOM_STATE):
    imputer = KNNImputer(n_neighbors=5)
    X_imp = pd.DataFrame(imputer.fit_transform(X_tr), columns=X_tr.columns, index=X_tr.index)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imp), columns=X_imp.columns, index=X_imp.index)

    scorecard = calculate_feature_metrics(X_scaled, y_tr)
    total_features = len(scorecard)

    # |Spearman rho| >= metric_cutoff. Symmetric by construction (unlike the classification pipeline's
    # two-sided AUC>=c OR <=1-c formulation, which needed two branches to express the same idea).
    mask_kept = scorecard['Spearman_rho'].abs() >= metric_cutoff
    print(f"Filter: Keeping |Spearman rho| >= {metric_cutoff}")

    screened = scorecard[mask_kept].copy()
    screened['p_value_fdr_screened'] = benjamini_hochberg(screened['p_value'])
    mask_sig = screened['p_value_fdr_screened'] < p_value_cutoff
    kept_features_info = screened[mask_sig]

    print(f"Effect-size screen: {len(screened)} of {total_features} pass.")
    print(f"FDR within screened pool: {mask_sig.sum()} of {len(screened)} pass q<{p_value_cutoff}.")

    if len(kept_features_info) == 0:
        raise RuntimeError(
            f"No features survived the univariate filter for this fold (0 of {total_features}). "
            f"{len(screened)} passed the effect-size filter, but none passed FDR q<{p_value_cutoff} "
            f"within the screened pool. Loosen --p-value-cutoff and/or --metric-cutoff and re-run."
        )

    X_filtered = X_scaled[kept_features_info.index]
    initial_n = X_filtered.shape[1]
    p_vals = scorecard.loc[X_filtered.columns, 'p_value']
    X_spearman = spearman_filter(X_filtered, threshold=decorrelation_threshold, tie_breaker_series=p_vals)
    final_n = X_spearman.shape[1]
    print(f"Spearman decorrelation: Dropped {initial_n - final_n} features (from {initial_n} to {final_n})")

    # LassoCV: regression L1, playing the same role LogisticRegressionCV(penalty='l1') plays in the
    # classification pipeline. Its own internal path optimizes (negative) MSE -- it doesn't take an
    # arbitrary `scoring=`, unlike LogisticRegressionCV -- so the CV-selected alpha is MSE-optimal,
    # then features are ranked/capped by |coefficient| exactly as in the classification pipeline.
    inner_splits = _safe_stratified_splits(X_spearman, y_tr, 5, random_state)
    lasso_cv = LassoCV(cv=inner_splits, max_iter=10000, n_alphas=100, random_state=random_state)
    lasso_cv.fit(X_spearman, y_tr)

    coefs = pd.Series(lasso_cv.coef_, index=X_spearman.columns)
    nonzero_coefs = coefs[coefs != 0]
    ranked_by_magnitude = nonzero_coefs.abs().sort_values(ascending=False)
    selected = nonzero_coefs.loc[ranked_by_magnitude.head(target_count).index]
    print(f"LassoCV: Dropped {len(coefs) - len(selected)} features (from {len(coefs)} to {len(selected)}), "
          f"alpha={lasso_cv.alpha_:.5f}")
    return selected, imputer, scaler, X_spearman.columns, lasso_cv, scorecard


def stratified_resample(X, y, frac, rng, replace=False):
    """Class-stratified resample -- ported from main_pipeline.py's identical helper. Stratifies on
    y's rounded integer TILs level (0/1/2/3, same convention as _safe_stratified_splits), resampling
    each level independently to `frac` of its own size (floored at 2) so every resample's class
    proportions are an exact, invariant function of the full dev set's proportions -- not something
    that merely averages out over many resamples. See project discussion/verification for why this
    matters (empirically confirmed zero-variance class proportions across 200 classification resamples)."""
    y_int = y.round().astype(int)
    parts_X, parts_y = [], []
    for cls in y_int.unique():
        idx = y.index[y_int == cls]
        n = max(int(round(len(idx) * frac)), 2)
        n = min(n, len(idx)) if not replace else n
        drawn = rng.choice(idx, size=n, replace=replace)
        parts_X.append(X.loc[drawn])
        parts_y.append(y.loc[drawn])
    return pd.concat(parts_X), pd.concat(parts_y)


# Meinshausen-Buhlmann-style stability selection -- regression counterpart of main_pipeline.py's
# run_stability_selection. Same rationale: replaces the single, full-dev-set LassoCV pass that
# currently produces consensus_features.csv with many independent, class(TILs-level)-stratified
# resamples, each rerunning run_fold_pruning from scratch, tallying selection frequency + sign
# stability. Does not touch/replace run_cross_validation's nested-CV OOF diagnostic.
def run_stability_selection(X, y_dev, target_count, p_value_cutoff, metric_cutoff,
                             n_resamples=200, resample_frac=0.75, decorrelation_threshold=0.85,
                             random_state=RANDOM_STATE):
    rng = np.random.default_rng(random_state)
    selection_counts = pd.Series(0, index=X.columns, dtype=int)
    sign_records = {col: [] for col in X.columns}
    n_valid = 0

    for _ in tqdm(range(n_resamples), desc="Stability selection resamples"):
        X_rs, y_rs = stratified_resample(X, y_dev, resample_frac, rng, replace=False)
        try:
            selected_coefs, *_ = run_fold_pruning(
                X_rs, y_rs, target_count, p_value_cutoff, metric_cutoff=metric_cutoff,
                decorrelation_threshold=decorrelation_threshold, random_state=random_state,
            )
        except RuntimeError:
            continue
        n_valid += 1
        nonzero = selected_coefs[selected_coefs != 0]
        selection_counts.loc[nonzero.index] += 1
        for feat, coef in nonzero.items():
            sign_records[feat].append(np.sign(coef))

    if n_valid == 0:
        raise RuntimeError(
            "Stability selection: every resample failed the univariate screen -- no signal "
            "survives even under resampling. Loosen --metric-cutoff/--p-value-cutoff and re-run."
        )

    def _sign_stable(feat):
        signs = sign_records[feat]
        return bool(signs) and (all(s > 0 for s in signs) or all(s < 0 for s in signs))

    stability_df = pd.DataFrame({
        'Selection_Frequency': selection_counts / n_valid,
        'Selection_Count': selection_counts,
        'N_Resamples_Valid': n_valid,
        'Sign_Stable': [_sign_stable(f) for f in X.columns],
    })
    stability_df = stability_df[stability_df['Selection_Count'] > 0]
    stability_df = stability_df.sort_values('Selection_Frequency', ascending=False)
    return stability_df


# --- Cross-validation wrapper --------------------------------------------------------------------------

def run_cross_validation(X, y_dev, target_count, p_value_cutoff, metric_cutoff,
                          model_names=None, tuning_cv=5, decorrelation_threshold=0.85,
                          random_state=RANDOM_STATE):
    if model_names is None:
        model_names = NESTED_CV_MODELS

    outer_splits = _safe_stratified_splits(X, y_dev, 5, random_state)
    n_folds = len(outer_splits)

    fold_coefficients = pd.DataFrame(0.0, index=X.columns, columns=[f'Fold_{i+1}' for i in range(n_folds)])
    oof_predictions = pd.DataFrame(np.nan, index=X.index, columns=model_names)

    for fold, (train_idx, val_idx) in enumerate(outer_splits):
        X_tr, y_tr = X.iloc[train_idx], y_dev.iloc[train_idx]
        X_val = X.iloc[val_idx]

        try:
            selected_coefs, imputer, scaler, kept_features, lasso, scorecard = run_fold_pruning(
                X_tr, y_tr, target_count, p_value_cutoff, metric_cutoff=metric_cutoff,
                decorrelation_threshold=decorrelation_threshold, random_state=random_state,
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
        print(f"Fold {fold+1} complete. Features selected by LassoCV: {active_count}")

        fold_features = selected_coefs.index
        if len(fold_features) == 0:
            continue

        X_tr_scaled = pd.DataFrame(scaler.transform(imputer.transform(X_tr)), columns=X_tr.columns, index=X_tr.index)
        X_val_scaled = pd.DataFrame(scaler.transform(imputer.transform(X_val)), columns=X_val.columns, index=X_val.index)
        X_tr_final = X_tr_scaled[fold_features]
        X_val_final = X_val_scaled[fold_features]

        for model_name in model_names:
            model = tune_model(X_tr_final, y_tr, model_name, cv=tuning_cv, random_state=random_state)
            oof_predictions.loc[X_val.index, model_name] = model.predict(X_val_final)

    fold_cols = [f'Fold_{i+1}' for i in range(n_folds)]
    fold_coefficients['Selection_Frequency'] = (fold_coefficients[fold_cols] != 0).sum(axis=1)
    fold_coefficients['Mean_Absolute_Coefficient'] = fold_coefficients[fold_cols].abs().mean(axis=1)

    def _sign_stable(row):
        nonzero = row[fold_cols][row[fold_cols] != 0]
        if len(nonzero) == 0:
            return False
        return bool((nonzero > 0).all() or (nonzero < 0).all())
    fold_coefficients['Sign_Stable'] = fold_coefficients.apply(_sign_stable, axis=1)

    fold_coefficients = fold_coefficients[fold_coefficients['Selection_Frequency'] > 0]
    fold_coefficients = fold_coefficients.sort_values(by=['Selection_Frequency', 'Mean_Absolute_Coefficient'], ascending=False)

    dev_imputer = KNNImputer(n_neighbors=5)
    X_dev_imp = pd.DataFrame(dev_imputer.fit_transform(X), columns=X.columns, index=X.index)
    dev_scaler = StandardScaler()
    X_dev_scaled = pd.DataFrame(dev_scaler.fit_transform(X_dev_imp), columns=X.columns, index=X.index)
    global_scorecard = calculate_feature_metrics(X_dev_scaled, y_dev)

    return fold_coefficients, global_scorecard, oof_predictions, n_folds


# Regression counterpart of summarize_nested_cv: Spearman rho (primary, AUC's role) + R^2 + MAE, each
# with bootstrap CIs via the same generic bootstrap_ci used everywhere else in this project.
def summarize_nested_cv(oof_predictions, y_dev, n_bootstrap=1000, random_state=RANDOM_STATE):
    rows = []
    for model_name in oof_predictions.columns:
        pred = oof_predictions[model_name]
        valid_mask = pred.notna()
        y_valid = y_dev[valid_mask]
        pred_valid = pred[valid_mask]

        rho, _ = spearmanr(y_valid, pred_valid)
        rho_ci = bootstrap_ci(y_valid, pred_valid, _spearman_rho, n_bootstrap=n_bootstrap, seed=random_state)
        r2 = r2_score(y_valid, pred_valid)
        r2_ci = bootstrap_ci(y_valid, pred_valid, r2_score, n_bootstrap=n_bootstrap, seed=random_state)
        mae = mean_absolute_error(y_valid, pred_valid)
        mae_ci = bootstrap_ci(y_valid, pred_valid, mean_absolute_error, n_bootstrap=n_bootstrap, seed=random_state)

        rows.append({
            'model': model_name,
            'oof_spearman_rho': rho,
            'rho_ci_lower': rho_ci['ci_lower'],
            'rho_ci_upper': rho_ci['ci_upper'],
            'oof_r2': r2,
            'r2_ci_lower': r2_ci['ci_lower'],
            'r2_ci_upper': r2_ci['ci_upper'],
            'oof_mae': mae,
            'mae_ci_lower': mae_ci['ci_lower'],
            'mae_ci_upper': mae_ci['ci_upper'],
            'n_dev': int(valid_mask.sum()),
        })
    return pd.DataFrame(rows).sort_values('oof_spearman_rho', ascending=False)


# Regression counterpart of run_feature_diagnostics: dev vs. lock Spearman rho per consensus feature.
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
    print(f"\n{'Feature Name':<40} | {'Dev rho':<8} | {'Dev p':<8} | {'Lock rho':<8} | {'Lock p':<8}")
    print("-" * 85)
    for feat in consensus_features:
        try:
            dev_data = pd.concat([X_dev_diag[feat], y_dev_filtered], axis=1).dropna()
            rho_dev, p_dev = spearmanr(dev_data[feat], dev_data[y_dev_filtered.name])
            lock_data = pd.concat([X_lock_numeric[feat], y_lock], axis=1).dropna()
            rho_lock, p_lock = spearmanr(lock_data[feat], lock_data[y_lock.name])
            print(f"{feat:<40} | {rho_dev:<8.3f} | {p_dev:<8.3f} | {rho_lock:<8.3f} | {p_lock:<8.3f}")
        except Exception as e:
            print(f"{feat:<40} | Error: {e}")
    print("=" * 85)


# =====================================================================================================
# Shared model factory
# =====================================================================================================

def get_model(model_name, random_state=RANDOM_STATE):
    if model_name == 'linear':
        # Ridge, not plain OLS -- same "keep it heavily regularized on a small-n feature set" spirit
        # as the classification pipeline's logistic(C=0.01). Tuned via PARAM_GRIDS' alpha grid.
        return Ridge(alpha=10.0, random_state=random_state)
    elif model_name == 'svm_linear':
        return SVR(kernel='linear')
    elif model_name == 'svm_rbf':
        return SVR(kernel='rbf', C=10, gamma=0.01)
    elif model_name == 'rf':
        return RandomForestRegressor(n_estimators=300, max_depth=3, random_state=random_state)
    elif model_name == 'extratrees':
        return ExtraTreesRegressor(n_estimators=300, max_depth=3, random_state=random_state)
    elif model_name == 'xgboost':
        return XGBRegressor(n_estimators=100, max_depth=2, learning_rate=0.01, subsample=0.6,
                             random_state=random_state, n_jobs=1)
    elif model_name == 'gp':
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
        return GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=random_state)
    elif model_name == 'tabpfn':
        torch.set_num_threads(1)
        return TabPFNRegressor(device='cpu', random_state=random_state)
    else:
        raise ValueError(f"Model {model_name} not recognized.")


# See main_pipeline.py's PARAM_GRIDS comment -- same rationale, regression-appropriate keys. gp/tabpfn
# intentionally absent (untuned, same reasons as the classification pipeline).
PARAM_GRIDS = {
    'linear': {'alpha': [0.1, 1, 10, 100, 1000]},
    'svm_linear': {'C': [0.001, 0.01, 0.1, 1, 10]},
    'svm_rbf': {'C': [1, 10, 100, 1000], 'gamma': ['scale', 'auto', 0.01]},
    'rf': {'max_depth': [2, 3, 4, 5, 6], 'min_samples_leaf': [1, 2, 5, 10]},
    'extratrees': {'max_depth': [2, 3, 4, 5, 6], 'min_samples_leaf': [1, 2, 5, 10]},
    'xgboost': {'n_estimators': [25, 50, 100, 200, 300], 'max_depth': [1, 2, 3],
                'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3], 'subsample': [0.6, 0.8, 1.0]},
}


def tune_model(X, y, model_name, cv=5, random_state=RANDOM_STATE, n_jobs=6):
    estimator = get_model(model_name, random_state=random_state)
    if model_name not in PARAM_GRIDS:
        estimator.fit(X, y)
        return estimator
    cv_splits = _safe_stratified_splits(X, y, cv, random_state)
    gs = GridSearchCV(estimator=estimator, param_grid=PARAM_GRIDS[model_name], scoring=SPEARMAN_SCORER,
                       cv=cv_splits, n_jobs=n_jobs, refit=True)
    gs.fit(X, y)
    return gs.best_estimator_


# =====================================================================================================
# STAGE 2: Hyperparameter search
# =====================================================================================================

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


def load_saved_estimator(model_name, input_dir):
    if input_dir is None:
        return None
    estimator_path = os.path.join(input_dir, f"best_estimator_{model_name}.pkl")
    if os.path.exists(estimator_path):
        print(f"Loading saved estimator for {model_name} from: {estimator_path}")
        with open(estimator_path, 'rb') as f:
            return pickle.load(f)
    return None


def perform_grid_search(X, y, model_names=None, out_dir=None, cv=5, n_jobs=6,
                         random_state=RANDOM_STATE, save=True):
    if model_names is None:
        model_names = ALL_MODELS
    os.makedirs(out_dir, exist_ok=True)
    cv_splits = _safe_stratified_splits(X, y, cv, random_state)

    results = []
    for m in model_names:
        if m not in PARAM_GRIDS:
            print(f"Skipping unknown/untuned model: {m}")
            continue
        print(f"\nRunning grid search for: {m}")
        estimator = get_model(m, random_state=random_state)
        gs = GridSearchCV(estimator=estimator, param_grid=PARAM_GRIDS[m], scoring=SPEARMAN_SCORER,
                           cv=cv_splits, n_jobs=n_jobs, refit=True)
        gs.fit(X, y)
        print(f"Best Spearman rho for {m}: {gs.best_score_:.4f} with params: {gs.best_params_}")
        results.append({'model': m, 'best_score': gs.best_score_, 'best_params': json.dumps(gs.best_params_)})
        if save:
            with open(os.path.join(out_dir, f"best_estimator_{m}.pkl"), 'wb') as f:
                pickle.dump(gs.best_estimator_, f)
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

# Unchanged from main_pipeline.py -- fully generic (any metric_fn(y_true, y_pred) -> float).
def bootstrap_ci(y_true, y_pred, metric_fn, n_bootstrap=1000, ci=95, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    y_true_arr = y_true.values if hasattr(y_true, 'values') else np.asarray(y_true)
    y_pred_arr = y_pred.values if hasattr(y_pred, 'values') else np.asarray(y_pred)
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
        alpha = (100 - ci) / 2
        ci_lower = np.percentile(bootstrap_scores, alpha)
        ci_upper = np.percentile(bootstrap_scores, 100 - alpha)
    else:
        point_estimate = ci_lower = ci_upper = np.nan
    return {'point_estimate': point_estimate, 'ci_lower': ci_lower, 'ci_upper': ci_upper,
            'bootstrap_scores': bootstrap_scores, 'n_valid': n_valid}


def create_shap_explainer(model, X, model_name=None):
    model_type = (model_name or type(model).__name__).lower()
    if any(keyword in model_type for keyword in
           ['xgb', 'xgboost', 'booster', 'rf', 'randomforest', 'extratrees', 'extra_trees']):
        return shap.TreeExplainer(model)
    if 'linear' in model_type and hasattr(model, 'coef_') and 'svm' not in model_type:
        return shap.LinearExplainer(model, X.values if hasattr(X, 'values') else X)
    if 'svm' in model_type or 'svr' in model_type:
        if hasattr(model, 'coef_') and getattr(model, 'kernel', None) == 'linear':
            return shap.LinearExplainer(model, X.values if hasattr(X, 'values') else X)
        bg_sample = shap.sample(X, min(50, len(X)))
        bg_data = bg_sample.values if hasattr(bg_sample, 'values') else bg_sample
        return shap.KernelExplainer(model.predict, bg_data)
    if any(keyword in model_type for keyword in ['gp', 'gaussianprocess', 'tabpfn']):
        bg_sample = shap.sample(X, min(50, len(X)))
        bg_data = bg_sample.values if hasattr(bg_sample, 'values') else bg_sample
        return shap.KernelExplainer(model.predict, bg_data)
    return shap.Explainer(model, X)


def plot_shap_summary(model, X, out_dir, model_name, plot_type='bar'):
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    explainer = create_shap_explainer(model, X, model_name=model_name)
    shap_values = explainer(X)
    plt.figure()
    shap.summary_plot(shap_values, X, plot_type=plot_type, show=False)
    ax = plt.gca()
    ax.set_xlabel("SHAP value", fontsize=12, labelpad=12)
    plt.tight_layout()
    filename = f'{model_name}_shap_summary_{plot_type}.png'
    path = os.path.join(out_dir, filename)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    return path


# Regression counterpart of create_roc_comparison_plot: predicted-vs-actual scatter, one panel per
# model, annotated with lock Spearman rho -- plays the same "one glance, compare every model" role.
def create_prediction_scatter_plot(results_list, out_dir, filename='model_prediction_scatter.png'):
    successful = [r for r in results_list if r.get('status') == 'success']
    if not successful:
        return None
    n = len(successful)
    ncols = min(4, n)
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)
    for i, r in enumerate(successful):
        ax = axes[i // ncols][i % ncols]
        y_true, y_pred = r['y_lock'], r['y_pred']
        ax.scatter(y_true, y_pred, alpha=0.7, edgecolor='black', linewidth=0.5)
        lims = [min(y_true.min(), y_pred.min()) - 0.3, max(y_true.max(), y_pred.max()) + 0.3]
        ax.plot(lims, lims, 'k--', linewidth=1, alpha=0.6)
        ax.set_title(f"{r['model_name']}\nrho={r['lock_rho']:.2f}, R2={r['lock_r2']:.2f}, MAE={r['lock_mae']:.2f}",
                     fontsize=10)
        ax.set_xlabel("Actual TILs_resection")
        ax.set_ylabel("Predicted")
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis('off')
    plt.tight_layout()
    path = os.path.join(out_dir, filename)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    return path


def train_and_evaluate_model(X_dev, y_dev, X_lock, y_lock, model_name, out_dir, param_dir,
                              random_state=RANDOM_STATE, n_bootstrap=1000, generate_shap=False):
    print(f"\n{'='*60}\nTraining: {model_name.upper()}\n{'='*60}")
    try:
        model = load_saved_estimator(model_name, param_dir)
        if model is not None:
            model_source = 'gridsearch_pickle'
            print(f"Using pickled model for {model_name} from param_dir.")
        else:
            model_source = 'default_fallback'
            model = get_model(model_name, random_state=random_state)
            model.fit(X_dev, y_dev)

        y_pred = pd.Series(model.predict(X_lock), index=X_lock.index)

        rho, _ = spearmanr(y_lock, y_pred)
        rho_ci = bootstrap_ci(y_lock, y_pred, _spearman_rho, n_bootstrap=n_bootstrap, seed=random_state)
        r2 = r2_score(y_lock, y_pred)
        r2_ci = bootstrap_ci(y_lock, y_pred, r2_score, n_bootstrap=n_bootstrap, seed=random_state)
        mae = mean_absolute_error(y_lock, y_pred)
        mae_ci = bootstrap_ci(y_lock, y_pred, mean_absolute_error, n_bootstrap=n_bootstrap, seed=random_state)

        results = {
            'model_name': model_name, 'model': model, 'model_source': model_source,
            'lock_rho': rho, 'rho_ci_lower': rho_ci['ci_lower'], 'rho_ci_upper': rho_ci['ci_upper'],
            'lock_r2': r2, 'r2_ci_lower': r2_ci['ci_lower'], 'r2_ci_upper': r2_ci['ci_upper'],
            'lock_mae': mae, 'mae_ci_lower': mae_ci['ci_lower'], 'mae_ci_upper': mae_ci['ci_upper'],
            'y_pred': y_pred, 'y_lock': y_lock,
            'n_dev_samples': len(X_dev), 'n_lock_samples': len(X_lock), 'status': 'success',
        }
        print(f"Lock Spearman rho:  {rho:.2f} (95% CI: {rho_ci['ci_lower']:.2f} - {rho_ci['ci_upper']:.2f})")
        print(f"Lock R^2:           {r2:.2f} (95% CI: {r2_ci['ci_lower']:.2f} - {r2_ci['ci_upper']:.2f})")
        print(f"Lock MAE:           {mae:.2f} (95% CI: {mae_ci['ci_lower']:.2f} - {mae_ci['ci_upper']:.2f})")
        print(f"  Dev Samples: {len(X_dev)}  Lock Samples: {len(X_lock)}  Model source: {model_source}")

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
        return {'model_name': model_name, 'status': 'failed', 'error': str(e)}


def train_multiple_models(in_dir, param_dir, out_dir, model_list=None, n_bootstrap=1000,
                           random_state=RANDOM_STATE, generate_shap=True):
    if model_list is None:
        model_list = ALL_MODELS
    X_dev, X_lock, y_dev, y_lock = load_preprocessed_data(in_dir, include_lock=True)

    results_list = []
    formatted_results = []
    model_objects = {}
    for model_name in model_list:
        result = train_and_evaluate_model(
            X_dev, y_dev, X_lock, y_lock, model_name, out_dir, param_dir=param_dir,
            random_state=random_state, n_bootstrap=n_bootstrap, generate_shap=generate_shap,
        )
        results_list.append(result)
        if result['status'] == 'success':
            formatted_results.append({
                'Model': model_name, 'Model Source': result['model_source'],
                'Spearman_rho': f"{result['lock_rho']:.2f}",
                'rho 95% CI': f"({result['rho_ci_lower']:.2f}, {result['rho_ci_upper']:.2f})",
                'R2': f"{result['lock_r2']:.2f}",
                'R2 95% CI': f"({result['r2_ci_lower']:.2f}, {result['r2_ci_upper']:.2f})",
                'MAE': f"{result['lock_mae']:.2f}",
                'MAE 95% CI': f"({result['mae_ci_lower']:.2f}, {result['mae_ci_upper']:.2f})",
                'N Dev': result['n_dev_samples'], 'N Lock': result['n_lock_samples'],
            })
            model_objects[model_name] = result['model']
        else:
            formatted_results.append({
                'Model': model_name, 'Model Source': '-', 'Spearman_rho': '-', 'rho 95% CI': '-',
                'R2': '-', 'R2 95% CI': '-', 'MAE': '-', 'MAE 95% CI': '-',
                'Error': result.get('error', 'Unknown'), 'N Dev': '-', 'N Lock': '-',
            })

    comparison_df = pd.DataFrame(formatted_results)
    create_prediction_scatter_plot(results_list, out_dir)

    print(f"\n{'='*100}\nMODEL COMPARISON SUMMARY (with 95% Bootstrap Confidence Intervals)\n{'='*100}")
    print(comparison_df.to_string(index=False))
    print(f"{'='*100}\n")
    comparison_df.to_csv(f"{out_dir}/model_comparison.csv", index=False)
    print(f"Comparison saved to: {out_dir}/model_comparison.csv")

    successful = [r for r in results_list if r.get('status') == 'success']
    if successful:
        best = max(successful, key=lambda r: r['lock_rho'])
        best_model = model_objects[best['model_name']]
        with open(f"{out_dir}/best_model.pkl", 'wb') as f:
            pickle.dump(best_model, f)
        print(f"Best model ({best['model_name']}) saved: {best['lock_rho']:.3f} Spearman rho")

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
        df_clinical_feats = load_clinical_features(args.clinical_data_path)
        df_filtered = df_filtered.join(df_clinical_feats, how='left')
        print(f"Joined clinical features. Feature matrix now has {df_filtered.shape[1]} columns.")

    X_dev, y_dev, X_lock, y_lock, df_clinical_dev, df_clinical_lock = partition_with_constraints(
        df_filtered, args.outcomes_path, args.icc_path, clinical_data_path=args.clinical_data_path,
        label_col=args.label_col, test_size=args.test_size, output_dir=args.output_dir,
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
        X_dev_clean_icc = pd.concat([X_dev_clean_icc, X_dev_clean[clinical_cols]], axis=1)
        print(f"Clinical features bypass ICC filter, added back: {len(clinical_cols)}")

    y_dev_filtered = y_dev.loc[X_dev_clean_icc.index]
    print(f"After filtering: X_dev has {X_dev_clean_icc.shape[0]} samples, y_dev has {len(y_dev_filtered)} samples")

    # Nested-CV OOF diagnostic: unaffected by --feature-selection-method -- always reselects fresh
    # per outer fold, so it stays the honest reference number regardless of which method built the
    # final consensus_features.
    fold_coefficients, global_scorecard, oof_predictions, n_folds = run_cross_validation(
        X_dev_clean_icc, y_dev_filtered, target_count=args.lasso_target_count,
        p_value_cutoff=args.p_value_cutoff, metric_cutoff=args.metric_cutoff,
        decorrelation_threshold=args.decorrelation_threshold, random_state=RANDOM_STATE,
    )
    fold_coefficients.to_csv(f"{args.output_dir}/lasso_coeffs_fold.csv", index=True)

    if args.feature_selection_method == 'stability':
        stability_cache_path = f"{args.output_dir}/stability_selection.csv"
        if os.path.exists(stability_cache_path) and not args.stability_force_recompute:
            print(f"Reusing cached stability selection frequencies from: {stability_cache_path}")
            stability_df = pd.read_csv(stability_cache_path, index_col=0)
        else:
            stability_df = run_stability_selection(
                X_dev_clean_icc, y_dev_filtered, target_count=args.lasso_target_count,
                p_value_cutoff=args.p_value_cutoff, metric_cutoff=args.metric_cutoff,
                n_resamples=args.stability_n_resamples, resample_frac=args.stability_resample_frac,
                decorrelation_threshold=args.decorrelation_threshold, random_state=RANDOM_STATE,
            )
            stability_df.to_csv(stability_cache_path, index=True)
        frequent_features = stability_df[
            (stability_df['Selection_Frequency'] >= args.stability_threshold) & stability_df['Sign_Stable']
        ]
        consensus_features = frequent_features.head(args.target_feature_count).index.tolist()
        selection_error = (
            f"No consensus features survived stability selection: 0 features were selected in "
            f">= {args.stability_threshold:.0%} of {args.stability_n_resamples} resamples with a "
            f"stable coefficient sign (see stability_selection.csv). Loosen --stability-threshold "
            f"and/or re-run."
        )
    else:
        frequent_features = fold_coefficients[
            (fold_coefficients['Selection_Frequency'] >= min(args.min_fold_n, n_folds)) & fold_coefficients['Sign_Stable']
        ]
        consensus_features = frequent_features.head(args.target_feature_count).index.tolist()
        selection_error = (
            f"No consensus features survived selection: 0 features were selected by LassoCV in "
            f">= --min-fold-n={args.min_fold_n} of the {n_folds} outer folds with a stable coefficient "
            f"sign. Each fold's own selection was non-empty (see lasso_coeffs_fold.csv), they just "
            f"didn't overlap enough across folds. Loosen --min-fold-n and/or re-run."
        )

    if len(consensus_features) == 0:
        raise RuntimeError(selection_error)

    nested_cv_summary = summarize_nested_cv(oof_predictions, y_dev_filtered, random_state=RANDOM_STATE)
    nested_cv_summary.to_csv(f"{args.output_dir}/nested_cv_diagnostics.csv", index=False)

    print(f"\n{'='*100}\nDEV-ONLY NESTED CV DIAGNOSTICS (honest, leakage-free -- does not touch the lock set)\n{'='*100}")
    print(nested_cv_summary.to_string(index=False))
    print(f"{'='*100}\n")
    print(f"Consensus features identified: {len(consensus_features)}")
    print(consensus_features)
    global_scorecard.to_csv(f"{args.output_dir}/univariate_metrics.csv")

    run_feature_diagnostics(X_dev_clean_icc, y_dev_filtered, X_lock, y_lock, consensus_features, args.output_dir)

    imputer = KNNImputer(n_neighbors=5)
    scaler = StandardScaler()
    X_dev_subset = X_dev_clean_icc[consensus_features]
    X_lock_subset = X_lock[consensus_features]
    X_dev_impute = pd.DataFrame(imputer.fit_transform(X_dev_subset), columns=consensus_features, index=X_dev_subset.index)
    X_dev_final = pd.DataFrame(scaler.fit_transform(X_dev_impute), columns=consensus_features, index=X_dev_subset.index)
    X_lock_impute = pd.DataFrame(imputer.transform(X_lock_subset), columns=consensus_features, index=X_lock.index)
    X_lock_final = pd.DataFrame(scaler.transform(X_lock_impute), columns=consensus_features, index=X_lock.index)

    X_dev_final.to_csv(f"{args.output_dir}/X_dev_processed.csv")
    X_lock_final.to_csv(f"{args.output_dir}/X_lock_processed.csv")
    pd.Series(y_dev_filtered, name='y_dev').to_csv(f"{args.output_dir}/y_dev.csv")
    pd.Series(y_lock, name='y_lock').to_csv(f"{args.output_dir}/y_lock.csv")
    pd.Series(consensus_features, name='feature').to_csv(f"{args.output_dir}/consensus_features.csv", index=False)

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
    results_df = perform_grid_search(X_dev, y_dev, model_names=args.models, out_dir=args.output_dir,
                                      cv=args.cv, n_jobs=args.n_jobs, random_state=RANDOM_STATE, save=True)
    print(results_df)


def run_train_stage(args):
    os.makedirs(args.output_dir, exist_ok=True)
    comparison_df, _ = train_multiple_models(
        in_dir=args.input_data_dir, param_dir=args.input_params_dir, out_dir=args.output_dir,
        model_list=args.models, n_bootstrap=args.bootstrap, random_state=RANDOM_STATE,
        generate_shap=args.shap,
    )
    print(comparison_df)


# =====================================================================================================
# CLI
# =====================================================================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Radiomics regression pipeline: TILs_resection (0-3) as a continuous target.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--stage', choices=['partition', 'gridsearch', 'train'], default=DEFAULT_STAGE)
    parser.add_argument('--sequences', nargs='+', default=['pvp', 'tp', 'hbp', 'dwi_b1', 'adc'])
    parser.add_argument('--exclude-feature-families', nargs='*', default=['wavelet'])
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
    parser.add_argument('--icc-path', type=str, default=None)
    parser.add_argument('--label-col', type=str, default='TILs_resection')
    parser.add_argument('--test-size', type=int, default=30)
    parser.add_argument('--cv-threshold', type=float, default=0.1)
    parser.add_argument('--icc-threshold', type=float, default=0.85)
    parser.add_argument('--decorrelation-threshold', type=float, default=0.85,
        help="Absolute Spearman correlation above which spearman_filter drops one of a pair of "
             "features. See main_pipeline.py's identical flag/rationale.")
    parser.add_argument('--lasso-target-count', type=int, default=100)
    parser.add_argument('--p-value-cutoff', type=float, default=0.2,
        help="FDR q-value threshold within the effect-size-screened pool. Same semantics as main_pipeline.py.")
    parser.add_argument('--metric-cutoff', type=float, default=0.25,
        help="Minimum |Spearman rho| for a feature to pass the effect-size screen. Not a literal "
             "equivalent of the classification pipeline's AUC>=0.65 (different metric family) -- "
             "chosen as a defensible small-moderate-effect threshold; adjust and re-run to check "
             "sensitivity to this choice, same as main_pipeline.py's --metric-cutoff.")
    parser.add_argument('--min-fold-n', type=int, default=3)
    parser.add_argument('--target-feature-count', type=int, default=10)
    parser.add_argument('--feature-selection-method', choices=['fold-consensus', 'stability'],
        default='fold-consensus',
        help="'fold-consensus' (default, unchanged): consensus_features chosen from the same "
             "outer-fold split as the nested-CV OOF diagnostic. 'stability': Meinshausen-Buhlmann "
             "stability selection over many more (see --stability-n-resamples) TILs-level-stratified "
             "resamples of the whole dev set -- see main_pipeline.py's identical flag/rationale.")
    parser.add_argument('--stability-n-resamples', type=int, default=200)
    parser.add_argument('--stability-resample-frac', type=float, default=0.75)
    parser.add_argument('--stability-threshold', type=float, default=0.7)
    parser.add_argument('--stability-force-recompute', action='store_true',
        help="Recompute stability_selection.csv even if one already exists in --output-dir.")
    parser.add_argument('--compare-variables', nargs='+', default=[
        'Age', 'Gender', 'BMI', 'Cirrhosis', 'Capsule (on PVP yes/no)', 'Mosaic appearance (yes/no)',
        'Macrovascular invasion (yes/no)', 'Maximum tumor size on T1 portal venous phase (PVP)',
        'TILs_resection', 'TLS_resection',
    ])
    parser.add_argument('--compare-outcome-col', type=str, default='TILs_resection')
    parser.add_argument('--clinical-output-dir', type=str, default=None)
    parser.add_argument('--feature-set', choices=['radiomics', 'both'], default='radiomics')

    parser.add_argument('--input-dir', type=str, default=None)
    parser.add_argument('--cv', type=int, default=5)
    parser.add_argument('--n-jobs', type=int, default=6)

    parser.add_argument('--input-data-dir', type=str, default=None)
    parser.add_argument('--input-params-dir', type=str, default=None)
    parser.add_argument('--bootstrap', type=int, default=1000)
    parser.add_argument('--shap', action='store_true')

    parser.add_argument('--models', nargs='+', default=None)
    parser.add_argument('--output-dir', type=str, default=None)

    args = parser.parse_args()
    if args.icc_path is None:
        args.icc_path = args.path_r1

    if args.output_dir is None:
        base = f"tils_regression_pipeline_output/{'_'.join(args.sequences)}"
        args.output_dir = {
            'partition': f"{base}/clean_partition_select",
            'gridsearch': f"{base}/optimized_params",
            'train': f"{base}/trained_model_outputs",
        }[args.stage]

    if args.stage == 'gridsearch' and args.input_dir is None:
        args.input_dir = f"tils_regression_pipeline_output/{'_'.join(args.sequences)}/clean_partition_select"
    if args.stage == 'train':
        if args.input_data_dir is None:
            args.input_data_dir = f"tils_regression_pipeline_output/{'_'.join(args.sequences)}/clean_partition_select"
        if args.input_params_dir is None:
            args.input_params_dir = f"tils_regression_pipeline_output/{'_'.join(args.sequences)}/optimized_params"

    return args


if __name__ == "__main__":
    args = parse_arguments()
    if args.stage == 'partition':
        run_partition_stage(args)
    elif args.stage == 'gridsearch':
        run_gridsearch_stage(args)
    elif args.stage == 'train':
        run_train_stage(args)
