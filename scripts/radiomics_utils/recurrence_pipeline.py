"""
Recurrence survival pipeline: LASSO-Cox + Random Survival Forest for time-to-recurrence.

Stages (run one at a time via --stage):
  --stage partition  : sequence preselection, dev/lock split, global cleanup, ICC filtering,
                       nested-CV consensus feature selection, save processed artifacts.
  --stage gridsearch : tune RSF hyperparameters on dev set via CV, pickle best estimator.
                       (LASSO-Cox alpha is tuned by inner CV in the train stage — no grid needed.)
  --stage train      : load artifacts, fit LASSO-Cox and RSF on dev, evaluate on lock set
                       (C-index + bootstrap CI, time-dependent AUC at 1/2/3/5yr, KM curves).

Example:
  python recurrence_pipeline.py --stage partition
  python recurrence_pipeline.py --stage gridsearch
  python recurrence_pipeline.py --stage train --bootstrap 1000
"""

import os
import json
import pickle
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from scipy.stats import chi2_contingency, mannwhitneyu
from pingouin import intraclass_corr

from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, KFold, GridSearchCV, train_test_split

from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc

from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

# --- Constants -------------------------------------------------------------------
RANDOM_STATE   = 42
PARTITION_SEED = 997

# Time-dependent AUC evaluation timepoints (days)
TD_AUC_TIMES  = [365, 730, 1095, 1825]
TD_AUC_LABELS = ['1yr', '2yr', '3yr', '5yr']

ALL_MODELS   = ['lasso_cox', 'rsf']
DEFAULT_STAGE = 'partition'

# RSF is the only model with a meaningful external grid; LASSO-Cox alpha is handled by inner CV.
PARAM_GRIDS_SURVIVAL = {
    'rsf': {
        'min_samples_split': [5, 10, 20],
        'min_samples_leaf':  [3, 5, 10, 15],
        'max_features':      ['sqrt', 0.3, 0.5],
    }
}


# ================================================================================
# Utility helpers
# ================================================================================

def make_survival_array(event_series, time_series):
    """Pack event + time into a sksurv structured array."""
    return np.array(
        list(zip(event_series.astype(bool), time_series.astype(float))),
        dtype=[('Status', '?'), ('Time', '<f8')]
    )


def preselect_sequences(df, allowed_sequences, id_col='PatientID'):
    if not allowed_sequences:
        print("No sequence filter applied. Keeping all features.")
        return df
    cols = [id_col] if id_col in df.columns else []
    for col in df.columns:
        if any(seq.lower() in col.lower() for seq in allowed_sequences):
            cols.append(col)
    filtered = df[cols]
    print(f"Preselection: {df.shape[1]} → {filtered.shape[1]} features ({allowed_sequences})")
    return filtered


def global_cleanup(df, cv_threshold, id_col='PatientID'):
    if id_col in df.columns:
        df = df.set_index(id_col)
    df = df.select_dtypes(include=[np.number])
    df = df.replace([np.inf, -np.inf], np.nan)

    n0 = len(df)
    df = df.dropna(how='all')
    if len(df) < n0:
        print(f"Dropped {n0 - len(df)} fully-empty rows.")

    thresh = len(df) * 0.9
    df = df.dropna(axis=1, thresh=thresh)

    std = df.std()
    mean_abs = df.mean().abs()
    cv = std / mean_abs.replace(0, np.nan)
    keep = (std > 1e-8) & ((cv > cv_threshold) | cv.isna())
    df = df.loc[:, keep]

    print(f"Cleanup done. Columns remaining: {df.shape[1]}")
    return df


def interobserver_stability(df, path_r1, path_r2, threshold, id_col='PatientID'):
    df_r1 = pd.read_csv(path_r1)
    df_r2 = pd.read_csv(path_r2)

    common_ids = np.intersect1d(df_r1[id_col], df_r2[id_col])
    r1 = df_r1.set_index(id_col).loc[common_ids].sort_index()
    r2 = df_r2.set_index(id_col).loc[common_ids].sort_index()

    features = [f for f in df.columns
                if f in r1.columns and f in r2.columns and r1[f].dtype != 'object']
    r1, r2 = r1[features], r2[features]
    print(f"ICC: {len(features)} features across {len(common_ids)} patients...")

    stable = []
    for feat in tqdm(features, desc="ICC", unit="feature"):
        data = pd.DataFrame({
            'targets': list(common_ids) * 2,
            'raters':  ['R1'] * len(common_ids) + ['R2'] * len(common_ids),
            'values':  pd.concat([r1[feat], r2[feat]])
        })
        try:
            icc = intraclass_corr(data=data, targets='targets', raters='raters', ratings='values')
            if icc.set_index('Type').loc['ICC3', 'ICC'] >= threshold:
                stable.append(feat)
        except Exception:
            continue

    print(f"ICC ≥ {threshold}: {len(stable)} of {len(features)} features kept.")
    return df[stable]


# ================================================================================
# Stage 1: Partition
# ================================================================================

def partition_with_constraints_survival(df_filtered, outcomes_path, icc_path,
                                        clinical_data_path, event_col, time_col,
                                        test_size, output_dir, random_state=PARTITION_SEED):
    df_outcomes = pd.read_excel(outcomes_path)
    df_icc      = pd.read_csv(icc_path)

    df_merged = (df_filtered
                 .reset_index()
                 .merge(df_outcomes[['PatientID', event_col, time_col]], on='PatientID', how='inner')
                 .set_index('PatientID'))

    n0 = len(df_merged)
    df_merged = df_merged.dropna(subset=[event_col, time_col])
    print(f"Dropped {n0 - len(df_merged)} patients with missing survival data.")

    icc_ids   = df_icc['PatientID'].unique()
    df_must   = df_merged[df_merged.index.isin(icc_ids)]
    df_rest   = df_merged[~df_merged.index.isin(icc_ids)]

    X_rest        = df_rest.drop(columns=[event_col, time_col])
    event_rest    = df_rest[event_col]

    X_dev_rest, X_lock, _, _ = train_test_split(
        X_rest, event_rest,
        test_size=test_size, stratify=event_rest, random_state=random_state
    )

    X_dev  = pd.concat([df_must.drop(columns=[event_col, time_col]), X_dev_rest])
    y_dev  = df_merged.loc[X_dev.index,  [event_col, time_col]]
    y_lock = df_merged.loc[X_lock.index, [event_col, time_col]]

    df_clinical = pd.read_excel(clinical_data_path)
    df_clinical[df_clinical['PatientID'].isin(X_dev.index)].to_csv(
        f"{output_dir}/clinical_metadata_dev.csv", index=False)
    df_clinical[df_clinical['PatientID'].isin(X_lock.index)].to_csv(
        f"{output_dir}/clinical_metadata_lock.csv", index=False)

    n_events_dev  = int(y_dev[event_col].sum())
    n_events_lock = int(y_lock[event_col].sum())
    print(f"Dev:  {len(X_dev)}  ({len(df_must)} ICC-constrained, {n_events_dev} events)")
    print(f"Lock: {len(X_lock)} ({n_events_lock} events)")
    return X_dev, y_dev, X_lock, y_lock


def compare_dev_lock_survival(y_dev, y_lock, event_col, time_col, output_dir):
    """Basic summary table comparing dev and lock survival outcomes."""
    rows = []
    for label, y in [('Dev', y_dev), ('Lock', y_lock)]:
        rows.append({
            'Set': label,
            'N': len(y),
            'Events': int(y[event_col].sum()),
            'Event rate (%)': f"{100 * y[event_col].mean():.1f}",
            'Median follow-up (days)': f"{y[time_col].median():.0f}",
        })
    df = pd.DataFrame(rows)
    df.to_csv(f"{output_dir}/dev_lock_survival_summary.csv", index=False)
    print(df.to_string(index=False))
    return df


# ================================================================================
# Fold-level feature selection (survival)
# ================================================================================

def calculate_survival_feature_metrics(X, y_struct):
    """Univariate Cox C-index for each feature. Returns DataFrame indexed by feature."""
    metrics = []
    skipped = 0
    for col in X.columns:
        try:
            cox = CoxPHSurvivalAnalysis(tol=1e-3)
            cox.fit(X[[col]], y_struct)
            risk = cox.predict(X[[col]])
            c, _, _, _, _ = concordance_index_censored(y_struct['Status'], y_struct['Time'], risk)
            hr = float(np.exp(cox.coef_[0]))
            if np.isnan(c) or hr > 1e4 or hr < 1e-4:
                skipped += 1
                continue
            metrics.append({'Feature': col, 'C_index': c, 'Hazard_Ratio': hr})
        except Exception:
            skipped += 1
    if skipped:
        print(f"  Univariate Cox: {skipped} features skipped (fit failure / numerical issue).")
    if not metrics:
        raise RuntimeError("All features failed univariate Cox. Check event count and feature variance.")
    return pd.DataFrame(metrics).set_index('Feature')


def spearman_filter(df, threshold, tie_breaker_series):
    corr = df.corr(method='spearman').abs()
    to_drop = set()
    cols = df.columns
    print("Pruning correlated features...")
    for i in tqdm(range(len(cols)), desc="Spearman"):
        for j in range(i + 1, len(cols)):
            f1, f2 = cols[i], cols[j]
            if f1 not in to_drop and f2 not in to_drop:
                if corr.loc[f1, f2] > threshold:
                    if tie_breaker_series is not None:
                        # Keep the feature with higher C-index
                        if tie_breaker_series[f1] >= tie_breaker_series[f2]:
                            to_drop.add(f2)
                        else:
                            to_drop.add(f1)
                    else:
                        to_drop.add(f2)
    return df.drop(columns=list(to_drop))


def _select_lasso_cox_alpha(X, y_struct, alphas, n_inner=3, random_state=RANDOM_STATE):
    """Inner CV to select the best LASSO-Cox regularisation strength."""
    kf = KFold(n_splits=n_inner, shuffle=True, random_state=random_state)
    best_alpha, best_score = alphas[0], -np.inf
    for alpha in alphas:
        scores = []
        for tr_idx, va_idx in kf.split(X):
            try:
                m = CoxnetSurvivalAnalysis(alphas=[alpha], l1_ratio=1.0, max_iter=1000)
                m.fit(X.iloc[tr_idx], y_struct[tr_idx])
                scores.append(m.score(X.iloc[va_idx], y_struct[va_idx]))
            except Exception:
                pass
        if scores and np.mean(scores) > best_score:
            best_score = np.mean(scores)
            best_alpha = alpha
    return best_alpha


def run_fold_pruning_survival(X_tr, y_tr_struct, target_count,
                               c_index_cutoff=0.6, random_state=RANDOM_STATE):
    """
    Per-fold feature selection:
      impute → scale → univariate Cox C-index filter → Spearman → LASSO-Cox (inner CV alpha).
    Returns selected feature names + fit preprocessing objects.
    """
    imputer = KNNImputer(n_neighbors=5)
    X_imp = pd.DataFrame(imputer.fit_transform(X_tr), columns=X_tr.columns, index=X_tr.index)
    scaler = StandardScaler()
    X_sc  = pd.DataFrame(scaler.fit_transform(X_imp), columns=X_imp.columns, index=X_imp.index)

    scorecard = calculate_survival_feature_metrics(X_sc, y_tr_struct)

    # Two-sided C-index filter: keep if C-index > cutoff OR < (1 - cutoff)
    lower = 1.0 - c_index_cutoff
    mask  = (scorecard['C_index'] >= c_index_cutoff) | (scorecard['C_index'] <= lower)
    kept  = scorecard[mask]
    print(f"  Univariate C-index filter (≥{c_index_cutoff} or ≤{lower:.2f}): "
          f"{len(kept)} / {len(scorecard)} features pass.")

    if len(kept) == 0:
        raise RuntimeError(
            f"No features passed the C-index filter (threshold={c_index_cutoff}). "
            "Lower --c-index-cutoff and re-run."
        )

    X_filt = X_sc[kept.index]
    n0 = X_filt.shape[1]
    X_sp = spearman_filter(X_filt, threshold=0.85, tie_breaker_series=scorecard.loc[kept.index, 'C_index'])
    print(f"  Spearman: {n0} → {X_sp.shape[1]} features.")

    alphas = np.logspace(-4, 1, 20)
    best_alpha = _select_lasso_cox_alpha(X_sp, y_tr_struct, alphas, random_state=random_state)

    lasso = CoxnetSurvivalAnalysis(alphas=[best_alpha], l1_ratio=1.0, max_iter=1000)
    lasso.fit(X_sp, y_tr_struct)

    coefs = pd.Series(lasso.coef_[:, 0], index=X_sp.columns)
    nonzero = coefs[coefs != 0]
    # Fall back to top-by-magnitude if LASSO zeros everything (very small datasets)
    if len(nonzero) == 0:
        print("  LASSO zeroed all features; falling back to top-by-magnitude.")
        nonzero = coefs.abs().nlargest(min(target_count, len(coefs)))
        nonzero = coefs.loc[nonzero.index]

    selected = nonzero.abs().nlargest(target_count)
    selected = nonzero.loc[selected.index]  # preserve sign
    print(f"  LASSO-Cox (alpha={best_alpha:.5f}): selected {len(selected)} features.")

    return selected, imputer, scaler, X_sp.columns, lasso, scorecard


# ================================================================================
# Cross-validation (outer loop: feature consensus + OOF C-index)
# ================================================================================

def run_cross_validation_survival(X, y_dev_df, event_col, time_col, target_count,
                                   c_index_cutoff=0.6, random_state=RANDOM_STATE):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    events = y_dev_df[event_col]

    fold_coefficients = pd.DataFrame(0.0, index=X.columns,
                                      columns=[f'Fold_{i+1}' for i in range(5)])
    oof_risk = pd.Series(np.nan, index=X.index, name='risk_score')

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, events)):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr_df    = y_dev_df.iloc[tr_idx]
        y_va_df    = y_dev_df.iloc[va_idx]

        y_tr_struct = make_survival_array(y_tr_df[event_col], y_tr_df[time_col])

        selected_coefs, imputer, scaler, spearman_feats, lasso, scorecard = run_fold_pruning_survival(
            X_tr, y_tr_struct, target_count, c_index_cutoff=c_index_cutoff, random_state=random_state
        )

        fold_coefficients.loc[selected_coefs.index, f'Fold_{fold+1}'] = selected_coefs.values

        # Apply preprocessing to validation fold
        X_va_imp = pd.DataFrame(imputer.transform(X_va), columns=X_va.columns, index=X_va.index)
        X_va_sc  = pd.DataFrame(scaler.transform(X_va_imp), columns=X_va.columns, index=X_va.index)
        X_va_sp  = X_va_sc.reindex(columns=spearman_feats, fill_value=0.0)

        risk = lasso.predict(X_va_sp)
        oof_risk.iloc[va_idx] = risk

        y_va_struct = make_survival_array(y_va_df[event_col], y_va_df[time_col])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            c_val = concordance_index_censored(y_va_struct['Status'], y_va_struct['Time'], risk)[0]
        print(f"Fold {fold+1}: val C-index = {c_val:.3f}, features selected = {len(selected_coefs)}\n")

    fold_cols = [f'Fold_{i+1}' for i in range(5)]
    fold_coefficients['Selection_Frequency'] = (fold_coefficients[fold_cols] != 0).sum(axis=1)
    fold_coefficients['Mean_Absolute_Coefficient'] = fold_coefficients[fold_cols].abs().mean(axis=1)

    def _sign_stable(row):
        nz = row[fold_cols][row[fold_cols] != 0]
        if len(nz) == 0:
            return False
        return bool((nz > 0).all() or (nz < 0).all())

    fold_coefficients['Sign_Stable'] = fold_coefficients.apply(_sign_stable, axis=1)
    fold_coefficients = fold_coefficients[fold_coefficients['Selection_Frequency'] > 0]
    fold_coefficients = fold_coefficients.sort_values(
        ['Selection_Frequency', 'Mean_Absolute_Coefficient'], ascending=False
    )

    # Global dev-set scorecard for reporting
    dev_imp = KNNImputer(n_neighbors=5)
    X_imp   = pd.DataFrame(dev_imp.fit_transform(X), columns=X.columns, index=X.index)
    dev_sc  = StandardScaler()
    X_sc    = pd.DataFrame(dev_sc.fit_transform(X_imp), columns=X.columns, index=X.index)
    y_struct_all = make_survival_array(y_dev_df[event_col], y_dev_df[time_col])
    global_scorecard = calculate_survival_feature_metrics(X_sc, y_struct_all)

    return fold_coefficients, global_scorecard, oof_risk


def summarize_nested_cv_survival(oof_risk, y_dev_df, event_col, time_col,
                                  n_bootstrap=1000, random_state=RANDOM_STATE):
    """Pooled OOF C-index with bootstrap CI."""
    valid = oof_risk.notna()
    risk_valid = oof_risk[valid].values
    y_valid    = y_dev_df.loc[oof_risk[valid].index]
    y_struct   = make_survival_array(y_valid[event_col], y_valid[time_col])

    c_point = concordance_index_censored(y_struct['Status'], y_struct['Time'], risk_valid)[0]

    rng = np.random.default_rng(random_state)
    boot_scores = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(risk_valid), size=len(risk_valid))
        y_b = y_struct[idx]
        r_b = risk_valid[idx]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                c_b = concordance_index_censored(y_b['Status'], y_b['Time'], r_b)[0]
                if not np.isnan(c_b):
                    boot_scores.append(c_b)
        except Exception:
            pass

    boot_scores = np.array(boot_scores)
    ci_lo = float(np.percentile(boot_scores, 2.5)) if len(boot_scores) else np.nan
    ci_hi = float(np.percentile(boot_scores, 97.5)) if len(boot_scores) else np.nan

    print(f"\nNested CV OOF C-index: {c_point:.3f} (95% CI: {ci_lo:.3f} – {ci_hi:.3f})")
    return {'oof_c_index': c_point, 'ci_lower': ci_lo, 'ci_upper': ci_hi, 'n': int(valid.sum())}


# ================================================================================
# Model factory
# ================================================================================

def get_survival_model(model_name, random_state=RANDOM_STATE):
    if model_name == 'lasso_cox':
        # Final LASSO-Cox fitted on consensus features; alpha chosen by inner CV in training stage.
        return None  # built dynamically in train_and_evaluate_survival_model
    elif model_name == 'rsf':
        return RandomSurvivalForest(
            n_estimators=300, min_samples_split=10, min_samples_leaf=5,
            random_state=random_state, n_jobs=-1
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ================================================================================
# Bootstrap CI for C-index
# ================================================================================

def bootstrap_c_index(y_struct, risk_scores, n_bootstrap=1000, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    boot = []
    n = len(risk_scores)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                c = concordance_index_censored(
                    y_struct['Status'][idx], y_struct['Time'][idx], risk_scores[idx]
                )[0]
            if not np.isnan(c):
                boot.append(c)
        except Exception:
            pass
    boot = np.array(boot)
    c_pt = concordance_index_censored(y_struct['Status'], y_struct['Time'], risk_scores)[0]
    ci_lo = float(np.percentile(boot, 2.5)) if len(boot) else np.nan
    ci_hi = float(np.percentile(boot, 97.5)) if len(boot) else np.nan
    return {'c_index': c_pt, 'ci_lower': ci_lo, 'ci_upper': ci_hi, 'n_valid': len(boot)}


# ================================================================================
# Time-dependent AUC
# ================================================================================

def compute_td_auc(y_train_struct, y_test_struct, risk_scores, times=None):
    """
    Returns a DataFrame of AUC at each timepoint.
    Times outside the observed event range are dropped automatically.
    """
    if times is None:
        times = TD_AUC_TIMES

    # sksurv requires times strictly inside the observed event time range
    min_t = y_test_struct['Time'][y_test_struct['Status']].min()
    max_t = y_test_struct['Time'].max()
    valid_times = [t for t in times if min_t < t < max_t]

    if not valid_times:
        print("  Warning: no valid TD-AUC timepoints within observed event range.")
        return pd.DataFrame({'time_days': times, 'AUC': np.nan, 'label': TD_AUC_LABELS[:len(times)]})

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        auc_vals, mean_auc = cumulative_dynamic_auc(
            y_train_struct, y_test_struct, risk_scores, times=valid_times
        )

    rows = []
    valid_labels = [TD_AUC_LABELS[TD_AUC_TIMES.index(t)] for t in valid_times if t in TD_AUC_TIMES]
    for t, a, lbl in zip(valid_times, auc_vals, valid_labels):
        rows.append({'time_days': t, 'label': lbl, 'AUC': a})
    print(f"  Mean TD-AUC: {mean_auc:.3f}")
    return pd.DataFrame(rows)


# ================================================================================
# Plots
# ================================================================================

def plot_km_by_risk(risk_scores, y_struct, out_dir, model_name, event_col_label='Recurrence'):
    """KM curves for high vs. low risk groups (split at median risk score)."""
    median_risk = np.median(risk_scores)
    high = risk_scores > median_risk
    low  = ~high

    time   = y_struct['Time']
    status = y_struct['Status']

    lr = logrank_test(time[high], time[low], event_observed_A=status[high], event_observed_B=status[low])

    fig, ax = plt.subplots(figsize=(8, 6))
    for mask, label, color in [(high, 'High risk', '#d62728'), (low, 'Low risk', '#1f77b4')]:
        kmf = KaplanMeierFitter()
        kmf.fit(time[mask], event_observed=status[mask], label=label)
        kmf.plot_survival_function(ax=ax, ci_show=True, color=color)

    ax.set_xlabel('Time (days)', fontsize=12)
    ax.set_ylabel(f'{event_col_label}-free survival', fontsize=12)
    ax.set_title(f'KM Curves — {model_name.upper()}\n(log-rank p = {lr.p_value:.4f})', fontsize=13)
    ax.legend(fontsize=11)
    plt.tight_layout()
    path = os.path.join(out_dir, f'{model_name}_km_curves.png')
    plt.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  KM plot saved: {path}")
    return path


def plot_td_auc_curve(auc_dfs, model_names, out_dir):
    """AUC vs time curve for all models on the lock set."""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#1f77b4', '#d62728']
    for df, name, color in zip(auc_dfs, model_names, colors):
        valid = df.dropna(subset=['AUC'])
        ax.plot(valid['time_days'] / 365.25, valid['AUC'], marker='o',
                label=name.upper(), color=color, linewidth=2)
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1.5, label='Random')
    ax.set_xlabel('Time (years)', fontsize=12)
    ax.set_ylabel('Time-dependent AUC', fontsize=12)
    ax.set_title('Time-dependent AUC — Lock Set', fontsize=13)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    path = os.path.join(out_dir, 'td_auc_curve.png')
    plt.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  TD-AUC plot saved: {path}")
    return path


# ================================================================================
# Train and evaluate a single model
# ================================================================================

def train_and_evaluate_survival_model(X_dev, y_dev_df, X_lock, y_lock_df,
                                       model_name, event_col, time_col,
                                       out_dir, param_dir=None, n_bootstrap=1000,
                                       random_state=RANDOM_STATE):
    print(f"\n{'='*60}")
    print(f"Training: {model_name.upper()}")
    print(f"{'='*60}")

    y_dev_struct  = make_survival_array(y_dev_df[event_col],  y_dev_df[time_col])
    y_lock_struct = make_survival_array(y_lock_df[event_col], y_lock_df[time_col])

    try:
        model_source = 'default'
        if model_name == 'lasso_cox':
            alphas = np.logspace(-4, 1, 20)
            best_alpha = _select_lasso_cox_alpha(X_dev, y_dev_struct, alphas, random_state=random_state)
            model = CoxnetSurvivalAnalysis(alphas=[best_alpha], l1_ratio=1.0, max_iter=1000)
            model.fit(X_dev, y_dev_struct)
            model_source = f'inner_cv (alpha={best_alpha:.5f})'
            print(f"  LASSO-Cox alpha (inner CV): {best_alpha:.5f}")
            coefs = pd.Series(model.coef_[:, 0], index=X_dev.columns)
            coefs[coefs != 0].sort_values(key=abs, ascending=False).to_csv(
                os.path.join(out_dir, 'lasso_cox_final_coefs.csv'))
        else:
            saved = load_saved_estimator(model_name, param_dir)
            if saved is not None:
                model = saved
                model_source = 'gridsearch_pickle'
            else:
                model = get_survival_model(model_name, random_state=random_state)
                model_source = 'default_fallback'
            model.fit(X_dev, y_dev_struct)

        risk_lock = model.predict(X_lock)

        ci_result = bootstrap_c_index(y_lock_struct, risk_lock, n_bootstrap=n_bootstrap, seed=random_state)
        td_auc_df = compute_td_auc(y_dev_struct, y_lock_struct, risk_lock)

        print(f"  Lock C-index: {ci_result['c_index']:.3f} "
              f"(95% CI: {ci_result['ci_lower']:.3f} – {ci_result['ci_upper']:.3f})")
        print(f"  Time-dependent AUC:")
        for _, row in td_auc_df.iterrows():
            print(f"    {row['label']}: {row['AUC']:.3f}")

        km_path = plot_km_by_risk(risk_lock, y_lock_struct, out_dir, model_name)

        with open(os.path.join(out_dir, f'{model_name}_model.pkl'), 'wb') as f:
            pickle.dump(model, f)

        return {
            'model_name': model_name,
            'model': model,
            'model_source': model_source,
            'c_index': ci_result['c_index'],
            'ci_lower': ci_result['ci_lower'],
            'ci_upper': ci_result['ci_upper'],
            'td_auc': td_auc_df,
            'risk_lock': risk_lock,
            'status': 'success',
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'model_name': model_name, 'status': 'failed', 'error': str(e)}


# ================================================================================
# Stage 2: Gridsearch (RSF only)
# ================================================================================

def perform_grid_search_survival(X, y_struct, out_dir, cv=5, n_jobs=6, random_state=RANDOM_STATE):
    """
    Tune RSF hyperparameters on the dev set via stratified CV (stratified on event status).
    Saves the best estimator as a pickle and the full cv_results to CSV.
    LASSO-Cox is intentionally excluded — its alpha is selected by inner CV in the train stage.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Pre-generate splits stratified on event status; passing them as cv= bypasses
    # the GridSearchCV stratification issue with structured arrays.
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    splits = list(skf.split(X, y_struct['Status']))

    base_rsf = RandomSurvivalForest(n_estimators=300, random_state=random_state, n_jobs=1)
    gs = GridSearchCV(
        estimator=base_rsf,
        param_grid=PARAM_GRIDS_SURVIVAL['rsf'],
        cv=splits,
        n_jobs=n_jobs,
        refit=True,
        verbose=1,
    )
    gs.fit(X, y_struct)

    print(f"\nBest RSF C-index (dev CV): {gs.best_score_:.4f}")
    print(f"Best params: {gs.best_params_}")

    pkl_path = os.path.join(out_dir, 'best_estimator_rsf.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(gs.best_estimator_, f)

    cv_df = pd.DataFrame(gs.cv_results_).sort_values('rank_test_score')
    cv_df.to_csv(os.path.join(out_dir, 'cv_results_rsf.csv'), index=False)

    summary = pd.DataFrame([{'model': 'rsf', 'best_score': gs.best_score_,
                              'best_params': json.dumps(gs.best_params_)}])
    summary.to_csv(os.path.join(out_dir, 'grid_search_results.csv'), index=False)

    return gs.best_estimator_


def load_saved_estimator(model_name, param_dir):
    """Load a pickled best estimator written by gridsearch stage, or return None."""
    if param_dir is None:
        return None
    path = os.path.join(param_dir, f'best_estimator_{model_name}.pkl')
    if os.path.exists(path):
        print(f"  Loading pickled estimator for {model_name} from: {path}")
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None


# ================================================================================
# Data loading helpers (between stages)
# ================================================================================

def load_preprocessed_survival(in_dir):
    X_dev  = pd.read_csv(f"{in_dir}/X_dev_processed.csv", index_col=0)
    X_lock = pd.read_csv(f"{in_dir}/X_lock_processed.csv", index_col=0)
    y_dev  = pd.read_csv(f"{in_dir}/y_dev.csv", index_col=0)
    y_lock = pd.read_csv(f"{in_dir}/y_lock.csv", index_col=0)
    print(f"Loaded — X_dev: {X_dev.shape}, X_lock: {X_lock.shape}")
    return X_dev, X_lock, y_dev, y_lock


# ================================================================================
# Stage runners
# ================================================================================

def run_partition_stage(args):
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    df_all = pd.read_csv(args.features_path, index_col=0)
    df_filtered = preselect_sequences(df_all, allowed_sequences=args.sequences)

    X_dev, y_dev, X_lock, y_lock = partition_with_constraints_survival(
        df_filtered,
        outcomes_path=args.outcomes_path,
        icc_path=args.icc_path,
        clinical_data_path=args.clinical_data_path,
        event_col=args.event_col,
        time_col=args.time_col,
        test_size=args.test_size,
        output_dir=args.output_dir,
    )

    compare_dev_lock_survival(y_dev, y_lock, args.event_col, args.time_col, args.output_dir)

    X_dev_clean     = global_cleanup(X_dev, cv_threshold=args.cv_threshold)
    X_dev_clean_icc = interobserver_stability(X_dev_clean, args.path_r1, args.path_r2,
                                              threshold=args.icc_threshold)
    y_dev_filtered  = y_dev.loc[X_dev_clean_icc.index]

    print(f"\nAfter cleanup + ICC: {X_dev_clean_icc.shape[0]} samples, "
          f"{X_dev_clean_icc.shape[1]} features")

    fold_coefficients, global_scorecard, oof_risk = run_cross_validation_survival(
        X_dev_clean_icc, y_dev_filtered,
        event_col=args.event_col, time_col=args.time_col,
        target_count=args.lasso_target_count,
        c_index_cutoff=args.c_index_cutoff,
        random_state=RANDOM_STATE,
    )

    nested_cv_result = summarize_nested_cv_survival(
        oof_risk, y_dev_filtered, args.event_col, args.time_col,
        n_bootstrap=1000, random_state=RANDOM_STATE,
    )

    fold_coefficients.to_csv(f"{args.output_dir}/lasso_coeffs_fold.csv")
    global_scorecard.to_csv(f"{args.output_dir}/univariate_cox_metrics.csv")
    pd.DataFrame([nested_cv_result]).to_csv(f"{args.output_dir}/nested_cv_diagnostics.csv", index=False)

    frequent = fold_coefficients[
        (fold_coefficients['Selection_Frequency'] >= args.min_fold_n) &
        fold_coefficients['Sign_Stable']
    ]
    consensus_features = frequent.head(args.target_feature_count).index.tolist()
    print(f"\nConsensus features ({len(consensus_features)}): {consensus_features}")

    # Final preprocessing: fit on dev, apply to lock
    imputer = KNNImputer(n_neighbors=5)
    scaler  = StandardScaler()

    X_dev_sub  = X_dev_clean_icc[consensus_features]
    X_lock_sub = X_lock.reindex(columns=consensus_features, fill_value=np.nan)

    X_dev_imp  = pd.DataFrame(imputer.fit_transform(X_dev_sub),  columns=consensus_features, index=X_dev_sub.index)
    X_dev_proc = pd.DataFrame(scaler.fit_transform(X_dev_imp),   columns=consensus_features, index=X_dev_sub.index)
    X_lock_imp  = pd.DataFrame(imputer.transform(X_lock_sub),    columns=consensus_features, index=X_lock_sub.index)
    X_lock_proc = pd.DataFrame(scaler.transform(X_lock_imp),     columns=consensus_features, index=X_lock_sub.index)

    X_dev_proc.to_csv(f"{args.output_dir}/X_dev_processed.csv")
    X_lock_proc.to_csv(f"{args.output_dir}/X_lock_processed.csv")
    y_dev_filtered.to_csv(f"{args.output_dir}/y_dev.csv")
    y_lock.to_csv(f"{args.output_dir}/y_lock.csv")
    pd.Series(consensus_features, name='feature').to_csv(f"{args.output_dir}/consensus_features.csv", index=False)

    with open(f"{args.output_dir}/imputer.pkl", 'wb') as f:
        pickle.dump(imputer, f)
    with open(f"{args.output_dir}/scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)

    print(f"\nPartition artifacts saved to: {args.output_dir}")
    print(f"Dev: {X_dev_proc.shape} | Lock: {X_lock_proc.shape}")


def run_gridsearch_stage(args):
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    in_dir = args.input_dir or args.partition_dir

    X_dev, _, y_dev, _ = load_preprocessed_survival(in_dir)
    y_dev_struct = make_survival_array(y_dev[args.event_col], y_dev[args.time_col])

    perform_grid_search_survival(
        X_dev, y_dev_struct,
        out_dir=args.output_dir,
        cv=args.cv,
        n_jobs=args.n_jobs,
        random_state=RANDOM_STATE,
    )


def run_train_stage(args):
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    in_dir = args.input_dir or args.partition_dir

    X_dev, X_lock, y_dev, y_lock = load_preprocessed_survival(in_dir)

    results_list = []
    auc_dfs, model_names_ok = [], []

    for model_name in ALL_MODELS:
        result = train_and_evaluate_survival_model(
            X_dev, y_dev, X_lock, y_lock,
            model_name=model_name,
            event_col=args.event_col,
            time_col=args.time_col,
            out_dir=args.output_dir,
            param_dir=args.input_params_dir,
            n_bootstrap=args.bootstrap,
            random_state=RANDOM_STATE,
        )
        results_list.append(result)
        if result['status'] == 'success':
            auc_dfs.append(result['td_auc'])
            model_names_ok.append(model_name)

    if len(auc_dfs) > 1:
        plot_td_auc_curve(auc_dfs, model_names_ok, args.output_dir)

    # Summary table
    rows = []
    for r in results_list:
        if r['status'] == 'success':
            row = {'Model': r['model_name'],
                   'Source': r.get('model_source', ''),
                   'C-index': f"{r['c_index']:.3f}",
                   '95% CI': f"({r['ci_lower']:.3f} – {r['ci_upper']:.3f})"}
            for _, td_row in r['td_auc'].iterrows():
                row[f"AUC {td_row['label']}"] = f"{td_row['AUC']:.3f}" if not np.isnan(td_row['AUC']) else 'N/A'
            rows.append(row)
        else:
            rows.append({'Model': r['model_name'], 'Source': '-',
                         'C-index': 'FAILED', '95% CI': r.get('error', '')})

    summary = pd.DataFrame(rows)
    summary.to_csv(f"{args.output_dir}/model_comparison.csv", index=False)
    print(f"\n{'='*80}")
    print("MODEL COMPARISON (Lock Set)")
    print(f"{'='*80}")
    print(summary.to_string(index=False))
    print(f"{'='*80}\n")


# ================================================================================
# CLI
# ================================================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Recurrence survival pipeline: LASSO-Cox + RSF.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--stage', choices=['partition', 'gridsearch', 'train'], default=DEFAULT_STAGE)

    # --- shared paths ---
    _base_data = '/Volumes/T7 Shield/Restrepo/HCC_Retrospective_radiomics'
    _sheets    = f"{_base_data}/Results/Radiomics_sheets"
    parser.add_argument('--features-path', type=str,
        default=f"{_sheets}/SA1_2/R-SA1_2_I-sitkBSpline_S-1mm_W-perseq.csv")
    parser.add_argument('--path-r1', type=str,
        default=f"{_sheets}/inter-reader/Reader1/R-Reader1_I-sitkBSpline_S-1mm_W-perseq.csv")
    parser.add_argument('--path-r2', type=str,
        default=f"{_sheets}/inter-reader/Reader2/R-Reader2_I-sitkBSpline_S-1mm_W-perseq.csv")
    parser.add_argument('--outcomes-path', type=str,
        default=f"{_base_data}/databases/recurrence_outcomes.xlsx")
    parser.add_argument('--clinical-data-path', type=str,
        default=f"{_base_data}/databases/SA1-2_clinical_imaging.xlsx")
    parser.add_argument('--icc-path', type=str, default=None,
        help="Defaults to --path-r1.")

    # --- outcome columns ---
    parser.add_argument('--event-col', type=str, default='event',
        help="Column in outcomes file: 1=recurred, 0=censored.")
    parser.add_argument('--time-col', type=str, default='time_days',
        help="Column in outcomes file: follow-up duration in days.")

    # --- sequence selection ---
    parser.add_argument('--sequences', nargs='+', default=['pvp', 'tp', 'hbp', 'dwi_b1', 'adc'])

    # --- partition stage ---
    parser.add_argument('--test-size', type=int, default=30)
    parser.add_argument('--cv-threshold', type=float, default=0.1)
    parser.add_argument('--icc-threshold', type=float, default=0.85)
    parser.add_argument('--lasso-target-count', type=int, default=100)
    parser.add_argument('--c-index-cutoff', type=float, default=0.6,
        help="Univariate Cox C-index threshold (features above this OR below 1-threshold are kept).")
    parser.add_argument('--min-fold-n', type=int, default=3,
        help="Minimum number of folds a feature must appear in for consensus.")
    parser.add_argument('--target-feature-count', type=int, default=15,
        help="Number of consensus features to carry into the final model.")

    # --- gridsearch stage ---
    parser.add_argument('--cv', type=int, default=5)
    parser.add_argument('--n-jobs', type=int, default=6)

    # --- train stage ---
    parser.add_argument('--bootstrap', type=int, default=1000)
    parser.add_argument('--shap', action='store_true',
        help="(Reserved) Generate SHAP plots (not yet implemented for survival models).")

    # --- I/O ---
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--input-dir', type=str, default=None,
        help="Input dir for the current stage (gridsearch reads partition artifacts; "
             "train reads partition artifacts).")
    parser.add_argument('--input-params-dir', type=str, default=None,
        help="Gridsearch output dir containing pickled RSF estimator. "
             "If omitted, train stage falls back to default RSF hyperparameters.")

    args = parser.parse_args()

    if args.icc_path is None:
        args.icc_path = args.path_r1

    seq_tag = "_".join(args.sequences)
    _base_out = f"{_base_data}/Results/main_pipeline/recurrence_survival/binwidth_perseq/{seq_tag}"
    _partition_dir  = f"{_base_out}/partition_select"
    _gridsearch_dir = f"{_base_out}/optimized_params"
    _train_dir      = f"{_base_out}/trained_model_outputs"

    # Stash resolved partition dir so stage runners can reference it regardless of --output-dir
    args.partition_dir = _partition_dir

    if args.output_dir is None:
        if args.stage == 'partition':
            args.output_dir = _partition_dir
        elif args.stage == 'gridsearch':
            args.output_dir = _gridsearch_dir
        else:
            args.output_dir = _train_dir

    if args.input_dir is None:
        args.input_dir = _partition_dir
    if args.input_params_dir is None:
        args.input_params_dir = _gridsearch_dir

    return args


if __name__ == '__main__':
    args = parse_arguments()

    print("\n" + "=" * 80)
    print(f"RECURRENCE SURVIVAL PIPELINE — STAGE: {args.stage.upper()}")
    print(f"Output: {args.output_dir}")
    print("=" * 80 + "\n")

    if args.stage == 'partition':
        run_partition_stage(args)
    elif args.stage == 'gridsearch':
        run_gridsearch_stage(args)
    elif args.stage == 'train':
        run_train_stage(args)

    print(f"\n{'='*80}")
    print(f"STAGE '{args.stage}' COMPLETE")
    print(f"{'='*80}\n")
