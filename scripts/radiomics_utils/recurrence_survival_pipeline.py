import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from lifelines import CoxPHFitter  # <-- FIXED: Added missing import
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from scipy.stats import mannwhitneyu, chi2_contingency
from sklearn.model_selection import StratifiedKFold, train_test_split
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis  # <-- FIXED: Added missing imports

def preselect_sequences(df, allowed_sequences, id_col='PatientID'):
    """
    Filters the dataframe to only include specific sequences based on column names.
    """
    if not allowed_sequences:
        print("No sequence filter applied. Keeping all features.")
        return df

    cols_to_keep = [id_col] if id_col in df.columns else []
    
    for col in df.columns:
        if any(seq.lower() in col.lower() for seq in allowed_sequences):
            cols_to_keep.append(col)
            
    filtered_df = df[cols_to_keep]
    print(f"Preselection complete. Kept sequences: {allowed_sequences}")
    print(f"Features reduced from {df.shape[1]} to {filtered_df.shape[1]}")
    
    return filtered_df

def partition_with_constraints_survival(df_filtered, outcomes_path, icc_path, master_metadata_path, 
                                        event_col, time_col, test_size, output_dir):
    """
    Partitions survival data, ensuring ICC patients are in Dev while stratifying by event status.
    """
    df_outcomes = pd.read_excel(outcomes_path)
    df_icc = pd.read_csv(icc_path)
    
    df_merged = df_filtered.reset_index().merge(
        df_outcomes[['PatientID', event_col, time_col]], on='PatientID', how='inner'
    ).set_index('PatientID')

    before_drop = len(df_merged)
    df_merged = df_merged.dropna(subset=[event_col, time_col])
    print(f"Dropped {before_drop - len(df_merged)} patients from merge pool due to missing survival data ({event_col}/{time_col}).")
    
    icc_ids = df_icc['PatientID'].unique()
    df_dev_must = df_merged[df_merged.index.isin(icc_ids)]
    df_rest = df_merged[~df_merged.index.isin(icc_ids)]
    
    if len(df_rest) < test_size:
        raise ValueError(f"Not enough patients for a test set of size {test_size}.")
        
    X_rest = df_rest.drop(columns=[event_col, time_col])
    y_rest_event = df_rest[event_col]
    
    X_dev_rest, X_lock, _, _ = train_test_split(
        X_rest, y_rest_event, 
        test_size=test_size, 
        stratify=y_rest_event, 
        random_state=997
    )
    
    X_dev = pd.concat([df_dev_must.drop(columns=[event_col, time_col]), X_dev_rest])
    
    y_dev_survival = df_merged.loc[X_dev.index, [event_col, time_col]]
    y_lock_survival = df_merged.loc[X_lock.index, [event_col, time_col]]

    df_master = pd.read_excel(master_metadata_path)
    df_master_dev = df_master[df_master['PatientID'].isin(X_dev.index)]
    df_master_lock = df_master[df_master['PatientID'].isin(X_lock.index)]
    
    df_master_dev.to_csv(f"{output_dir}/clinical_metadata_dev.csv", index=False)
    df_master_lock.to_csv(f"{output_dir}/clinical_metadata_lock.csv", index=False)
    
    return X_dev, y_dev_survival, X_lock, y_lock_survival, df_master_dev, df_master_lock

def compare_dev_test_survival(df_dev, df_lock, variables, output_dir, event_col, time_col):
    """
    Creates a summary table comparing Development and Test sets for Survival Workflows.
    """
    df_dev = df_dev[df_dev[event_col].notnull() & df_dev[time_col].notnull()].copy()
    df_lock = df_lock[df_lock[event_col].notnull() & df_lock[time_col].notnull()].copy()
    
    results = []  # <-- FIXED: Initialized empty results list
    
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

    results_df = pd.DataFrame(results)
    results_df.to_excel(f"{output_dir}/dev_test_comparison.xlsx", index=False)
    return results_df

def global_cleanup(df, var_threshold, id_col='PatientID'):
    """Robust cleaning for Radiomics features."""
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
    df = df.loc[:, df.std() > var_threshold]
    
    print(f"Cleanup finished. Columns remaining: {df.shape[1]}")
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

from sksurv.linear_model import CoxPHSurvivalAnalysis

def calculate_survival_feature_metrics(X, y_structured):
    """
    High-speed univariate Cox using sksurv.
    Only returns features that fit cleanly.
    """
    from sksurv.metrics import concordance_index_censored
    
    metrics = []
    skipped = 0
    failed = []
    
    # Diagnostic info
    print(f"  Univariate Cox: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  y_structured dtype: {y_structured.dtype}, shape: {y_structured.shape}")
    print(f"  Event distribution: {np.sum(y_structured['Status'])} events, {np.sum(~y_structured['Status'])} censored")
    
    for col_idx, col in enumerate(X.columns):
        cox = CoxPHSurvivalAnalysis(tol=1e-3)
        
        try:
            # Fit the single feature
            cox.fit(X[[col]], y_structured)
            
            # Get risk scores (higher = higher risk)
            risk_scores = cox.predict(X[[col]])
            
            # Compute concordance index using sksurv's metric
            c_index, _, _, _, _ = concordance_index_censored(
                y_structured['Status'], y_structured['Time'], risk_scores
            )
            
            # Calculate Hazard Ratio for interpretability
            hazard_ratio = np.exp(cox.coef_[0])
            
            # Skip features with numerical garbage
            if np.isnan(c_index) or hazard_ratio > 1e4 or hazard_ratio < 1e-4:
                skipped += 1
                continue

            metrics.append({
                'Feature': col,
                'p_value': 0.05,  # Placeholder (not computed for sksurv)
                'C_index': c_index,
                'Hazard_Ratio': hazard_ratio
            })

        except Exception as e:
            skipped += 1
            if col_idx < 5:
                failed.append(f"    {col}: {type(e).__name__}: {str(e)[:100]}")
            continue
    
    if failed:
        print(f"  First {len(failed)} failed fits:")
        for msg in failed:
            print(msg)
    
    if skipped > 0:
        print(f"  Skipped {skipped} features due to numerical issues or fitting failure.")
    
    if len(metrics) == 0:
        raise ValueError(
            f"All {X.shape[1]} features failed univariate Cox regression. "
            "Check: (1) fold sample size ≥ 2, (2) event distribution (not all 0 or all 1), "
            "(3) feature variance (not constant/near-constant)."
        )
    
    print(f"  Successfully fit {len(metrics)} features.")
    return pd.DataFrame(metrics).set_index('Feature')

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

def run_survival_fold_pruning(X_tr, y_tr_struct, target_count, p_value_cutoff=1.00, c_index_cutoff=0.50, alpha=0.01):
    """
    Pruning workflow for an isolated training fold using survival parameters.
    
    Args:
        alpha (float): Fixed regularization strength for CoxnetSurvivalAnalysis. Default 0.1.
    """
    imputer = KNNImputer(n_neighbors=5)
    X_imp = pd.DataFrame(imputer.fit_transform(X_tr), columns=X_tr.columns, index=X_tr.index)
    
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imp), columns=X_imp.columns, index=X_imp.index)

    scorecard = calculate_survival_feature_metrics(X_scaled, y_tr_struct)
    total_features = len(scorecard)

    # Keep top features by C-index (no hard thresholds to avoid filtering everything out)
    univariate_keep_count = min(total_features, max(target_count * 5, 100))
    kept_features_info = scorecard.nlargest(univariate_keep_count, 'C_index')
    
    print(f"Univariate selection: Keeping top {len(kept_features_info)} features by C-index.")
    X_filtered = X_scaled[kept_features_info.index]

    if X_filtered.shape[1] == 0:
        raise ValueError("No features remain after univariate selection.")

    initial_n = X_filtered.shape[1]
    p_vals = scorecard.loc[X_filtered.columns, 'p_value']
    
    X_spearman = spearman_filter(X_filtered, threshold=0.85, tie_breaker_series=p_vals)
    kept_features = X_spearman.columns
    final_n = X_spearman.shape[1]
    print(f"Spearman: Dropped {initial_n - final_n} features (from {initial_n} to {final_n})")

    if X_spearman.shape[1] == 0:
        # Keep top features by C-index if Spearman filtered everything
        print("Warning: Spearman filtering removed all features. Keeping top features by C-index.")
        fallback_count = min(len(kept_features_info), max(target_count * 2, 20))
        X_spearman = X_filtered[kept_features_info.head(fallback_count).index]
        kept_features = X_spearman.columns
        print(f"Fallback retained {len(kept_features)} features.")

    lasso_survival = CoxnetSurvivalAnalysis(l1_ratio=1.0, alphas=[alpha])
    lasso_survival.fit(X_spearman, y_tr_struct)

    selected_features = list(X_spearman.columns)
    
    for alpha_idx in range(len(lasso_survival.alphas_)):
        coefs = pd.Series(lasso_survival.coef_[:, alpha_idx], index=X_spearman.columns)
        active_feats = coefs[coefs != 0].abs().sort_values(ascending=False)
        
        if len(active_feats) > 0:
            selected_features = active_feats.head(target_count)
            break
    else:
        coefs = pd.Series(lasso_survival.coef_[:, -1], index=X_spearman.columns)
        selected_features = coefs[coefs != 0].abs().sort_values(ascending=False).head(target_count)
        if len(selected_features) == 0:
            selected_features = coefs.abs().sort_values(ascending=False).head(target_count)

    print(f"Cox-LASSO (alpha={alpha}): Selected top {len(selected_features)} non-zero features.")
    return selected_features, imputer, scaler, kept_features, lasso_survival, scorecard

def run_cross_validation_survival(X, y_dev_survival, target_count, event_col, time_col):
    """
    Coordinates 5-Fold survival cross-validation.
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = []
    events = y_dev_survival[event_col]

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, events)):
        X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
        y_tr_df = y_dev_survival.iloc[train_idx]
        y_va_df = y_dev_survival.iloc[val_idx]
        
        y_tr_struct = np.array(
            list(zip(y_tr_df[event_col].astype(bool), y_tr_df[time_col])),
            dtype=[('Status', '?'), ('Time', '<f8')]
        )
        
        selected_coefs, imputer, scaler, kept_features, lasso_model, scorecard = run_survival_fold_pruning(
            X_tr, y_tr_struct, target_count, p_value_cutoff=1.0, c_index_cutoff=0.50, alpha=0.01
        )
        
        X_va_imp = pd.DataFrame(imputer.transform(X_va), columns=X_va.columns, index=X_va.index)
        X_va_scaled = pd.DataFrame(scaler.transform(X_va_imp), columns=X_va.columns, index=X_va.index)
        X_va_final = X_va_scaled.reindex(columns=kept_features, fill_value=0)
        
        risk_scores = lasso_model.predict(X_va_final)
        
        from sksurv.metrics import concordance_index_censored
        va_c_index = concordance_index_censored(y_va_df[event_col].astype(bool), y_va_df[time_col], risk_scores)[0]
        
        cv_results.append({
            'fold': fold + 1,
            'coefs': selected_coefs,
            'features': list(selected_coefs.index),
            'c_index': va_c_index
        })
        print(f"Fold {fold+1} complete. Validation C-index: {va_c_index:.3f}\n")

    flattened_data = []
    for res in cv_results:
        for feat, coef in zip(res['features'], res['coefs']):
            flattened_data.append({
                'Fold': res['fold'],
                'Global_C_Index': res['c_index'],
                'Feature': feat,
                'Coefficient': coef
            })
            
    return pd.DataFrame(flattened_data), scorecard

def finalize_consensus(consensus_features, scorecard, target_count=15, metric='C_index'):
    """
    Reduces the consensus feature list to a fixed target length using ranking metrics.
    """
    if scorecard is None or len(consensus_features) == 0:
        print("No scorecard or empty consensus; returning original consensus list.")
        return consensus_features

    sc = scorecard.copy()
    missing = [f for f in consensus_features if f not in sc.index]
    if missing:
        print(f"Warning: {len(missing)} feature(s) missing from scorecard: {missing}")
    sc_sub = sc.loc[[f for f in consensus_features if f in sc.index]]

    if len(sc_sub) <= target_count:
        print(f"Consensus features ({len(sc_sub)}) <= target_count ({target_count}); no reduction applied.")
        return list(sc_sub.index)

    # <-- FIXED: Handled non-C_index fallbacks robustly to prevent unbound variables
    if metric == 'C_index':
        sc_sorted = sc_sub.sort_values(by='C_index', ascending=False)
    elif metric == 'p_value':
        sc_sorted = sc_sub.sort_values(by='p_value', ascending=True)
    else:
        sc_sorted = sc_sub.copy()

    selected = list(sc_sorted.index[:target_count])
    print(f"Reduced consensus from {len(consensus_features)} to {len(selected)} using metric '{metric}'")
    return selected

def get_survival_model(model_name, random_state=42):
    if model_name == 'cox_ph':
        return CoxPHSurvivalAnalysis()
    elif model_name == 'random_survival_forest':
        return RandomSurvivalForest(n_estimators=100, min_samples_split=10, 
                                    min_samples_leaf=5, random_state=random_state, n_jobs=-1)
    elif model_name == 'gradient_boosting_survival':
        return GradientBoostingSurvivalAnalysis(n_estimators=100, learning_rate=0.05, 
                                                random_state=random_state)
    else:
        raise ValueError(f"Survival model {model_name} not recognized.")
    
def train_and_eval_survival_model(X_dev_final, y_dev_df, X_lock, y_lock_df, model_name, event_col, time_col):
    """
    Fits a survival model and returns the validation C-index.
    """
    y_dev_struct = np.array(list(zip(y_dev_df[event_col].astype(bool), y_dev_df[time_col])),
                            dtype=[('Status', '?'), ('Time', '<f8')])
    y_lock_struct = np.array(list(zip(y_lock_df[event_col].astype(bool), y_lock_df[time_col])),
                             dtype=[('Status', '?'), ('Time', '<f8')])
    
    model = get_survival_model(model_name)
    model.fit(X_dev_final, y_dev_struct)
    
    lock_c_index = model.score(X_lock, y_lock_struct)
    return model, lock_c_index


########  -----------------------------------------------------------------------------------------  #####################
# Execution starts here


if __name__ == "__main__":
    my_sequences = ['t2','pre','ap','pvp','tp','hbp','dwi_b1','dwi_b2','adc']
    
    base_dir = '/Volumes/T7 Shield/Restrepo/HCC_Retrospective_radiomics/Results/Radiomics_sheets'
    features_path = f"{base_dir}/SA1_2/R-SA1_2_I-sitkBSpline_S-1mm_W-5.csv"
    path_r1 = f"{base_dir}/inter-reader/R-Reader1_I-sitkBSpline_S-1mm_W-5.csv"
    path_r2 = f"{base_dir}/inter-reader/R-Reader2_I-sitkBSpline_S-1mm_W-5.csv"
    outcomes_path = '/Volumes/T7 Shield/Restrepo/HCC_Retrospective_radiomics/databases/survival_outcomes.xlsx'
    clinical_imaging_path = '/Volumes/T7 Shield/Restrepo/HCC_Retrospective_radiomics/databases/SA1-2_clinical_imaging.xlsx'

    out_dir = '/Volumes/T7 Shield/Restrepo/HCC_Retrospective_radiomics/Results/main_pipeline/recurrence_survival/binwidth5/' + "_".join(my_sequences)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # 1. Pipeline Input Preselection
    df_all_seq = pd.read_csv(features_path, index_col=0)
    df_filtered = preselect_sequences(df_all_seq, allowed_sequences=my_sequences)
    
    # Define your survival targets here
    EVENT_COL = '2-year recurrence'  # Must contain 0 or 1
    TIME_COL = 'TTR or LFU' # Continuous float numbers
    
    # 2. Partition and generate baseline checks
    X_dev, y_dev, X_lock, y_lock, df_master_dev, df_master_lock = partition_with_constraints_survival(
        df_filtered, outcomes_path, path_r1, clinical_imaging_path, 
        event_col=EVENT_COL, time_col=TIME_COL, test_size=30, output_dir=out_dir
    )
    
    variables=['Age', 'Gender', 'BMI', 'Cirrhosis', 'Capsule on PVP', 'Mosaic appearance', 'Macrovascular invasion', 'Tumor size on PVP', '2-year recurrence']

    compare_dev_test_survival(df_master_dev, df_master_lock, variables, out_dir, EVENT_COL, TIME_COL)
    
    # 3. Clean and isolate based on Inter-reader reliability properties
    X_dev_clean = global_cleanup(X_dev, id_col='PatientID', var_threshold=0.05)
    X_dev_clean_icc = interobserver_stability(X_dev_clean, path_r1, path_r2, id_col='PatientID', threshold=0.85)
    y_dev_filtered = y_dev.loc[X_dev_clean_icc.index]
    
    # 4. Cross validation for consensus selection
    results_df, scorecard = run_cross_validation_survival(
        X_dev_clean_icc, y_dev_filtered, target_count=100, event_col=EVENT_COL, time_col=TIME_COL
    )
    
    feature_counts = results_df['Feature'].value_counts()
    consensus_features = feature_counts[feature_counts >= 3].index.tolist()
    scorecard.to_csv(f"{out_dir}/univariate_survival_metrics.csv")

    # 5. Lock-down top features and transform test sets safely
    consensus_features = finalize_consensus(consensus_features, scorecard, target_count=15, metric='C_index')
    
    imputer = KNNImputer(n_neighbors=5)
    scaler = StandardScaler()
    
    X_dev_subset = X_dev_clean_icc[consensus_features]
    X_lock_subset = X_lock[consensus_features]

    X_dev_final = pd.DataFrame(imputer.fit_transform(X_dev_subset), columns=consensus_features, index=X_dev_subset.index)
    X_dev_final = pd.DataFrame(scaler.fit_transform(X_dev_final), columns=consensus_features, index=X_dev_subset.index)

    X_lock_final = pd.DataFrame(imputer.transform(X_lock_subset), columns=consensus_features, index=X_lock.index)
    X_lock_final = pd.DataFrame(scaler.transform(X_lock_final), columns=consensus_features, index=X_lock.index)

    # 6. Final Evaluation Model
    model, lock_c_index = train_and_eval_survival_model(
        X_dev_final, y_dev_filtered, X_lock_final, y_lock, 
        model_name='random_survival_forest', event_col=EVENT_COL, time_col=TIME_COL
    )
    print(f"\nFinal Random Survival Forest Concordance Index on Lock set: {lock_c_index:.3f}")

    # 7. Save preprocessing artifacts for model iteration
    print("\nSaving preprocessing artifacts...")
    X_dev_final.to_csv(f"{out_dir}/X_dev_processed.csv")
    X_lock_final.to_csv(f"{out_dir}/X_lock_processed.csv")
    pd.DataFrame(y_dev_filtered).to_csv(f"{out_dir}/y_dev.csv", index=False)
    pd.DataFrame(y_lock).to_csv(f"{out_dir}/y_lock.csv", index=False)
    pd.Series(consensus_features, name='feature').to_csv(f"{out_dir}/consensus_features.csv", index=False)
    pd.Series(imputer.n_features_in_).to_frame().T.to_csv(f"{out_dir}/imputer_n_features.csv", index=False)