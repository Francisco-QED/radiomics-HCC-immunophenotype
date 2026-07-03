##### ------------ MRI-radiomics for prediction of immunophenotype in HCC -------------- #####
# Requirements:
# - Radiomics features are extracted for all patients and stored in a CSV file.
# - Inter-reader features for ICC calculation are available from patients segmented by two readers.
# - Clinical metadata and outcomes are available in CSV or Excel format, including the binary label.
#
# Pipeline overview:
# 1) Preselect MRI sequences defined by the configuration.
# 2) Partition the dataset into development and lock sets.
# 3) Apply global cleanup and interobserver stability [or intraclass correlation (ICC)] filtering.
# 4) Run cross-validation and identify consensus features.
# 5) Train and evaluate final models on the lock set.

import numpy as np
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from tabpfn import TabPFNClassifier
from sklearn.impute import KNNImputer
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from scipy.stats import mannwhitneyu, chi2_contingency
from sklearn.model_selection import StratifiedKFold, train_test_split

# The first step is to select the MRI sequences that will be included in the model #
# df: DataFrame containing radiomics features for all patients. Each column represents a 
#     feature, and the column names include the sequence type (e.g., T1, T2, DWI).
# allowed_sequences: List of sequences to keep (e.g., ['T1', 'T2', 'DWI']). If empty, all sequences are kept.
def preselect_sequences(df, allowed_sequences, id_col='PatientID'):
    if not allowed_sequences:
        print("No sequence filter applied. Keeping all features.")
        return df

    # Keep the ID column
    cols_to_keep = [id_col] if id_col in df.columns else []
    
    # Identify columns that match our allowed keywords
    for col in df.columns:
        if any(seq.lower() in col.lower() for seq in allowed_sequences):
            cols_to_keep.append(col)
            
    filtered_df = df[cols_to_keep]
    print(f"Preselection complete. Kept sequences: {allowed_sequences}")
    print(f"Features reduced from {df.shape[1]} to {filtered_df.shape[1]}")
    return filtered_df

# --- Partitioning with ICC and lock set constraints ----------------------------------------------------------------
# Partition the data into development and lock sets.Ensure all inter-reader patients remain in the development set,
# while the remaining pool is stratified by outcome for the lock set. 

# outcomes_path: Path to the excel file containing outcomes and labels. 
# icc_path: Path to the CSV file containing patients with double ROIs for ICC calculation.
# clinical_data_path: Path to the clinical metadata file (Excel) containing patient clinical information.
# cutoff: Threshold for binarizing the outcome variable 
# test_size: Number of patients to include in the lock set.

def partition_with_constraints(df_filtered, outcomes_path, icc_path, clinical_data_path, label_col, cutoff, test_size, output_dir):
    #Load Data
    #df_raw = pd.read_csv(features_path, index_col=0) # X
    df_outcomes = pd.read_excel(outcomes_path)       # Y + labels
    df_outcomes['y_bin'] = (df_outcomes[label_col] >= cutoff).astype(int)
    df_icc = pd.read_csv(icc_path)                 # List of patients with double ROIs
    
    # Merge Features and Outcomes
    df_merged = df_filtered.reset_index().merge(df_outcomes[['PatientID', 'y_bin']], on='PatientID', how='inner').set_index('PatientID')
    
    # Separate the pools
    icc_ids = df_icc['PatientID'].unique()
    # Dev-Must: Patients with double ROIs (inter-reader cases must be in the development set)
    df_dev_must = df_merged[df_merged.index.isin(icc_ids)]
    # Flexible: Everyone else
    df_rest = df_merged[~df_merged.index.isin(icc_ids)]
        
    X_rest = df_rest.drop(columns=['y_bin'])
    y_rest = df_rest['y_bin']
    
    # Split the flexible pool
    X_dev_rest, X_lock, y_dev_rest, y_lock = train_test_split(
        X_rest, y_rest, 
        test_size=test_size, 
        stratify=y_rest, 
        random_state=997
    )
    
    # Combine to form Final Sets
    X_dev = pd.concat([df_dev_must.drop(columns=['y_bin']), X_dev_rest])
    y_dev = pd.concat([df_dev_must['y_bin'], y_dev_rest])
    
    # Filter clinical sheet based on the final patient IDs in each set
    df_clinical = pd.read_excel(clinical_data_path)
    df_clinical_dev = df_clinical[df_clinical['PatientID'].isin(X_dev.index)]
    df_clinical_lock = df_clinical[df_clinical['PatientID'].isin(X_lock.index)]
    
    # Save for records
    df_clinical_dev.to_csv(f"{output_dir}/clinical_metadata_dev.csv", index=False)
    df_clinical_lock.to_csv(f"{output_dir}/clinical_metadata_lock.csv", index=False)

    print(f"Partitioning Complete.")
    print(f"Dev Set: {len(X_dev)} (Includes {len(df_dev_must)} ICC patients)")
    print(f"Lock Set: {len(X_lock)}")
    print(f"Sub-sheets created: Dev ({len(df_clinical_dev)}), Lock ({len(df_clinical_lock)})")
    
    return X_dev, y_dev, X_lock, y_lock, df_clinical_dev, df_clinical_lock
    
# This function compares the development and lock sets for continuous and categorical variables, providing summary statistics and p-values 
# for statistical tests. This is to ensure that the dev and lock sets are comparable and that there are no significant differences in key variables.
# df_dev, df_lock: master clinical dataframes for the development and lock sets, respectively.
# variables: pre-selected list of variables to compare (e.g., Age, Gender, etc)

# Creates a summary table comparing Development and Test sets.
    # Continuous (Age, BMI): Median [IQR] + Mann-Whitney U
    # Categorical (Cirrhosis, Gender): n (%) for all levels + Chi-Square

# df_dev, df_lock: Clinical dataframes for the development and lock sets, respectively.
def compare_dev_test(df_dev, df_lock, variables, output_dir, outcome_col):

    # Drop rows with missing outcome values.
    before_dev = len(df_dev)    
    df_dev = df_dev[df_dev[outcome_col].notnull()].copy()
    if before_dev != len(df_dev):
        print(f"Dropped {before_dev - len(df_dev)} dev rows missing '{outcome_col}'")
    before_lock = len(df_lock)
    df_lock = df_lock[df_lock[outcome_col].notnull()].copy()
    if before_lock != len(df_lock):
        print(f"Dropped {before_lock - len(df_lock)} lock rows missing '{outcome_col}'")

    # Now compare central tendencies in both sets for each variable, and perform statistical tests to see if they differ significantly.
    results = []
    
    for var in variables:
        # Determine if variable is Categorical (strings like Y/N, M/F) or Continuous
        is_categorical = df_dev[var].dtype == 'object' or df_dev[var].nunique() < 5
        
        if is_categorical:
            # --- Categorical Logic ---
            def get_cat_str(df, column):
                counts = df[column].value_counts()
                total = len(df[column].dropna())
                # Create a string like "Y: 40 (42.1%), N: 55 (57.9%)"
                return ", ".join([f"{idx}: {val} ({(val/total)*100:.1f}%)" 
                                 for idx, val in counts.items()])

            dev_str = get_cat_str(df_dev, var)
            lock_str = get_cat_str(df_lock, var)
            
            # Chi-Square Test for independence
            combined_series = pd.concat([df_dev[var], df_lock[var]])
            # Align the group labels with the combined_series index to avoid accidental reindexing
            group_labels = pd.Series(['Dev'] * len(df_dev) + ['Lock'] * len(df_lock), index=combined_series.index)
            contingency = pd.crosstab(combined_series, group_labels)
            print(f"Contingency Table for {var}:\n{contingency}\n")
            try:
                # Use correction=True for small sample sizes in the 2x2 table
                _, p, _, _ = chi2_contingency(contingency)
            except ValueError:
                p = np.nan
            test_type = 'Chi-Square'

        else:
            # --- Continuous Logic ---
            # Calculate Median and Interquartile Range (IQR)
            q1_d, med_d, q3_d = df_dev[var].quantile([0.25, 0.5, 0.75])
            q1_l, med_l, q3_l = df_lock[var].quantile([0.25, 0.5, 0.75])
            
            dev_str = f"{med_d:.1f} [{q1_d:.1f} - {q3_d:.1f}]"
            lock_str = f"{med_l:.1f} [{q1_l:.1f} - {q3_l:.1f}]"
            
            # Mann-Whitney U Test (Independent samples)
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


### This performs the cleanup that needs to happen inside the development set only, like inter-reader reliabiltiy, 
# drop nans and drop near-zero variance features. We do this to avoid data leakage, and then we will apply the same 
# transformations to the lock set later.
# df: devlopment set X_dev from "partition_with_constraints"
# var_threshold: threshold for near-zero variance filtering (e.g., 0.05)
def global_cleanup(df, var_threshold, id_col='PatientID'):
    # Set Index
    if id_col in df.columns:
        df = df.set_index(id_col)
    # Force numeric-only
    df = df.select_dtypes(include=[np.number])
    # (Replace Inf with NaN so they can be dropped/imputed)
    df = df.replace([np.inf, -np.inf], np.nan)
    # Remove rows that are entirely NaN (e.g., patients with no features).
    # 'how=all' ensures we only drop patients if every single feature is empty.
    initial_rows = len(df)
    df = df.dropna(how='all')
    if len(df) < initial_rows:
        print(f"Dropped {initial_rows - len(df)} rows that were entirely empty.")
    # Drop columns that are mostly NaN (e.g., > 10% missing)
    # Threshold adjustable
    thresh = len(df) * 0.9 
    df = df.dropna(axis=1, thresh=thresh)

    #### Most importantly..... drop low-variance columns ####
    df = df.loc[:, df.std() > var_threshold]
    
    print(f"Cleanup finished. Columns remaining: {df.shape[1]}")
    return df

# This function will calculate the ICC for each feature across the two readers, and 
# only keep those that are above the threshold.
# df: cleaned up X_dev from "global_cleanup"
# path_r1: path to reader 1's data
# path_r2: path to reader 2's data
# threshold: minimum ICC value for feature inclusion
# id_col: column name for patient ID
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


# --- Fold-specific preprocessing and feature selection ------------------------------------------------
# The remaining feature removal and selection occurs inside each training fold to avoid data leakage.
# This includes KNN imputation, scaling, univariate selection, and correlation filtering.

# The next two functions are auxiliary functions that will be use in the "run_fold_pruning" function, which is 
# the main function that implements the fold-specific pruning steps. For each fold, there will be knn imputation, 
# z-scaling, univariate filtering, correlation filtering, and LASSO-based selection. The function "run_fold_pruning" 
# will learn the parameters of imputation, scaling, and feature selection from the training fold and later apply them 
# to the lock set (to avoid data leakage).

# The first step is to calculate univariate metrics for each feature (AUCs, Mann-Whitney U test, and odds ratios)
# Input X should be the training fold feature matrix created in the main CV function.

def calculate_feature_metrics(X, y):
    metrics = []
    for col in X.columns:
        # Mann-Whitney U test (p-value)
        group0 = X[col][y == 0]
        group1 = X[col][y == 1]
        _, p = mannwhitneyu(group0, group1, alternative='two-sided')
        
        # Univariate AUC
        # We compute AUC on the scaled feature
        auc = roc_auc_score(y, X[col])
        # Force AUC to be > 0.5 (if < 0.5, the direction is just flipped)
        #if auc < 0.5:
        #    auc = 1 - auc
            
        # Odds Ratio (Univariate Logistic Regression)
        # Fit a simple model with one feature
        model = LogisticRegression(solver='liblinear')
        model.fit(X[[col]], y)
        
        # The coefficient (beta) represents log-odds.
        # Exponentiate it to get the Odds Ratio.
        beta = model.coef_[0][0]
        odds_ratio = np.exp(beta)
        
        metrics.append({
            'Feature': col,
            'p_value': p,
            'AUC': auc,
            'Odds_Ratio': odds_ratio
        })
        
    return pd.DataFrame(metrics).set_index('Feature')

# Then, we consider the spearman correlation between features, and drop one of each pair that 
# is above the threshold. The tie-breaker is the p-value (or AUC or OR) from above, where we keep 
# the one with the better score (e.g., lower p-value, higher AUC, or OR further from 1).

def spearman_filter(df, threshold, tie_breaker_series):
    corr_matrix = df.corr(method='spearman').abs()
    to_drop = set()
    cols = df.columns
    
    # Identify pairs to drop
    print("Pruning redundant features...")
    for i in tqdm(range(len(cols)), desc="Spearman Filter"):
        for j in range(i + 1, len(cols)):
            f1, f2 = cols[i], cols[j]
            if f1 not in to_drop and f2 not in to_drop:
                if corr_matrix.loc[f1, f2] > threshold:
                    # Tie-breaking logic: drop the one with 'worse' score
                    if tie_breaker_series is not None:
                        if tie_breaker_series[f1] <= tie_breaker_series[f2]:
                            to_drop.add(f2)
                        else:
                            to_drop.add(f1)
                    else:
                        to_drop.add(f2) # Default behavior
    
    return df.drop(columns=list(to_drop))

## This is the function that will implement the fold-specific pruning steps, including imputation, scaling, univariate filtering, correlation 
# filtering, and LASSO-based selection. This function "learns" the parameters of imputation, scaling, and feature selection from the training 
# fold and later applies them to the lock set (to avoid data leakage).
def run_fold_pruning(X_tr, y_tr, target_count, p_value_cutoff, metric, metric_cutoff):
    # Impute and scale
    imputer = KNNImputer(n_neighbors=5)
    X_imp = pd.DataFrame(imputer.fit_transform(X_tr), columns=X_tr.columns, index=X_tr.index)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imp), columns=X_imp.columns, index=X_imp.index)

    # Pruning based on p-value, AUC
    scorecard = calculate_feature_metrics(X_scaled, y_tr)
    total_features = len(scorecard)

    if metric.lower() == 'auc':
        # Two-sided AUC filter 
        lower_limit = 1 - metric_cutoff
        mask_kept = (scorecard['AUC'] >= metric_cutoff) | (scorecard['AUC'] <= lower_limit)
        print(f"Filter: Keeping AUC >= {metric_cutoff} or <= {lower_limit:.2f}")
        
    elif metric.lower() == 'or':
        # Symmetrical OR filter using Log-Odds (e.g., OR > 1.5 or OR < 0.67)
        # np.log(1.0) is 0, so we check distance from 0
        log_threshold = np.abs(np.log(metric_cutoff))
        mask_kept = np.abs(np.log(scorecard['Odds_Ratio'])) > log_threshold
        print(f"Filter: Keeping OR > {metric_cutoff} or < {1/metric_cutoff:.2f}")

    # Apply significance filter
    mask_sig = scorecard['p_value'] < p_value_cutoff
    # Final decision: Magnitude OR Significance (less strict)
    final_mask = mask_kept | mask_sig  # Changed from & to | to be less strict
    kept_features_info = scorecard[final_mask]
    
    print(f"Discarding {total_features - len(kept_features_info)} features.")
    
    X_filtered = X_scaled[kept_features_info.index]

    # Spearman
    initial_n = X_filtered.shape[1]
    p_vals = scorecard.loc[X_filtered.columns, 'p_value']
    X_spearman = spearman_filter(X_filtered, threshold=0.85, tie_breaker_series=p_vals)
    kept_features = X_spearman.columns
    final_n = X_spearman.shape[1]
    print(f"Spearman: Dropped {initial_n - final_n} features (from {initial_n} to {final_n})")
    # LASSO
    lasso_cv = LogisticRegression(penalty='l1', solver='liblinear', C=1, random_state=42)
    lasso_cv.fit(X_spearman, y_tr)

    # Extract
    coefs = pd.Series(lasso_cv.coef_[0], index=X_spearman.columns)
    selected = coefs[coefs != 0].abs().sort_values(ascending=False).head(target_count)
    print(f"LASSO: Dropped {len(coefs) - len(selected)} features (from {len(coefs)} to {len(selected)})")
    return selected, imputer, scaler, kept_features, lasso_cv, scorecard

# --- Cross-validation wrapper ----------------------------------------------------------------------------------
# Run the main K-fold training loop, applying fold-specific pruning and then evaluating on the validation fold.
# Input X should be the cleaned, ICC-filtered development features, and y_dev the matching labels.

def run_cross_validation(X, y_dev, target_count, p_value_cutoff, metric, metric_cutoff):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Create a master dataframe to hold coefficients for every single feature across all 5 folds
    # Initialize with zeros; if a feature is dropped by a fold, its coefficient stays 0.0
    fold_coefficients = pd.DataFrame(0.0, index=X.columns, columns=[f'Fold_{i+1}' for i in range(5)])

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_dev)):
        X_tr, y_tr = X.iloc[train_idx], y_dev.iloc[train_idx]
        
        # Run your fold pruning (with the tightened/regularized Lasso as discussed)
        selected_coefs, imputer, scaler, kept_features, lasso, scorecard = run_fold_pruning(
            X_tr, y_tr, target_count, p_value_cutoff, metric, metric_cutoff=metric_cutoff
        )
        
        # selected_coefs is a pandas Series indexed by Feature Name with the coefficient as the value
        # Map these back to our master dataframe for this specific fold column
        fold_coefficients.loc[selected_coefs.index, f'Fold_{fold+1}'] = selected_coefs.values
        
        # Calculate how many features this fold actually selected
        active_count = (selected_coefs != 0).sum()
        print(f"Fold {fold+1} complete. Features selected by LASSO: {active_count}")

    # Calculate selection frequency (how many folds out of 5 the feature appeared in)
    fold_coefficients['Selection_Frequency'] = (fold_coefficients[[f'Fold_{i+1}' for i in range(5)]] != 0).sum(axis=1)
    
    # Calculate Mean Absolute Coefficient across the folds to show overall feature strength
    fold_coefficients['Mean_Absolute_Coefficient'] = fold_coefficients[[f'Fold_{i+1}' for i in range(5)]].abs().mean(axis=1)

    fold_coefficients = fold_coefficients[fold_coefficients['Selection_Frequency'] > 0]

    # Sort the table so the most consistently selected and strongest features are at the top
    fold_coefficients = fold_coefficients.sort_values(by=['Selection_Frequency', 'Mean_Absolute_Coefficient'], ascending=False)

    # We also return a clean global scorecard calculated from the full dev set to pass downstream
    dev_imputer = KNNImputer(n_neighbors=5)
    X_dev_imp = pd.DataFrame(dev_imputer.fit_transform(X), columns=X.columns, index=X.index)
    dev_scaler = StandardScaler()
    X_dev_scaled = pd.DataFrame(dev_scaler.fit_transform(X_dev_imp), columns=X.columns, index=X.index)
    global_scorecard = calculate_feature_metrics(X_dev_scaled, y_dev)
    
    return fold_coefficients, global_scorecard  

### After the best features are selected, we combine them via a model (e.g., logistic regression, SVM, random forest, XGBoost, TabPFN)

def get_model(model_name, random_state=42):
    if model_name == 'logistic':
        return LogisticRegression(penalty='l2', C=10, solver='lbfgs', max_iter=1000)
    elif model_name == 'svm_linear':
        # Probability=True is required for ROC-AUC
        return SVC(kernel='linear', probability=True, random_state=random_state)
    elif model_name == 'svm_rbf':
        # Requires tuning C (regularization) and gamma (boundary flexibility)
        return SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
    elif model_name == 'rf':
        return RandomForestClassifier(n_estimators=100, max_depth=3, random_state=random_state)
    elif model_name == 'xgboost':
        # Low depth to prevent overfitting on N=89
        return XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, 
                             use_label_encoder=False, eval_metric='logloss', random_state=random_state)
    elif model_name == 'tabpfn':
        # TabPFN is typically used as a classifier without much tuning
        return TabPFNClassifier(device='cpu')
    else:
        raise ValueError(f"Model {model_name} not recognized.")

 
####  Finally train and evaluate models with  the selected features  ####
def train_and_eval_model(X_dev_final, y_dev, X_lock, y_lock, model_name):
    if len(X_dev_final) != len(y_dev):
        raise ValueError(
            f"Development feature/label length mismatch: "
            f"X has {len(X_dev_final)} samples, y has {len(y_dev)} labels."
        )
    if len(X_lock) != len(y_lock):
        raise ValueError(
            f"Lock feature/label length mismatch: "
            f"X has {len(X_lock)} samples, y has {len(y_lock)} labels."
        )

    model = get_model(model_name)
    
    # Fit the model
    model.fit(X_dev_final, y_dev)
    
    # Predict probabilities (needed for AUC)
    y_probs = model.predict_proba(X_lock)[:, 1]
    auc = roc_auc_score(y_lock, y_probs)
    
    return model, auc


#-------------------------------------------------------------------------------------------------------------------------------------#

# --- EXECUTION ---------------------------------------------------------------------------------------------
# Main script entry point. This loads the data, runs preprocessing, identifies consensus features, and evaluates a final model.

if __name__ == "__main__":

    # ----------------------- MRI sequences to include in the analysis -----------------------------------------------#
    my_sequences = ['pvp','tp','hbp','dwi_b1','adc']
    # My params (depend on what endpoint we consider, response, binary recurrence, or TILs/TLS)

    ###### First define relevant paths ######
    base_dir = '/Volumes/T7 Shield/Restrepo/HCC_Retrospective_radiomics/Results/Radiomics_sheets'
    features_path = f"{base_dir}/SA1_2/R-SA1_2_I-sitkBSpline_S-1mm_W-5.csv"
    path_r1 = f"{base_dir}/inter-reader/R-Reader1_I-sitkBSpline_S-1mm_W-5.csv"
    path_r2 = f"{base_dir}/inter-reader/R-Reader2_I-sitkBSpline_S-1mm_W-5.csv"
    outcomes_path = "/Volumes/T7 Shield/Restrepo/HCC_Retrospective_radiomics/databases/immunophenotype_outcomes.xlsx"
    clinical_imaging_path = '/Volumes/T7 Shield/Restrepo/HCC_Retrospective_radiomics/databases/SA1-2_clinical_imaging.xlsx'

    # Output folder will be named following the sequences chosen
    out_dir = '/Volumes/T7 Shield/Restrepo/HCC_Retrospective_radiomics/Results/main_pipeline/test_3_vs_0-2/dev110_test30/binwidth5/' + "_".join(my_sequences) + "/clean_partition_select"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Training and evaluation will be done after this, once we have the consensus features from the cross-validation results. The first 
    # steps are to load the data, preselect sequences, partition into dev and lock sets, perform global cleaning and interobserver stability 
    # filtering, and then run the cross-validation to identify consensus features.
    df_all_seq = pd.read_csv(features_path, index_col=0)
    df_filtered = preselect_sequences(df_all_seq, allowed_sequences=my_sequences)
    
    X_dev, y_dev, X_lock, y_lock, df_clinical_dev, df_clinical_lock = partition_with_constraints(
        df_filtered,
        outcomes_path,
        path_r1,
        label_col='TILs_resection',
        clinical_data_path=clinical_imaging_path,
        cutoff=3,
        test_size=30,
        output_dir=out_dir,
    )
    
    variables=['Age', 'Gender', 'BMI', 'Cirrhosis', 'Capsule on PVP', 'Mosaic appearance', 'Macrovascular invasion', 'Tumor size on PVP', 'TILs score', 'TLS score']
    compare_dev_test(df_clinical_dev, df_clinical_lock, variables, output_dir=out_dir, outcome_col='TILs score')

    X_dev_clean = global_cleanup(X_dev, id_col='PatientID', var_threshold=0.05)   # Development set after global cleaning
    X_dev_clean_icc = interobserver_stability(X_dev_clean, path_r1, path_r2, id_col='PatientID', threshold=0.85) 
    
    # Filter y_dev to match the remaining samples in X_dev_clean_icc
    y_dev_filtered = y_dev.loc[X_dev_clean_icc.index]
    print(f"After filtering: X_dev has {X_dev_clean_icc.shape[0]} samples, y_dev has {len(y_dev_filtered)} samples")
    
    fold_coefficients, global_scorecard = run_cross_validation(X_dev_clean_icc, y_dev_filtered, target_count=100, p_value_cutoff = 0.05, metric = 'auc', metric_cutoff = 0.65)
    fold_coefficients.to_csv(f"{out_dir}/lasso_coeffs_fold.csv", index=True)

    # Apply consensus (e.g., >= 3 folds). Selecting features that were chosen in at least n out of 5 folds.
    min_fold_n = 3
    frequent_features = fold_coefficients[fold_coefficients['Selection_Frequency'] >= min_fold_n]
    target_feature_count = 10
    consensus_features = frequent_features.head(target_feature_count).index.tolist()
    print(f"Consensus features identified: {len(consensus_features)}")
    print(consensus_features)
    global_scorecard.to_csv(f"{out_dir}/univariate_metrics.csv")

# =========================================================================================================#
    #                                 FEATURE DIAGNOSTICS BLOCK                                         #
    # =========================================================================================================#
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy.stats import mannwhitneyu
    from sklearn.metrics import roc_auc_score
    import numpy as np

    print("\n" + "="*85)
    print("RUNNING FEATURE DIAGNOSTICS (DEV vs. LOCK)")
    print("="*85)

    # 1. Heatmap using the clean Development Set
    X_dev_diag = X_dev_clean_icc[consensus_features]
    plt.figure(figsize=(8, 6))
    corr_matrix = X_dev_diag.corr(method='spearman')
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1, square=True)
    plt.title(f"Spearman Correlation (Dev Set): Top {len(consensus_features)} Features")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/consensus_features_correlation.png", dpi=300)
    plt.close()
    print(f"-> Saved correlation heatmap to: {out_dir}/consensus_features_correlation.png")

    # 2. Clean structural anomalies in X_lock for safe raw diagnostic evaluation
    X_lock_diag = X_lock.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)[consensus_features]
    X_lock_diag = X_lock_diag.fillna(X_dev_diag.mean())  # Quick temporary fill using Dev mean for raw AUC calculus

    # 3. Print Side-by-Side Univariate Metric Breakdown Table
    print(f"\n{'Feature Name':<40} | {'Dev MW p':<10} | {'Dev AUC':<8} | {'Lock MW p':<10} | {'Lock AUC':<8}")
    print("-" * 85)
    
    for feat in consensus_features:
        try:
            # --- Development Set Stats (Drop NaNs dynamically per feature) ---
            dev_data = pd.concat([X_dev_diag[feat], y_dev_filtered], axis=1).dropna()
            dev_feat_clean = dev_data[feat]
            dev_y_clean = dev_data[y_dev_filtered.name if y_dev_filtered.name else y_dev_filtered.columns[0] if isinstance(y_dev_filtered, pd.DataFrame) else y_dev_filtered.name]
            
            d0 = dev_feat_clean[dev_y_clean == 0]
            d1 = dev_feat_clean[dev_y_clean == 1]
            
            _, p_dev = mannwhitneyu(d0, d1, alternative='two-sided')
            auc_dev = roc_auc_score(dev_y_clean, dev_feat_clean)
            if auc_dev < 0.5: auc_dev = 1 - auc_dev
            
            # --- Lock Set Stats (Drop NaNs dynamically per feature) ---
            # Slicing from raw X_lock before any filling to see true raw test performance
            X_lock_raw_feat = X_lock.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)[feat]
            lock_data = pd.concat([X_lock_raw_feat, y_lock], axis=1).dropna()
            lock_feat_clean = lock_data[feat]
            lock_y_clean = lock_data[y_lock.name if y_lock.name else y_lock.columns[0] if isinstance(y_lock, pd.DataFrame) else y_lock.name]
            
            l0 = lock_feat_clean[lock_y_clean == 0]
            l1 = lock_feat_clean[lock_y_clean == 1]
            
            _, p_lock = mannwhitneyu(l0, l1, alternative='two-sided')
            auc_lock = roc_auc_score(lock_y_clean, lock_feat_clean)
            if auc_lock < 0.5: auc_lock = 1 - auc_lock
            
            print(f"{feat:<50} | {p_dev:<10.3f} | {auc_dev:<8.3f} | {p_lock:<10.3f} | {auc_lock:<8.3f}")
        except Exception as e:
            print(f"{feat:<50} | Error: {e}")
    # =========================================================================================================#

    #######    Final model run on test set    #######

    # Initialize preprocessors
    imputer = KNNImputer(n_neighbors=5)
    scaler = StandardScaler()

    # Extract the selected features 
    X_dev_subset = X_dev_clean_icc[consensus_features]
    X_lock_subset = X_lock[consensus_features]

    # Process development set (train the imputer and scaler on the dev set)
    X_dev_impute = pd.DataFrame(imputer.fit_transform(X_dev_subset), columns=consensus_features, index=X_dev_subset.index)
    X_dev_final = pd.DataFrame(scaler.fit_transform(X_dev_impute), columns=consensus_features, index=X_dev_subset.index)

    # Process lock set (transform the test set using the dev set's imputer and scaler)
    X_lock_impute = pd.DataFrame(imputer.transform(X_lock_subset), columns=consensus_features, index=X_lock.index)
    X_lock_final = pd.DataFrame(scaler.transform(X_lock_impute), columns=consensus_features, index=X_lock.index)

    # Save preprocessing items for model iteration
    print("\nSaving preprocessing items for reference and reproducibility")
    X_dev_final.to_csv(f"{out_dir}/X_dev_processed.csv")
    X_lock_final.to_csv(f"{out_dir}/X_lock_processed.csv")
    pd.Series(y_dev_filtered, name='y_dev').to_csv(f"{out_dir}/y_dev.csv")
    pd.Series(y_lock, name='y_lock').to_csv(f"{out_dir}/y_lock.csv")
    pd.Series(consensus_features, name='feature').to_csv(f"{out_dir}/consensus_features.csv", index=False)
    pd.Series(imputer.n_features_in_).to_frame().T.to_csv(f"{out_dir}/imputer_n_features.csv", index=False)
    
    # Save transformers as pickle for exact reproducibility
    import pickle
    with open(f"{out_dir}/imputer.pkl", 'wb') as f:
        pickle.dump(imputer, f)
    with open(f"{out_dir}/scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"Artifacts saved to {out_dir}")
    print(f"Processed Development Set: {X_dev_final.shape}")
    print(f"Processed Lock Set: {X_lock_final.shape}")
    print(f"Consensus Features: {len(consensus_features)}")
    
    #--------------------------------------------------------------------------------------------------#
    # Example: Train and evaluate with one model
    model = 'svm_rbf'  # Change to 'logistic', 'rf', 'xgboost', or 'tabpfn' as needed
    print("\n" + "="*60)
    print(f"TRAINING EXAMPLE MODEL: {model}")
    print("="*60)
    model, auc = train_and_eval_model(X_dev_final, y_dev_filtered, X_lock_final, y_lock, model_name=model)
    final_test_df = pd.concat([X_lock_final, y_lock], axis=1)
    final_test_df.to_csv(f"{out_dir}/final_test_set_features.csv", index=True)
    print(f"Final Model AUC on Lock set: {auc:.3f}")
    
    print("\n" + "="*60)
    print("To train additional models without rerunning preprocessing,")
    print("run: python model_trainer.py")
    print("="*60)
