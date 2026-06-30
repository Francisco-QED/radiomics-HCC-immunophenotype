import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.svm import SVC
#from xgboost import XGBClassifier
from tabpfn import TabPFNClassifier
from scipy.stats import mannwhitneyu
from sklearn.impute import KNNImputer
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV


# Endpoint-aware partitioning function into test set and development set. We want to ensure 
# that the distribution of responders vs non-responders is balanced across the two sets, so we 
# will use stratified sampling based on the response label.

def partition_data(df_reliable, outcomes_path, label_col, cutoff,test_size=20):
    df_outcomes = pd.read_excel(outcomes_path)
    df_merged = df_reliable.reset_index().merge(
        df_outcomes[['PatientID', label_col]], on='PatientID', how='inner'
    ).set_index('PatientID')

    df_merged[label_col] = (df_merged[label_col] >= cutoff).astype(int)
    X = df_merged.drop(columns=[label_col])
    y = df_merged[label_col]
    
    # Stratified split to ensure even distribution of responders
    X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)
    
    print(f"Partitioning Complete: Dev={len(X_dev)}, Test={len(X_test)}")
    return X_dev, y_dev, X_test, y_test

# --- 1. Modular Spearman Filter ---
def drop_redundant_features(df, threshold, tie_breaker_series=None):
    """
    df: The dataframe of features (X)
    tie_breaker_series: A pandas Series with feature names as index 
                        (e.g., p-values or ICC scores)
    """
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


def calculate_feature_metrics(X, y):
    """
    Computes p-values, AUC, and Odds Ratios for each feature.
    """
    
    metrics = []
    for col in X.columns:
        # A. Mann-Whitney U test (p-value)
        group0 = X[col][y == 0]
        group1 = X[col][y == 1]
        _, p = mannwhitneyu(group0, group1, alternative='two-sided')
        
        # B. Univariate AUC
        # We compute AUC on the scaled feature
        auc = roc_auc_score(y, X[col])
        # Force AUC to be > 0.5 (if < 0.5, the direction is just flipped)
        if auc < 0.5:
            auc = 1 - auc
            
        # C. Odds Ratio (Univariate Logistic Regression)
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

# Usage:
# scorecard = calculate_feature_metrics(X_dev, y_dev)
# # Now filter: Keep features with high AUC and low p-values
# top_features = scorecard[(scorecard['AUC'] > 0.65) & (scorecard['p_value'] < 0.05)]

# --- 3. Internal Fold Logic (The "Inspectable" Engine) ---
def run_fold_pruning(X_tr, y_tr, target_count):
    # Impute
    imputer = KNNImputer(n_neighbors=5)
    X_imp = pd.DataFrame(imputer.fit_transform(X_tr), columns=X_tr.columns, index=X_tr.index)

    # Scale
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imp), columns=X_imp.columns, index=X_imp.index)
    # Pruning based on p-value, AUC
    scorecard = calculate_feature_metrics(X_scaled, y_tr)
    total_features = len(scorecard)
    kept_features = scorecard[scorecard['AUC'] > 0.5]
    print(f"I am discarding {total_features - len(kept_features)} features from the initial pool of {total_features} based on AUC <= 0.55.")
    discarded = scorecard[scorecard['AUC'] <= 0.5]
    sig_but_low_auc = discarded[discarded['p_value'] < 0.05]
    if len(sig_but_low_auc) > 0:
        print(f"Warning: You discarded {len(sig_but_low_auc)} statistically significant features!")
    X_filtered = X_scaled[kept_features.index]
    X_filtered.to_csv("/Volumes/T7 Shield/Restrepo/HCC_Retrospective_radiomics/Results/main_pipeline/filtered_features_fold.csv")

    # Spearman
    initial_n = X_filtered.shape[1]
    p_vals = scorecard.loc[X_filtered.columns, 'p_value']
    X_spearman = drop_redundant_features(X_filtered, threshold=0.6, tie_breaker_series=p_vals)
    kept_features = X_spearman.columns
    final_n = X_spearman.shape[1]
    print(f"Spearman: Dropped {initial_n - final_n} features (from {initial_n} to {final_n})")
    X_spearman.to_csv("/Volumes/T7 Shield/Restrepo/HCC_Retrospective_radiomics/Results/main_pipeline/spearman_features_fold.csv")

    # LASSO
    lasso_cv = LogisticRegression(penalty='l1', solver='liblinear', C=0.2, random_state=42)
    c_values = [0.1, 0.5, 1.0, 5.0, 10.0]
    lasso_cv = LogisticRegressionCV(
    Cs=c_values,
    penalty='l1',
    solver='liblinear',
    cv=3,                 # Inner loop (5-fold CV for tuning)
    scoring='roc_auc',
    random_state=42,
    max_iter=1000
    )
    lasso_cv.fit(X_spearman, y_tr)
    print(f"Optimal C found: {lasso_cv.C_}")

    # Extract
    coefs = pd.Series(lasso_cv.coef_[0], index=X_spearman.columns)
    selected = coefs[coefs != 0].abs().sort_values(ascending=False).head(target_count)
    print(f"LASSO: Dropped {len(coefs) - len(selected)} features (from {len(coefs)} to {len(selected)})")
    return selected, imputer, scaler, kept_features, lasso_cv, scorecard

# --- 4. Main K-Fold Orchestrator ---
def run_cross_validation(X_dev, y_dev, target_count):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_dev, y_dev)):
        X_tr, X_va = X_dev.iloc[train_idx], X_dev.iloc[val_idx]
        y_tr, y_va = y_dev.iloc[train_idx], y_dev.iloc[val_idx]
        
        # Prune
        selected_coefs, imputer,  scaler, kept_features, lasso, scorecard = run_fold_pruning(X_tr, y_tr, target_count)
        
        # 2. Apply "Recipe" to Validation Data (NO LEAKAGE)
        X_va_imp = pd.DataFrame(imputer.transform(X_va), columns=X_va.columns, index=X_va.index)
        X_va_scaled = pd.DataFrame(scaler.transform(X_va_imp), columns=X_va.columns, index=X_va.index)
        X_va_final = X_va_scaled.reindex(columns=kept_features, fill_value=0) # Only keep features the training fold learned
        
        y_probs = lasso.predict_proba(X_va_final)[:, 1]
        auc = roc_auc_score(y_va, y_probs)

        # Store for analysis
        cv_results.append({
            'fold': fold + 1,
            'coefs': selected_coefs,
            'features': list(selected_coefs.index),
            'auc': auc
        })
        print(f"Fold {fold+1} complete. AUC: {auc:.3f}\n\n")
    #print(cv_results)

    # --- FLATTEN AFTER THE LOOP ---
    flattened_data = []
    for res in cv_results:
        for feat, coef in zip(res['features'], res['coefs']):
            flattened_data.append({
                'Fold': res['fold'],
                'Global_AUC': res['auc'],
                'Feature': feat,
                'Coefficient': coef
            })
    results_df = pd.DataFrame(flattened_data)     
    return results_df, scorecard   

### Introduce the classification/regression models that will be used after feature selection. #####

def get_model(model_name, random_state=42):
    """Factory function to return an initialized model."""
    if model_name == 'logistic':
        return LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)
    
    elif model_name == 'svm':
        # Probability=True is required for ROC-AUC
        return SVC(kernel='linear', probability=True, random_state=random_state)
    
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

###. Finally train and evaluate models with  the selected features. ####

def train_and_eval_model(X_train, y_train, X_val, y_val, model_name):
    """
    Fits a model and returns the fitted object and AUC score.
    """
    model = get_model(model_name)
    
    # Fit the model
    model.fit(X_train, y_train)
    
    # Predict probabilities (needed for AUC)
    y_probs = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_probs)
    
    return model, auc

# --- EXECUTION ---
if __name__ == "__main__":
    df_reliable = pd.read_csv("/Volumes/T7 Shield/Restrepo/HCC_Retrospective_radiomics/Results/main_pipeline/reliable_feature_pool.csv", index_col=0)
    outcomes_path = "/Volumes/T7 Shield/Restrepo/HCC_Retrospective_radiomics/databases/immunophenotype_outcomes.xlsx"
    out_dir = '/Volumes/T7 Shield/Restrepo/HCC_Retrospective_radiomics/Results/main_pipeline'
    X_dev, y_dev, X_lock, y_lock = partition_data(df_reliable, outcomes_path, label_col='TILs_resection', cutoff=3, test_size=20)

    results_df, scorecard = run_cross_validation(X_dev, y_dev, target_count=100)
    all_selected = results_df['Feature']
    feature_counts = pd.Series(all_selected).value_counts()
    # Apply consensus (e.g., >= 3 folds)
    consensus_features = feature_counts[feature_counts>=3].index.tolist()
    print(f"Consensus features identified: {len(consensus_features)}")
    print(consensus_features)

    scorecard.to_csv(f"{out_dir}/univariate_metrics.csv")
    pd.Series(consensus_features).to_csv(f"{out_dir}/consensus_features.csv", index=False)
    X_dev.to_csv(f"{out_dir}/X_dev.csv", index=True)

    ####### Final model run on test set #######

    # 1. Initialize preprocessors
    imputer = KNNImputer(n_neighbors=5)
    scaler = StandardScaler()

    # 2. Extract the features we agreed upon
    X_dev_subset = X_dev[consensus_features]
    X_lock_subset = X_lock[consensus_features]

    # 3. Process Dev Set (Fit AND Transform)
    X_dev_clean = pd.DataFrame(imputer.fit_transform(X_dev_subset), columns=consensus_features, index=X_dev.index)
    X_dev_scaled = pd.DataFrame(scaler.fit_transform(X_dev_clean), columns=consensus_features, index=X_dev.index)

    # 4. Process Lock Set (Only Transform using Dev's rules)
    X_lock_clean = pd.DataFrame(imputer.transform(X_lock_subset), columns=consensus_features, index=X_lock.index)
    X_lock_scaled = pd.DataFrame(scaler.transform(X_lock_clean), columns=consensus_features, index=X_lock.index)

    # 5. NOW train and evaluate
    model, auc = train_and_eval_model(X_dev_scaled, y_dev, X_lock_scaled, y_lock, model_name='logistic')

    print(f"Final Model AUC on Lock set: {auc:.3f}")