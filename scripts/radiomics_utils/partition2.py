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

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def partition_with_constraints(features_path, outcomes_path, icc_path, label_col, test_size=20, output_dir='.'):
    """
    Partitions data ensuring ICC patients are in Dev, while maintaining stratification.
    """
    # 1. Load Data
    df_raw = pd.read_csv(features_path, index_col=0) # X
    df_outcomes = pd.read_excel(outcomes_path)       # Y + labels
    df_outcomes['y_bin'] = (df_outcomes[label_col] >= 3).astype(int)
    df_icc = pd.read_csv(icc_path)                 # List of patients with double ROIs
    
    # 2. Merge Features and Outcomes
    df_merged = df_raw.reset_index().merge(
        df_outcomes[['PatientID', 'y_bin']], on='PatientID', how='inner'
    ).set_index('PatientID')
    
    # 3. Separate the pools
    icc_ids = df_icc['PatientID'].unique()
    
    # Dev-Must: Patients with double ROIs
    df_dev_must = df_merged[df_merged.index.isin(icc_ids)]
    
    # Flexible: Everyone else
    df_rest = df_merged[~df_merged.index.isin(icc_ids)]
    
    # 4. Stratified Split on the Flexible Pool
    # We need to calculate how many we need from the flexible pool to reach total test_size
    # (Assuming we want the total test set to be exactly test_size)
    remaining_needed = test_size
    
    # If the ICC pool is smaller than the test size, we take the rest from df_rest
    # But wait, ICC patients MUST be in dev. So we check:
    if len(df_rest) < test_size:
        raise ValueError(f"Not enough patients to form a test set of size {test_size} after excluding ICC patients.")
        
    X_rest = df_rest.drop(columns=['y_bin'])
    y_rest = df_rest['y_bin']
    
    # Split the flexible pool
    X_dev_rest, X_lock, y_dev_rest, y_lock = train_test_split(
        X_rest, y_rest, 
        test_size=test_size, 
        stratify=y_rest, 
        random_state=42
    )
    
    # 5. Combine to form Final Sets
    # X_dev = ICC Pool + The remaining flexible pool
    X_dev = pd.concat([df_dev_must.drop(columns=['y_bin']), X_dev_rest])
    y_dev = pd.concat([df_dev_must['y_bin'], y_dev_rest])
    
    # 6. Save Artifacts
    dev_data = X_dev.copy()
    #dev_data['y_bin'] = y_dev
    dev_data.to_csv(f"{output_dir}/X_dev.csv")
    
    lock_data = X_lock.copy()
    #lock_data['y_bin'] = y_lock
    lock_data.to_csv(f"{output_dir}/X_lock.csv")
    
    print(f"Partitioning Complete.")
    print(f"Dev Set: {len(X_dev)} (Includes {len(df_dev_must)} ICC patients)")
    print(f"Lock Set: {len(X_lock)}")
    
    return X_dev, y_dev, X_lock, y_lock








#--- Execution ---
X_dev, y_dev, X_lock, y_lock = partition_with_constraints(
features_path="/Volumes/T7 Shield/Restrepo/HCC_Retrospective_radiomics/Results/Radiomics_sheets/SA1/R-SA1_I-sitkBSpline_S-1mm_W-5.csv",
outcomes_path="/Volumes/T7 Shield/Restrepo/HCC_Retrospective_radiomics/databases/immunophenotype_outcomes.xlsx",
icc_path="/Volumes/T7 Shield/Restrepo/HCC_Retrospective_radiomics/Results/stability_diagnostics_inter-reader/clean_radiomics_sheets/R-Reader1_I-sitkBSpline_S-1mm_W-5_clean.csv",
label_col='TILs_resection',
test_size=15,
output_dir='/Volumes/T7 Shield/Restrepo/HCC_Retrospective_radiomics/Results/main_pipeline/radiomics_sheets/development_test_partition'
)