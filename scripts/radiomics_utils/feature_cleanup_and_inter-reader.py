import pandas as pd
import numpy as np
from tqdm import tqdm
from pingouin import intraclass_corr

# First, start with some prelimimanry to remove features with no variance or that are all NaN. This global cleanup can be applied to the global cohort (140 patients or so)


def stage1_global_cleanup(df, id_col='PatientID'):
    """Robust cleaning for Radiomics features."""
    # Set Index
    if id_col in df.columns:
        df = df.set_index(id_col)
    # Force numeric-only
    df = df.select_dtypes(include=[np.number])
    # (Replace Inf with NaN so they can be dropped/imputed)
    df = df.replace([np.inf, -np.inf], np.nan)
    # REMOVE ROWS WITH ALL NaNs
    # 'how=all' ensures we only drop patients if every single feature is empty.
    initial_rows = len(df)
    df = df.dropna(how='all')
    if len(df) < initial_rows:
        print(f"Dropped {initial_rows - len(df)} rows that were entirely empty.")
    # Drop columns that are mostly NaN (e.g., > 10% missing)
    # Threshold adjustable
    thresh = len(df) * 0.9 
    df = df.dropna(axis=1, thresh=thresh)
    # Drop low-variance columns
    df = df.loc[:, df.std() > 0.1]
    
    print(f"Cleanup finished. Columns remaining: {df.shape[1]}")
    return df

# Before we even get to K-Fold, we can do a vetting step based on inter-reader reliability. This will be a more stringent filter that 
# only keeps features that show good agreement (e.g., ICC > 0.85) between the two readers on the 20 patient subset. For this we need 
# separate observations from both readers on the same patients, so we can calculate the ICC for each feature and only keep those that are stable.

def filter_dataset_by_reliability(df_main, df_reader1, df_reader2, id_col='PatientID', threshold=0.85):
    """Subsets the main dataset to only include ICC-stable features."""
    common_ids = np.intersect1d(df_reader1[id_col], df_reader2[id_col])
    r1_sub = df_reader1.set_index(id_col).loc[common_ids].sort_index()
    r2_sub = df_reader2.set_index(id_col).loc[common_ids].sort_index()
    
    features = [f for f in r1_sub.columns if f in r2_sub.columns and r1_sub[f].dtype != 'object']
    stable_features = []

    for feat in tqdm(features, desc="Calculating ICCs", unit="feature"):
        data = pd.DataFrame({
            'targets': list(common_ids) * 2,
            'raters': ['R1'] * len(common_ids) + ['R2'] * len(common_ids),
            'values': pd.concat([r1_sub[feat], r2_sub[feat]])
        })
        try:
            icc = intraclass_corr(data=data, targets='targets', raters='raters', ratings='values')
            if icc.set_index('Type').loc['ICC3', 'ICC'] >= threshold:
                stable_features.append(feat)
        except:
            continue
            
    return df_main[stable_features]

# --- PROPER EXECUTION BLOCK ---
if __name__ == "__main__":
    # 1. Define paths
    base_dir = '/Volumes/T7 Shield/Restrepo/HCC_Retrospective_radiomics/Results/Radiomics_sheets'
    path_main = f"{base_dir}/SA1/R-SA1_I-sitkBSpline_S-1mm_W-5.csv"
    path_r1 = f"{base_dir}/subset/R-Reader1_I-sitkBSpline_S-1mm_W-5.csv"
    path_r2 = f"{base_dir}/subset/R-Reader2_I-sitkBSpline_S-1mm_W-5.csv"
    out_dir = '/Volumes/T7 Shield/Restrepo/HCC_Retrospective_radiomics/Results/main_pipeline'

    # 2. Load Data
    df_total = pd.read_csv(path_main)
    df_r1 = pd.read_csv(path_r1)
    df_r2 = pd.read_csv(path_r2)
    
    # 3. Run Cleaning
    print("Step 1: Running Global Cleanup...")
    df_clean = stage1_global_cleanup(df_total)
    
    # 4. Run Reliability Filter
    print("Step 2: Vetting features via Inter-Reader Reliability...")
    df_reliable = filter_dataset_by_reliability(df_clean, df_r1, df_r2, id_col='PatientID', threshold=0.85)
    
    # 5. Save or proceed to K-Fold
    print(f"Final pool of reliable features: {df_reliable.shape[1]}")
    df_reliable.to_csv(f"{out_dir}/reliable_feature_pool.csv", index='PatientID')