import pandas as pd
import numpy as np
from tqdm import tqdm

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
    r1 = df_reader1.set_index(id_col).loc[common_ids].sort_index()
    r2 = df_reader2.set_index(id_col).loc[common_ids].sort_index()

    features = [f for f in r1.columns if f in r2.columns and r1[f].dtype != 'object']
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
    return df_main[stable]

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