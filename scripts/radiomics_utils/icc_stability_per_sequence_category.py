import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_instability_from_scorecard(scorecard_path, icc_threshold):
    """
    Plots the composition of unstable features using a pre-calculated ICC scorecard.
    Expects a CSV with columns: 'Feature' and 'ICC'
    """
    # 1. Load your pre-calculated results
    df = pd.read_csv(scorecard_path)
    
    # Ensure columns are named correctly for logic below
    # If your CSV uses different column names, adjust these:
    if 'Feature' not in df.columns or 'ICC' not in df.columns:
        raise ValueError("CSV must contain 'Feature' and 'ICC' columns.")

    sequences = ['pre', 'ap', 'pvp', 'tp', 'hbp', 't2', 'dwi_b1', 'dwi_b2', 'adc']

    # Update the logic for Sequence and Category
    def categorize_feature(feat_str):
        feat_lower = feat_str.lower()
        
        # Match sequence
        seq_match = "Unknown"
        for seq in sequences:
            if feat_lower.startswith(seq.lower() + "_"):
                seq_match = seq
                break
        
        # Match category
        if 'shape' in feat_lower: cat = 'Shape'
        elif 'firstorder' in feat_lower: cat = 'First-Order'
        elif any(x in feat_lower for x in ['glcm', 'glszm', 'glrlm', 'gldm', 'ngtdm']): cat = 'Texture'
        #elif any(x in feat_lower for x in ['wavelet', 'log-sigma']): cat = 'Transform'
        else: cat = 'Other'

        return seq_match, cat

    # Apply both at once
    df[['Sequence', 'Category']] = df['Feature'].apply(
        lambda x: pd.Series(categorize_feature(x))
    )
    print(df)
   
    total_counts = df.groupby(['Sequence', 'Category']).size()

    # Stable counts (only the ones that pass the threshold)
    stable_counts = df[df['ICC'] >= icc_threshold].groupby(['Sequence', 'Category']).size()

    stability_rate = (stable_counts.div(total_counts, fill_value=0) * 100).unstack(fill_value=0)
    desired_order = ['Shape', 'First-Order', 'Texture']
    stability_rate = stability_rate[[c for c in desired_order if c in stability_rate.columns]]

    # Now plot stability_rate instead of summary_pct
    stability_rate.plot(kind='bar', stacked=False, figsize=(12, 7), colormap='jet')

    plt.title(f'Stability Success Rate per Category (ICC >= {icc_threshold})', fontsize=14)
    plt.ylabel('Percentage of Features that are Stable (%)')
    plt.ylim(0, 105) # Added headroom for the 100% mark
    plt.show()
    return df

if __name__ == '__main__':
    # Path to your existing scorecard file
    base_dir = "/Volumes/T7 Shield/Restrepo/HCC_Retrospective_radiomics/Results/stability_analysis"
    scorecard_path = os.path.join(base_dir, "R-Reader1_I-sitkBSpline_S-1mm-1.5mm_W-10.csv")
    
    flop = plot_instability_from_scorecard(scorecard_path, 0.85)
    #flop.to_csv("/Volumes/T7 Shield/Restrepo/HCC_Retrospective_radiomics/Results/stability_analysis/R-Reader1_I-sitkBSpline_S-1.5mm_W-10-25_composition.csv", index=False)
