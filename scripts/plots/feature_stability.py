import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from pingouin import intraclass_corr

# Must match the sequence prefixes used during extraction (see radiomics_params.yaml).
# Matched via startswith(seq + "_") rather than a naive split('_')[0], since some
# sequences (dwi_b1, dwi_b2) contain an underscore and would otherwise get truncated
# to a bogus 'dwi' bucket.
SEQUENCES = ['pre', 'ap', 'pvp', 'tp', 'hbp', 't2', 'dwi_b1', 'dwi_b2', 'adc']


def _match_sequence(feat_lower):
    for seq in SEQUENCES:
        if feat_lower.startswith(seq + "_"):
            return seq
    return "Unknown"


def analyze_feature_susceptibility(csv_paths, icc_threshold=0.90):
    """
    Identifies which feature categories are MOST UNSTABLE (ICC <= threshold)
    across different sequences.
    """
    raw_frames = {path: pd.read_csv(path) for path in tqdm(csv_paths, desc="Loading CSVs")}

    # Restrict every variant to the same patients before computing ICC -- pingouin's
    # intraclass_corr expects a balanced target x rater design, and a patient missing
    # from even one variant (e.g. a failed extraction run) would silently unbalance it.
    common_ids = set.intersection(*(set(df['PatientID']) for df in raw_frames.values()))

    all_data = []
    for path, df in raw_frames.items():
        dropped = set(df['PatientID']) - common_ids
        if dropped:
            print(f"  {os.path.basename(path)}: dropping {len(dropped)} patient(s) absent from other variants")
        df = df[df['PatientID'].isin(common_ids)]
        melted = df.melt(id_vars=['PatientID'], var_name='Feature', value_name='Value')
        melted['VariantID'] = os.path.basename(path)
        all_data.append(melted)

    full_df = pd.concat(all_data)
    unique_features = full_df['Feature'].unique()

    icc_results = []
    for feat in tqdm(unique_features, desc="Calculating ICC"):
        try:
            feat_data = full_df[full_df['Feature'] == feat]
            icc = intraclass_corr(data=feat_data, targets='PatientID',
                                 raters='VariantID', ratings='Value')
            val = icc.set_index('Type').loc['ICC3', 'ICC']
        except Exception:
            val = np.nan

        # Categorization Logic
        feat_lower = feat.lower()
        seq = _match_sequence(feat_lower)

        # Define categories based on PyRadiomics naming conventions
        if 'shape' in feat_lower:
            cat = 'Shape'
        elif 'firstorder' in feat_lower:
            cat = 'First-Order'
        elif any(x in feat_lower for x in ['glcm', 'glszm', 'glrlm', 'gldm', 'ngtdm']):
            cat = 'Texture'
        else:
            cat = 'Other'

        # Check if it's a transformed feature (Wavelet/LoG)
        if any(x in feat_lower for x in ['wavelet', 'log-sigma']):
            cat = 'Transform'

        icc_results.append({'Feature': feat, 'Sequence': seq, 'Category': cat, 'ICC': val})

    results_df = pd.DataFrame(icc_results)

    # 1. FILTER FOR UNSTABLE FEATURES (The "Non-Survivors")
    unstable_df = results_df[results_df['ICC'] <= icc_threshold]

    # 2. Calculate the composition of the failures
    # We want to see: Of the features that FAILED, what % were Texture, Shape, etc.?
    summary = unstable_df.groupby(['Sequence', 'Category']).size().unstack(fill_value=0)
    summary_pct = summary.div(summary.sum(axis=1), axis=0) * 100

    # 3. Plotting the "Susceptibility Histogram"
    # Using a "Warm/Alert" colormap (like 'YlOrRd' or 'autumn') to signify instability
    summary_pct.plot(kind='bar', stacked=True, figsize=(12, 7), colormap='YlOrRd')
    
    plt.title(f'Anatomy of Instability: Composition of Failed Features (ICC <= {icc_threshold})', fontsize=14)
    plt.ylabel('Percentage of Unstable Features (%)')
    plt.xlabel('MRI Sequence')
    plt.legend(title='Feature Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.show()

    return summary_pct


if __name__ == '__main__':
    csv_dir = "/Volumes/T7 Shield/Restrepo/HCC_Retrospective_radiomics/Results/stability_analysis"
    settings = ["vary_binwidth10-50_R1_I-sitkBSpline_S-1.5mm.csv", "vary_reader_I-sitkBSpline_S-1.5mm_W-10.csv"]
    paths = [os.path.join(csv_dir, sub) for sub in settings]
    susceptibility_summary = analyze_feature_susceptibility(paths, icc_threshold=0.90)
    print(susceptibility_summary)