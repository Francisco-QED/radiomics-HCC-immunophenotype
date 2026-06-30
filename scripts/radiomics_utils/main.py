import os
import yaml
import warnings
import pandas as pd
from tqdm import tqdm
import multiprocessing
#import radiomics_utils
from radiomics_utils.io import get_patient_files
from radiomics_utils.stability import calculate_stability_scorecard
from radiomics_utils.snr_binwidth import extract_snr_matrix_single_folder
from radiomics_utils.feature_stability import analyze_feature_susceptibility
from radiomics_utils.icc_spider_plot import visualize_sequence_stability_spider
from radiomics_utils.n4_correction import apply_n4_correction, n4_correct_folder
from radiomics_utils.extraction import extract_features_single, run_batch_extraction
from radiomics_utils.binwidth_texture_gallery import export_tumor_binning_comparison_3d
from radiomics_utils.annotation_audit import run_deep_sequence_audit, plot_slice_distribution
from radiomics_utils.missing_sequences_spider import audit_mri_availability, plot_missing_mri_spider

warnings.filterwarnings('ignore', category=RuntimeWarning, module='pingouin')
warnings.filterwarnings('ignore', message='divide by zero encountered in scalar divide')
warnings.filterwarnings('ignore', message='invalid value encountered in scalar divide')

#################################################################
############### Basic diagnostics #############################
################################################################

##### Check missing sequences ##########

# if __name__ == '__main__':
#     base_dir = "/Volumes/T7 Shield/Restrepo/HCC_Retrospective_radiomics/Total_segmentations"
#     out_dir = '/Volumes/T7 Shield/Restrepo/HCC_Retrospective_radiomics/Results/diagnostics'
#     folders = ['SA1', 'SA2']
#     sequences = ['pre', 'ap', 'pvp', 'tp', 'hbp', 't2', 'dwi_b1', 'dwi_b2', 'adc']
#     mri_df, total_pats = audit_mri_availability(base_dir, folders, sequences)
#     if not mri_df.empty:
#         plot_missing_mri_spider(mri_df, total_pats)
#         # Save a list of exactly which patients are missing which scans
#         #missing_list = mri_df[mri_df['Available'] == 0]
#         #missing_list.to_csv(out_dir+"/patients_missing_mri_scans.csv", index=False)
#         print(f"✅ Summary: Found {total_pats} total patients.")
#         print(f"📂 Detailed gaps saved to 'patients_missing_mri_scans.csv'")

#-------------------------------------------------------------------------------------------------#

#### Run annotation audit (identify and count segmented slices per sequence) ####

# if __name__ == '__main__':
#     base_dir = "/Volumes/T7 Shield/Restrepo/HCC_Retrospective_radiomics/Total_segmentations"
#     results_dir = '/Volumes/T7 Shield/Restrepo/HCC_Retrospective_radiomics/Results/diagnostics'
#     folders = ['SA1', 'SA2']
    
#     audit_df = run_deep_sequence_audit(base_dir, folders)

#     if not audit_df.empty:
#         audit_df.to_csv(results_dir+"/full_annotation_audit.csv", index=False)
#         # Save the "Problematic" 2D List
#         single_slice_list = audit_df[audit_df['Type'] == '2D']
#         single_slice_list.to_csv(results_dir+"/single_slice_sequences.csv", index=False)
        
#         # Print a quick summary table
#         print("\n📊 --- Audit Summary (Count of 2D vs 3D per Sequence) ---")
#         summary = audit_df.groupby(['Sequence', 'Type']).size().unstack(fill_value=0)
#         print(summary)
#         print(results_dir+"/single_slice_sequences.csv")
#         plot_slice_distribution(results_dir+"/full_annotation_audit.csv")
        
#     else:
#         print("❌ No masks found. Double-check the path on your T7 Shield.")

####################################################################
##############      N4 correction module        ####################
####################################################################

# base_dir = "/Volumes/T7 Shield/Restrepo/HCC_Retrospective_radiomics/Total_segmentations"
# output_dir = "/Volumes/T7 Shield/Restrepo/HCC_Retrospective_radiomics/Total_segmentations/N4_corrected"
# folders = ['SA1', 'SA2']
# if __name__ == '__main__':
#     n4_correct_folder(base_dir, folders, output_dir)

########################################################################################
######################  Radiomics extraction  ###########################################
########################################################################################

# Here, the user has the option to choose from a variety of settings, including interpolation scheme, binwidht, pixel spacing, etc. 
# The function then generated radiomics features for all these settings. 

# # --- Radiomics extraction for subset -----
# if __name__ == '__main__':
#     # Load config once
#     with open("/Users/Francisco/Desktop/RestrepoProjects/Active/Regeneron retrospective radiomics/scripts/radiomics_params.yaml", "r") as f:
#         config = yaml.safe_load(f)
        
#     run_batch_extraction(
#         base_dir="/Volumes/T7 Shield/Restrepo/HCC_Retrospective_radiomics/Total_segmentations/Inter-reader",
#         results_dir="/Volumes/T7 Shield/Restrepo/HCC_Retrospective_radiomics/Results/Radiomics_sheets2",
#         config=config
#     )

# ####################################################################
# ##########       Stability analysis module       #################
# ####################################################################

# Stability is asssessed by regarding each feature (f) as a function of 4 variables:
# reader, interpolator algorithm, re-sampling spacing, and bin_wdithds. 
# Then, each quantity will be allowed to assume multiple values and radiomics features will be extracted for each combination
# This will be done in a subset of the 140 patients (15-30 patients)

# ---- Stability evaluation ------- #

## Here, I will consider the ICC between pairs of features. That is, I will vary one variables and fix the other 3 independent varaibles,
## and I will calculate the ICC for the features. Maybe later I'll conduct a larger multivariable analysis to pick out the features that simultaneously robust to
## variations in all 4 variables

# csv_dir = "/Volumes/T7 Shield/Restrepo/HCC_Retrospective_radiomics/Results/Radiomics_sheets"
# icc_dir = "/Volumes/T7 Shield/Restrepo/HCC_Retrospective_radiomics/Results/icc"
# # Choose the variables to compare
# my_variants = [
#     "R-Reader1_I-sitkBSpline_S-1.5mm_W-10.csv",
#     "R-Reader1_I-sitkBSpline_S-1.5mm_W-25.csv",
# ]
# paths = [os.path.join(csv_dir, sub) for sub in my_variants]

# stability_scores = calculate_stability_scorecard(paths)

# # Only keep the 'Indestructible' features
# golden_features = stability_scores[stability_scores > 0.85].index.tolist()
# print(f"💎 {len(golden_features)} features are stable across all parameters.")

# stability_scores.to_csv(icc_dir+"/global_stability_scorecard.csv", header=['ICC_Score'])

# Plot Spider plot of ICC ####
# if __name__ == '__main__':
#     # Adjust this path to point to your saved scorecard CSV
#     scorecard_path = "/Volumes/T7 Shield/Restrepo/HCC_Retrospective_radiomics/Results/icc/vary_binwidth10-25_R1_I-sitkBSpline_S-1.5mm.csv" 
    
#     if os.path.exists(scorecard_path):
#         visualize_sequence_stability_spider(scorecard_path, threshold=0.85)
#     else:
#         print(f"❌ Could not find scorecard at: {scorecard_path}")



# --------- SNR auditor ---------------#

# FOLDER_PATH = "/Volumes/T7 Shield/Restrepo/HCC_Retrospective_radiomics/Total_segmentations/liver_segs" 
# SEQS = ['adc', 't2', 'hbp', 'ap', 'pre', 'pvp', 'tp', 'dwi_b1', 'dwi_b2']

# audit_results = extract_snr_matrix_single_folder(FOLDER_PATH, SEQS)
# # Save to CSV in the same folder
# audit_results.to_csv(os.path.join(FOLDER_PATH, "Liver_Noise_Audit_Results.csv"), index=False)
# print(audit_results.tail(1)) # See the recommended bin widths

# PATIENT_DIR = "/Volumes/T7 Shield/Restrepo/HCC_Retrospective_radiomics/Total_segmentations/liver_segs/P043"
# OUTPUT_DIR = '/Volumes/T7 Shield/Restrepo/HCC_Retrospective_radiomics/Total_segmentations/binning_analysis'
# # Specific sequences you requested
# TARGET_SEQS = ['adc', 't2', 'hbp', 'ap', 'pre', 'pvp', 'tp'] 
# export_tumor_binning_comparison_3d(PATIENT_DIR, OUTPUT_DIR, TARGET_SEQS)

