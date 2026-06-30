import os
import SimpleITK as sitk
import numpy as np
import pandas as pd
from tqdm import tqdm

def extract_snr_matrix_single_folder(folder_path, sequences):
    """
    Analyzes a single folder of 10-15 patients.
    Returns a DataFrame with patients as rows and Sequence Sigmas as columns.
    """
    all_data = []
    
    # Get all patient directories in the provided path
    patients = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
    
    for pid in tqdm(patients, desc="Analyzing Patient Noise"):
        pid_dir = os.path.join(folder_path, pid)
        patient_row = {'PatientID': pid}
        
        files = os.listdir(pid_dir)
        
        for seq in sequences:
            # Match the MRI file (contains seq name, but NOT 'liver')
            img_file = next((f for f in files if seq.lower() in f.lower() 
                             and "n4" in f.lower() 
                             and f.endswith('.nrrd')), None)
            print(img_file)
            
            # Match the Liver Mask (contains seq name AND 'liver')
            seg_file = next((f for f in files if seq.lower() in f.lower() 
                             and "liver" in f.lower() 
                             and f.endswith('.nrrd')), None)
            
            if img_file and seg_file:
                try:
                    img = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(pid_dir, img_file)))
                    mask = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(pid_dir, seg_file)))
                    
                    # Grab all manual patches from the liver mask
                    liver_voxels = img[mask > 0]
                    
                    if len(liver_voxels) > 10:
                        # We use Standard Deviation (Sigma) as our noise proxy
                        patient_row[f'{seq.upper()}_Sigma'] = np.std(liver_voxels)
                except Exception:
                    patient_row[f'{seq.upper()}_Sigma'] = np.nan
        
        all_data.append(patient_row)

    # 1. Create the DataFrame
    df = pd.DataFrame(all_data)
    
    # 2. Calculate the "Optimal" Bin Width row (2 * Max Sigma)
    sigma_cols = [c for c in df.columns if "_Sigma" in c]
    
    # We use .max() across the 10-15 patients to find the "Worst Case" noise for each sequence
    max_sigmas = df[sigma_cols].max()
    opt_bin_widths = (max_sigmas * 2).round(2)
    
    # Create the Summary Row
    summary_row = {'PatientID': 'OPTIMAL_BINWIDTH_RECOMMENDED'}
    for col in sigma_cols:
        summary_row[col] = opt_bin_widths[col]
    
    # 3. Finalize Table
    df = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)
    
    
    
    return df

# --- EXECUTION ---
# Path to the folder containing your 10-15 patient subfolders
FOLDER_PATH = "/Volumes/T7 Shield/Restrepo/Calibration_Set" 
SEQS = ['adc', 't2', 'hbp', 'ap', 'pre', 'pvp', 'tp', 'dwi_b1', 'dwi_b2']

# audit_results = extract_snr_matrix_single_folder(FOLDER_PATH, SEQS)
# print(audit_results.tail(1)) # See the recommended bin widths