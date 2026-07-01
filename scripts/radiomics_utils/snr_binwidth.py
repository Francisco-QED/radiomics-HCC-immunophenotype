import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage
from tqdm import tqdm

# Must match extraction_settings.normalizeScale in radiomics_params.yaml. PyRadiomics
# z-scores each whole image (mean 0, std 1) and multiplies by this scale before applying
# binWidth, so noise has to be measured in that same normalized space to be comparable
# to the bin width it will feed.
NORMALIZE_SCALE = 100


def _pooled_patch_sigma(intensities, mask):
    """
    Robust per-patient noise estimate. The delineated ROI is several disjoint patches
    spread across slices (drawn away from vessels/heterogeneous tissue), so a single
    global std over all pooled voxels would conflate true point-wise noise with real
    between-patch differences (residual bias field, slight tissue/slice drift). Instead,
    treat each connected patch as its own group and pool the WITHIN-group variance only
    -- this cancels out any between-patch mean shift and isolates the actual voxel-to-
    voxel noise.
    """
    labeled_mask, n_patches = ndimage.label(mask > 0)
    if n_patches <= 1:
        return np.std(intensities[mask > 0])

    ss_within = 0.0
    dof_within = 0
    for label_id in range(1, n_patches + 1):
        patch_vals = intensities[labeled_mask == label_id]
        if len(patch_vals) < 2:
            continue
        ss_within += ((patch_vals - patch_vals.mean()) ** 2).sum()
        dof_within += len(patch_vals) - 1

    if dof_within == 0:
        return np.std(intensities[mask > 0])

    return np.sqrt(ss_within / dof_within)


def extract_snr_matrix_single_folder(folder_path, sequences, normalize_scale=NORMALIZE_SCALE):
    """
    Analyzes a folder of calibration patients. For each patient/sequence, estimates
    voxel-level noise (sigma) from homogeneous liver patches, expressed in the same
    normalized units PyRadiomics discretizes on (see NORMALIZE_SCALE). Recommends a
    per-sequence bin width as 2x the MEDIAN sigma across the calibration cohort --
    median instead of max, so a single atypical calibration patient doesn't set the
    bin width for the whole sequence.
    """
    all_data = []
    patients = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]

    for pid in tqdm(patients, desc="Analyzing Patient Noise"):
        pid_dir = os.path.join(folder_path, pid)
        patient_row = {'PatientID': pid}
        files = os.listdir(pid_dir)

        for seq in sequences:
            # Matched by filename convention (contains "n4"), regardless of whether N4
            # bias correction was actually applied to that sequence.
            img_file = next((f for f in files if seq.lower() in f.lower()
                             and "n4" in f.lower()
                             and f.endswith('.nrrd')), None)
            seg_file = next((f for f in files if seq.lower() in f.lower()
                             and "liver" in f.lower()
                             and f.endswith('.nrrd')), None)

            if not (img_file and seg_file):
                continue

            try:
                img_arr = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(pid_dir, img_file))).astype(np.float64)
                mask_arr = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(pid_dir, seg_file)))

                liver_voxels = img_arr[mask_arr > 0]
                if len(liver_voxels) <= 10:
                    continue

                # PyRadiomics' normalize=True z-scores over the WHOLE image (not just
                # the ROI) before binning, so noise must be rescaled into that space.
                global_std = img_arr.std()
                if global_std == 0:
                    continue

                raw_sigma = _pooled_patch_sigma(img_arr, mask_arr)
                patient_row[f'{seq.upper()}_Sigma'] = raw_sigma * (normalize_scale / global_std)
            except Exception:
                patient_row[f'{seq.upper()}_Sigma'] = np.nan

        all_data.append(patient_row)

    df = pd.DataFrame(all_data)

    sigma_cols = [c for c in df.columns if "_Sigma" in c]
    median_sigmas = df[sigma_cols].median()
    opt_bin_widths = (median_sigmas * 2).round(2)

    summary_row = {'PatientID': 'OPTIMAL_BINWIDTH_RECOMMENDED'}
    for col in sigma_cols:
        summary_row[col] = opt_bin_widths[col]

    df = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)
    return df


# --- EXECUTION ---
# Path to the folder containing your calibration patient subfolders
FOLDER_PATH = '/Volumes/T7 Shield/Restrepo/HCC_Retrospective_radiomics/Total_segmentations/liver_segs'
SEQS = ['adc', 't2', 'hbp', 'ap', 'pre', 'pvp', 'tp', 'dwi_b1', 'dwi_b2']

if __name__ == '__main__':
    audit_results = extract_snr_matrix_single_folder(FOLDER_PATH, SEQS)
    print(audit_results.tail(1))  # Recommended bin widths (normalized units)
    audit_results.to_csv(os.path.join(FOLDER_PATH, "snr_binwidth_recommendations.csv"), index=False)
