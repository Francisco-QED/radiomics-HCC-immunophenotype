import os
import SimpleITK as sitk


def n4_correct_folder(base_dir, folders):
    """
    Applies N4 bias field correction to all .nrrd files in the raw_dir and saves them to output_dir.
    Skips files that are segmentations (contain 'seg.') or already have '_N4' in their name.
    """
    for folder in folders:
        patient_dir = os.path.join(base_dir, folder)
        patients = [x for x in os.listdir(patient_dir) if x[0]!='.']   

        for patient in patients:
            patient_path = os.path.join(patient_dir,patient)
            files = [x for x in os.listdir(patient_path) if x[0]!='.']
            print(files)
            for file in files:
                filename = os.path.join(patient_path,file)

                # Only process .nrrd files that ARE NOT segmentations or previous N4s
                if filename.endswith(".nrrd") and "seg." not in filename and "_N4" not in filename:
                    input_path = os.path.join(patient_dir, filename) 
                
                    # Define the new filename for the output folder
                    output_filename = filename.replace(".nrrd", "_N4.nrrd")
                    output_path = os.path.join(base_dir, folder, output_filename)
                    print(output_path)       
                    #print(f"Processing: {filename}...", end=" ", flush=True)
                
                    # Call the modular function
                    was_processed = apply_n4_correction(input_path, output_path)
                    
                    if was_processed:
                        print("✅ Done.")
                    else:
                        print("⏭️ Skipped (Exists).")

def apply_n4_correction(input_path, output_path, mask_path=None):
    """
    Modular N4 Correction.
    Returns: The path to the corrected image.
    """
    if os.path.exists(output_path):
        return output_path

    image = sitk.ReadImage(input_path)
    image = sitk.Cast(image, sitk.sitkFloat32)

    # Use existing segmentation as mask if provided, else generate Otsu mask
    if mask_path and os.path.exists(mask_path):
        mask = sitk.ReadImage(mask_path)
    else:
        mask = sitk.OtsuThreshold(image, 0, 1)

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    # High-reproducibility settings (standard iterations)
    corrector.SetMaximumNumberOfIterations([50, 50, 50, 50])
    
    corrected_image = corrector.Execute(image, mask)
    sitk.WriteImage(corrected_image, output_path)
    
    return output_path

# --- EXECUTION ---

base_dir = "/Volumes/T7 Shield/Restrepo/HCC_Retrospective_radiomics/Total_segmentations"
folders = ['SA1']

if __name__ == '__main__':
    n4_correct_folder(base_dir, folders)