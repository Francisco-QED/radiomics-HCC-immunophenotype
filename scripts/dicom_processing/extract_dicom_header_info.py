
import SimpleITK as sitk
from pathlib import Path
import pandas as pd

def extract_dicom_metadata(root_path, target_keys=None):
    """
    Traverses patient folders, finds the first .nrrd file, 
    and extracts specified or all metadata keys.
    
    Args:
        root_path (str): Path to the directory containing patient subfolders.
        target_keys (list, optional): Specific metadata keys to extract. 
                                      If None, extracts all found keys.
    """
    data = []
    root = Path(root_path)

    # Iterate through subdirectories
    for patient_dir in [p for p in root.iterdir() if p.is_dir()]:
        # Find first .nrrd file
        nrrd_files = list(patient_dir.glob("*.nrrd"))
        
        if not nrrd_files:
            print(f"No .nrrd found in {patient_dir.name}")
            continue
            
        file_path = nrrd_files[0]
        
        try:
            # Read image metadata without loading full pixel data
            reader = sitk.ImageFileReader()
            reader.SetFileName(str(file_path))
            reader.LoadPrivateTagsOn()
            reader.ReadImageInformation()
            
            # Dictionary to store patient info
            patient_info = {"PatientID": patient_dir.name, "File": file_path.name}
            
            # Get all keys or filter by target_keys
            keys = target_keys if target_keys else reader.GetMetaDataKeys()
            
            for key in keys:
                patient_info[key] = reader.GetMetaData(key)
                
            data.append(patient_info)
            
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    return pd.DataFrame(data)

# --- Usage ---
root_directory = "/Volumes/T7 Shield/Restrepo/HCC_Retrospective_radiomics/Total_segmentations/SA1"
# Define specific DICOM tags if known, e.g., ['0010|0010', '0008|0060']
keys_to_extract = None 

df = extract_dicom_metadata(root_directory, target_keys=['0018,0087'])

# Save to CSV for review
df.to_excel("/Volumes/T7 Shield/Restrepo/HCC_Retrospective_radiomics/databases/extracted_dicom_metadata.xlsx", index=False)
print(df.head())