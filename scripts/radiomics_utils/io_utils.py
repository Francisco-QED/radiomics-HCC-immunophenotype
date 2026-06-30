import os
from glob import glob

def get_patient_files(base_path, patient_id):
    """
    Finds all NRRD sequences and segmentations for a specific patient.
    Returns a dict: {'t1_pre': 'path/to/img', 't1_pre_seg': 'path/to/seg', ...}
    """
    # Look for all .nrrd files for this patient
    pattern = os.path.join(base_path, f"{patient_id}_*.nrrd")
    all_files = glob(pattern)
    
    file_map = {}
    
    for filepath in all_files:
        filename = os.path.basename(filepath)
        
        # Determine the sequence key (e.g., 't1_pre' or 't2wi')
        # This assumes format: P001_t1_pre.nrrd or P001_t1_pre.seg.nrrd
        parts = filename.replace(f"{patient_id}_", "").split('.')
        seq_name = parts[0]
        
        if "seg" in filename:
            file_map[f"{seq_name}_seg"] = filepath
        else:
            file_map[seq_name] = filepath
            
    return file_map