import os

def generate_merged_filename(filenames):
    # 1. Split path and keep the filename
    name1 = os.path.basename(filenames[0]).replace('.csv', '')
    ext = '.csv'
    
    parts1 = name1.split('_')
    parts2 = os.path.basename(filenames[1]).replace('.csv', '').split('_')
    
    new_filename_parts = []
    
    for p1, p2 in zip(parts1, parts2):
        if p1 == p2:
            # If the part is identical, keep it
            new_filename_parts.append(p1)
        else:
            # We found the change! 
            # Separate the prefix (e.g., 'W' or 'I') from the value
            prefix1, val1 = p1.split('-', 1)
            _, val2 = p2.split('-', 1)
            
            # --- Special Handling for Interpolator (sitkBSpline vs sitkBLinear) ---
            if prefix1 == 'I' and 'sitk' in val1 and 'sitk' in val2:
                # This logic cleans 'sitkBLinear' to 'Linear'
                clean_val2 = val2.replace('sitkB', '') 
                merged_val = f"{val1}-{clean_val2}"
            else:
                # Standard merging for other parameters (e.g., W-10 vs W-25 -> W-10-25)
                merged_val = f"{val1}-{val2}"
                
            new_filename_parts.append(f"{prefix1}-{merged_val}")
            
    return "_".join(new_filename_parts) + ext

# --- Testing the logic ---
files1 = ["R-Reader1_I-sitkBSpline_S-1.5mm_W-10.csv", "R-Reader1_I-sitkBLinear_S-1.5mm_W-25.csv"]

print(generate_merged_filename(files1)) 


