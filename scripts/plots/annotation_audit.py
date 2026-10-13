import os
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import SimpleITK as sitk
import matplotlib.pyplot as plt


def run_deep_sequence_audit(base_raw_path, folders):
    audit_data = []
    
    if not os.path.exists(base_raw_path):
        print(f"❌ Base path not found: {base_raw_path}")
        return pd.DataFrame()

    for folder in folders:
        reader_path = os.path.join(base_raw_path, folder)
        if not os.path.exists(reader_path):
            continue
        
        # Get all patient folders (P001, P002...)
        patient_folders = [d for d in os.listdir(reader_path) 
                           if os.path.isdir(os.path.join(reader_path, d)) and d[0]!='.']
        
        print(f"🕵️ Deep Auditing {len(patient_folders)} patients for {folder}...")

        for pid in tqdm(patient_folders, desc=f"Processing {folder}"):
            pid_folder = os.path.join(reader_path, pid)
            
            # Find all mask files in this patient's folder
            mask_files = [f for f in os.listdir(pid_folder) 
                          if '.seg.nrrd' in f and '_N4' not in f and f[0]!='.']
            
            for mask_file in mask_files:
                mask_path = os.path.join(pid_folder, mask_file)
                
                # Extract sequence name from filename (e.g., 'adc' from 'P001_adc.seg.nrrd')
                # Assumes format: PatientID_Sequence.seg.nrrd
                try:
                    parts = mask_file.split('_')
                    sequence = parts[1].replace('.seg.nrrd', '') if len(parts) > 1 else "unknown"
                except:
                    sequence = "unknown"

                try:
                    mask_img = sitk.ReadImage(mask_path)
                    mask_arr = sitk.GetArrayFromImage(mask_img)
                    
                    # Identify slices with label > 0
                    labeled_slices = [i for i, slc in enumerate(mask_arr) if slc.max() > 0]
                    slice_count = len(labeled_slices)
                    
                    audit_data.append({
                        'PatientID': pid,
                        'Reader': folder,
                        'Sequence': sequence,
                        'FileName': mask_file,
                        'Total_ROI_Slices': slice_count,
                        'Type': '2D' if slice_count == 1 else '3D',
                        'Slice_Index': labeled_slices[0] if slice_count == 1 else "N/A"
                    })
                except Exception as e:
                    print(f"❌ Error reading {mask_file}: {e}")

    return pd.DataFrame(audit_data)


def plot_slice_distribution(audit_csv_path):
    df = pd.read_csv(audit_csv_path)
    
    # Filter to only include real sequences (exclude 'unknown')
    df = df[df['Sequence'] != 'unknown']
    
    plt.figure(figsize=(12, 6))
    
    # Create a histogram showing the distribution of ROI slices
    sns.histplot(data=df, x='Total_ROI_Slices', bins=range(1, df['Total_ROI_Slices'].max() + 2), 
                 kde=False, color='skyblue', edgecolor='black')
    
    plt.axvline(1.5, color='red', linestyle='--', label='2D Threshold')
    plt.title('Distribution of Annotated Slices across 140 Patients (All Sequences)')
    plt.xlabel('Number of Slices with ROI')
    plt.ylabel('Count of Sequences')
    plt.xticks(range(1, df['Total_ROI_Slices'].max() + 1))
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    #plt.savefig("slice_count_distribution.png")
    plt.show()
    
    # Print the "2D vs 3D" breakdown
    summary = df['Type'].value_counts(normalize=True) * 100
    print(f"📈 Dataset Composition:")
    print(f"   - 3D (Volumetric): {summary.get('3D', 0):.1f}%")
    print(f"   - 2D (Single-Slice): {summary.get('2D', 0):.1f}%")



# --- EXECUTION ---
if __name__ == '__main__':
    base_dir = "/Volumes/T7 Shield/Restrepo/HCC_Retrospective_radiomics/Total_segmentations"
    results_dir = '/Volumes/T7 Shield/Restrepo/HCC_Retrospective_radiomics/Results/diagnostics'
    folders = ['SA1', 'SA2']
    
    audit_df = run_deep_sequence_audit(base_dir, folders)

    if not audit_df.empty:
        audit_df.to_csv(results_dir+"/full_annotation_audit.csv", index=False)
        # Save the "Problematic" 2D List
        single_slice_list = audit_df[audit_df['Type'] == '2D']
        single_slice_list.to_csv(results_dir+"/single_slice_sequences.csv", index=False)
        
        # Print a quick summary table
        print("\n📊 --- Audit Summary (Count of 2D vs 3D per Sequence) ---")
        summary = audit_df.groupby(['Sequence', 'Type']).size().unstack(fill_value=0)
        print(summary)
        print(results_dir+"/single_slice_sequences.csv")
        plot_slice_distribution(results_dir+"/full_annotation_audit.csv")
        
    else:
        print("❌ No masks found. Double-check the path on your T7 Shield.")

    




