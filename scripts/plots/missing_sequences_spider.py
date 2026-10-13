import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from tqdm import tqdm

def audit_mri_availability(base_path, folders, sequences):
    availability_data = []
    
    # Track unique patients found across all AIMs
    all_patients = set()

    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        if not os.path.exists(folder_path):
            print(f"⚠️ Path not found: {folder_path}")
            continue
            
        patient_folders = [d for d in os.listdir(folder_path) 
                           if os.path.isdir(os.path.join(folder_path, d))]
        
        print(f"📂 Auditing {folder}: {len(patient_folders)} patients...")

        for pid in tqdm(patient_folders, desc=f"Checking {folder}"):
            all_patients.add(pid)
            pid_folder = os.path.join(folder_path, pid)
            
            for seq in sequences:
                # Target filename: P001_adc.nrrd (Raw MRI, not segmentation)
                target_file = f"{pid}_{seq}.nrrd"
                file_path = os.path.join(pid_folder, target_file)
               
                # Check if the raw MRI file exists
                exists = os.path.exists(file_path)
                
                availability_data.append({
                    'PatientID': pid,
                    'Folder': folder,
                    'Sequence': seq,
                    'Available': 1 if exists else 0
                })

    return pd.DataFrame(availability_data), len(all_patients)

def plot_missing_mri_spider(df, total_n):
    # Calculate missing counts per sequence
    stats = []
    sequences = df['Sequence'].unique()
    
    for seq in sequences:
        present_count = df[(df['Sequence'] == seq) & (df['Available'] == 1)]['PatientID'].nunique()
        missing_count = total_n - present_count
        stats.append({'Sequence': seq.upper(), 'Missing': missing_count})
        
    plot_df = pd.DataFrame(stats)

    # We add a small buffer (e.g., +2) so the labels don't touch the outer line
    max_val = plot_df['Missing'].max()
    radial_max = max_val + (2 if max_val > 0 else 5)
    plot_df_closed = pd.concat([plot_df, plot_df.iloc[[0]]])

    # Plotly Spider Plot
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=plot_df_closed['Missing'],
        theta=plot_df_closed['Sequence'],
        fill='toself',
        fillcolor='rgba(31, 119, 180, 0.4)', # Muted blue
        line=dict(color='rgb(31, 119, 180)', width=2),
        name='Missing MRI Scans'
    ))

    fig.update_layout(
        title=f"MRI Data Gaps (n={total_n} patients total)",
        font=dict(family="Arial, sans-serif", size=16),
        polar=dict(
            radialaxis=dict(visible=True, range=[0, radial_max], title="Missing Count")    
        ),
        showlegend=False
    )
    
    fig.show()
    #fig.write_image("mri_availability_spider.png")

# --- EXECUTION ---
if __name__ == '__main__':
    base_dir = "/Volumes/T7 Shield/Restrepo/HCC_Retrospective_radiomics/Total_segmentations"
    results_dir = '/Volumes/T7 Shield/Restrepo/HCC_Retrospective_radiomics/Results/diagnostics' 
    folders = ['SA1', 'SA2']
    sequences = ['pre', 'ap', 'pvp', 'tp', 'hbp', 't2', 'dwi_b1', 'dwi_b2', 'adc']

    mri_df, total_pats = audit_mri_availability(base_dir, folders, sequences)
    print(mri_df)

    if not mri_df.empty:
        plot_missing_mri_spider(mri_df, total_pats)
        
        # Save a list of exactly which patients are missing which scans
        missing_list = mri_df[mri_df['Available'] == 0]
        missing_list.to_csv(results_dir+"/patients_missing_sequences.csv", index=False)
        print(f"✅ Summary: Found {total_pats} total patients.")
        print(f"📂 Detailed gaps saved to 'patients_missing_sequences.csv'")