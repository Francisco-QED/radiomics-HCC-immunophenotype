import os
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

def run_binning_analysis_and_plot(patient_path, sequences, test_widths=[5, 10, 15, 20]):
    # Setup Figure: 2 rows, N columns, sharex='col' locks alignment
    fig, axes = plt.subplots(2, len(sequences), figsize=(len(sequences)*2, 7), 
                             sharex='col', constrained_layout=True)
    
    files = os.listdir(patient_path)
    
    for c, seq in enumerate(sequences):
        img_file = next((f for f in files if seq.lower() in f.lower() and "seg" not in f.lower() and f.endswith('.nrrd')), None)
        seg_file = next((f for f in files if seq.lower() in f.lower() and "seg" in f.lower() and f.endswith('.nrrd')), None)
        
        if not (img_file and seg_file): continue
            
        img = sitk.ReadImage(os.path.join(patient_path, img_file))
        mask = sitk.ReadImage(os.path.join(patient_path, seg_file))
        
        stats = sitk.LabelShapeStatisticsImageFilter()
        stats.Execute(mask)
        bb = stats.GetBoundingBox(1) 
        
        roi_filter = sitk.RegionOfInterestImageFilter()
        roi_filter.SetSize([bb[3], bb[4], bb[5]])
        roi_filter.SetIndex([bb[0], bb[1], bb[2]])
        cropped_img = roi_filter.Execute(img)
        cropped_mask = roi_filter.Execute(mask)
        
        arr = sitk.GetArrayFromImage(cropped_img)
        mask_arr = sitk.GetArrayFromImage(cropped_mask)
        spacing = cropped_img.GetSpacing()
        
        # --- TOP ROW: Axial Slice ---
        ax_img = axes[0, c]
        phys_x = arr.shape[2] * spacing[0]
        phys_y = arr.shape[1] * spacing[1]
        mid_z = arr.shape[0] // 2
        
        ax_img.imshow(arr[mid_z, :, :], cmap='gray', extent=[0, phys_x, phys_y, 0], aspect='equal')
        mid_row = arr.shape[1] // 2
        ax_img.axhline(mid_row * spacing[1], color='r', linestyle='--') # Plot in mm
        ax_img.set_title(seq.upper())
        ax_img.axis('on')
        
        # --- BOTTOM ROW: Intensity Profiles ---
        ax_plot = axes[1, c]
        tumor_vals = arr[mask_arr > 0]
        tumor_min = tumor_vals.min()
        
        # 1. Add Original Curve (Normalized)
        mid_row = arr.shape[1] // 2
        raw_prof = arr[mid_z, mid_row, :].astype(float)
        # Normalize original to 0-1 for valid comparison with steps
        if raw_prof.max() > 0:
            raw_prof = raw_prof / raw_prof.max()
        x_coords = np.arange(len(raw_prof)) * spacing[0]
        ax_plot.plot(x_coords, raw_prof, label='Original', color='black', linestyle='--', linewidth=1.5)
        
        # 2. Add Binned Curves
        for bw in test_widths:
            binned = np.zeros_like(arr, dtype=np.float32)
            # Perform binning
            binned[mask_arr > 0] = np.floor((arr[mask_arr > 0] - tumor_min) / bw)
            
            prof = binned[mid_z, mid_row, :]
            # Normalize binned to 0-1 to match original scale
            if prof.max() > 0:
                prof = prof / prof.max()
            
            ax_plot.plot(x_coords, prof, label=f'{bw:.2f} BW')
            
        ax_plot.set_xlabel('Physical Distance (mm)')
        if c == 0: ax_plot.set_ylabel('Norm. Intensity/Level')
        ax_plot.grid(True, alpha=0.3)
        ax_plot.legend(fontsize=7, loc='best')
        
    plt.suptitle("Texture analysis: binning dependence", fontsize=16)
    plt.show()

# Execution
patient_dir = "/Volumes/T7 Shield/Restrepo/HCC_Retrospective_radiomics/Total_segmentations/SA1/P017"
target_seqs = ['adc', 't2', 'hbp', 'ap','pre', 'pvp', 'tp'] 
run_binning_analysis_and_plot(patient_dir, target_seqs, test_widths=[5, 10, 20])