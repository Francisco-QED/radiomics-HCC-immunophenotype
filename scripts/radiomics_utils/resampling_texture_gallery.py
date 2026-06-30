import os
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

def run_resampling_analysis_and_plot(patient_path, sequences, test_spacings=[1.0, 1.5, 2.0]):
    # sharex='col' forces the X-axis limits to be identical vertically
    fig, axes = plt.subplots(2, len(sequences), figsize=(len(sequences)*2, 7), 
                             sharex='col', constrained_layout=True)
    
    files = [x for x in os.listdir(patient_path) if x[0]!='.'] # Ignore hidden files
    
    for c, seq in enumerate(sequences):
        img_file = next((f for f in files if seq.lower() in f.lower() and "seg" not in f.lower() and f.endswith('.nrrd')), None)
        seg_file = next((f for f in files if seq.lower() in f.lower() and "seg" in f.lower() and f.endswith('.nrrd')), None)
        
        if not (img_file and seg_file): continue
            
        img = sitk.ReadImage(os.path.join(patient_path, img_file))
        mask = sitk.ReadImage(os.path.join(patient_path, seg_file))
        
        # Crop
        stats = sitk.LabelShapeStatisticsImageFilter()
        stats.Execute(mask)
        bb = stats.GetBoundingBox(1) 
        roi_filter = sitk.RegionOfInterestImageFilter()
        roi_filter.SetSize([bb[3], bb[4], bb[5]])
        roi_filter.SetIndex([bb[0], bb[1], bb[2]])
        cropped_img = roi_filter.Execute(img)
        
        # --- TOP ROW: Axial Slice ---
        ax_img = axes[0, c]
        img_arr = sitk.GetArrayFromImage(cropped_img)
        spacing = cropped_img.GetSpacing() # (x, y, z)
        
        # Calculate physical extent in mm: [0, width_mm, height_mm, 0]
        # img_arr shape is [z, y, x], so:
        phys_x = img_arr.shape[2] * spacing[0]
        phys_y = img_arr.shape[1] * spacing[1]
        
        mid_z = img_arr.shape[0] // 2
        
        # Use 'extent' to map pixels to physical mm space
        ax_img.imshow(img_arr[mid_z, :, :], cmap='gray', extent=[0, phys_x, phys_y, 0])
        ax_img.set_aspect('equal', adjustable='box')

        mid_row = img_arr.shape[1] // 2
        ax_img.axhline(mid_row * spacing[1], color='r', linestyle='--') # Plot in mm
        ax_img.set_title(seq.upper())
        ax_img.axis('on') # Keep axes on so you can see the matching limits
        
        ax_img.set_xlim([0, phys_x]) 
        ax_img.set_ylim([phys_y, 0]) # Note: Inverting Y for image

        # --- BOTTOM ROW: Intensity Profiles ---
        ax_plot = axes[1, c]
        spacings_to_plot = [cropped_img.GetSpacing()[0]] + test_spacings
        
        for sp in spacings_to_plot:
            resampler = sitk.ResampleImageFilter()
            resampler.SetOutputSpacing([sp, sp, sp])
            resampler.SetSize([int(round(osz * osp / sp)) for osz, osp in zip(cropped_img.GetSize(), cropped_img.GetSpacing())])
            resampler.SetInterpolator(sitk.sitkBSpline)
            resampler.SetOutputOrigin(cropped_img.GetOrigin())
            resampler.SetOutputDirection(cropped_img.GetDirection())
            resampled = resampled_img = resampler.Execute(cropped_img)
            
            arr = sitk.GetArrayFromImage(resampled_img)
            c_mid_z = arr.shape[0] // 2
            c_mid_row = arr.shape[1] // 2
            
            prof = arr[c_mid_z, c_mid_row, :]
            x_coords = np.arange(len(prof)) * sp
            
            if prof.max() > 0:
                ax_plot.plot(x_coords, prof/prof.max(), label=f'{sp:.2f}mm')
        
        ax_plot.set_xlabel('Physical Distance (mm)')
        ax_plot.set_xlim([0, phys_x])
        ax_plot.set_ylim([0, 1])
        if c == 0: ax_plot.set_ylabel('Norm. Intensity')
        ax_plot.grid(True, alpha=0.3)
        ax_plot.legend(fontsize=8)
        
    plt.suptitle("Texture analysis: resampling dependence", fontsize=16)
    plt.show()

# Execution
patient_dir = "/Volumes/T7 Shield/Restrepo/HCC_Retrospective_radiomics/Total_segmentations/SA1/P005"
target_seqs = ['adc', 't2', 'hbp', 'ap','pre', 'pvp', 'tp'] 
run_resampling_analysis_and_plot(patient_dir, target_seqs)