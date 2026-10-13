import os
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

def run_interpolation_analysis_and_plot(patient_path, sequences, interpolators, target_spacing=1.5):
    # Setup Figure
    fig, axes = plt.subplots(2, len(sequences), figsize=(len(sequences)*2.5, 7), 
                             sharex='col', constrained_layout=True)
    
    files = [x for x in os.listdir(patient_path) if x[0]!='.']
    
    for c, seq in enumerate(sequences):
        img_file = next((f for f in files if seq.lower() in f.lower() and "seg" not in f.lower() and f.endswith('.nrrd')), None)
        seg_file = next((f for f in files if seq.lower() in f.lower() and "seg" in f.lower() and f.endswith('.nrrd')), None)
        
        if not (img_file and seg_file): continue
            
        img = sitk.ReadImage(os.path.join(patient_path, img_file))
        mask = sitk.ReadImage(os.path.join(patient_path, seg_file))
        
        # Crop to ROI
        stats = sitk.LabelShapeStatisticsImageFilter()
        stats.Execute(mask)
        bb = stats.GetBoundingBox(1) 
        roi_filter = sitk.RegionOfInterestImageFilter()
        roi_filter.SetSize([bb[3], bb[4], bb[5]])
        roi_filter.SetIndex([bb[0], bb[1], bb[2]])
        cropped_img = roi_filter.Execute(img)
        
        # --- TOP ROW: Axial Slice (Reference) ---
        ax_img = axes[0, c]
        img_arr = sitk.GetArrayFromImage(cropped_img)
        spacing = cropped_img.GetSpacing()
        phys_x = img_arr.shape[2] * spacing[0]
        phys_y = img_arr.shape[1] * spacing[1]
        
        mid_z = img_arr.shape[0] // 2
        ax_img.imshow(img_arr[mid_z, :, :], cmap='gray', extent=[0, phys_x, phys_y, 0])
        ax_img.set_aspect('equal', adjustable='box')
        mid_row = img_arr.shape[1] // 2
        ax_img.axhline(mid_row * spacing[1], color='r', linestyle='--') # Plot in mm
        ax_img.set_title(seq.upper())
        ax_img.set_xlim([0, phys_x])
        ax_img.set_ylim([phys_y, 0])
        ax_img.axis('on')

        # --- BOTTOM ROW: Intensity Profiles ---
        ax_plot = axes[1, c]
        
        # 1. Plot Reference (Original)
        mid_row = img_arr.shape[1] // 2
        ref_prof = img_arr[mid_z, mid_row, :]
        x_ref = np.arange(len(ref_prof)) * spacing[0]
        ax_plot.plot(x_ref, ref_prof/ref_prof.max(), label='Original', color='black', linestyle='--')
        
        # 2. Plot Interpolators at fixed spacing (1.5mm)
        for name, interpolator in interpolators.items():
            resampler = sitk.ResampleImageFilter()
            resampler.SetOutputSpacing([target_spacing, target_spacing, target_spacing])
            resampler.SetSize([int(round(osz * osp / target_spacing)) for osz, osp in zip(cropped_img.GetSize(), cropped_img.GetSpacing())])
            resampler.SetInterpolator(interpolator)
            resampler.SetOutputOrigin(cropped_img.GetOrigin())
            resampler.SetOutputDirection(cropped_img.GetDirection())
            
            resampled = resampler.Execute(cropped_img)
            arr = sitk.GetArrayFromImage(resampled)
            
            # Dynamic re-indexing for the resampled size
            c_mid_z = arr.shape[0] // 2
            c_mid_row = arr.shape[1] // 2
            
            prof = arr[c_mid_z, c_mid_row, :]
            # Create X axis based on target_spacing
            x_coords = np.arange(len(prof)) * target_spacing
            
            if prof.max() > 0:
                ax_plot.plot(x_coords, prof/prof.max(), label=name, alpha=0.8)
        
        ax_plot.set_xlabel('Physical Distance (mm)')
        if c == 0: ax_plot.set_ylabel('Norm. Intensity')
        ax_plot.set_xlim([0, phys_x])
        ax_plot.set_ylim([0.5, 1])
        ax_plot.grid(True, alpha=0.3)
        if c==0: ax_plot.legend(fontsize=7)
        
    plt.suptitle(f"TextureAnalysis: interpolation dependence (Fixed Spacing: {target_spacing}mm)", fontsize=16)
    plt.show()

# Execution
patient_dir = "/Volumes/T7 Shield/Restrepo/HCC_Retrospective_radiomics/Total_segmentations/SA1/P015"
target_seqs = ['adc', 't2', 'hbp', 'ap','pre', 'pvp', 'tp'] 
# Mapping of names to SITK interpolator constants
interpolators = {
        #"Nearest": sitk.sitkNearestNeighbor,
        "Linear": sitk.sitkLinear,
        "BSpline": sitk.sitkBSpline,
        #"Gaussian": sitk.sitkGaussian
    }
run_interpolation_analysis_and_plot(patient_dir, target_seqs, interpolators=interpolators, target_spacing=1.5)