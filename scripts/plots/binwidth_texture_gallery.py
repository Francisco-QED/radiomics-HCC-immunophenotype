import os
import yaml
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

_YAML_PATH = os.path.join(os.path.dirname(__file__), '..', 'radiomics_params.yaml')


def load_per_sequence_bin_widths(yaml_path=_YAML_PATH):
    """Single source of truth for the SNR-derived per-sequence bin widths -- reads directly from
    radiomics_params.yaml (the same values actually used by extraction.py's 'perseq' binWidth
    setting) instead of a hardcoded copy that could silently drift out of sync."""
    with open(yaml_path) as f:
        config = yaml.safe_load(f)
    return config['execution']['per_sequence_bin_widths']


def run_binning_analysis_and_plot(patient_path, sequences, fixed_bw=5, per_seq_widths=None):
    """For each sequence: top row shows the axial mid-slice with the profile line marked; bottom
    row compares the original (unbinned) intensity profile against two binned versions -- the
    fixed binWidth (5, the project's actual non-per-sequence baseline) and that sequence's own
    SNR-derived per-sequence bin width (per_seq_widths, from radiomics_params.yaml) -- so the two
    binning strategies can be compared directly on the same physical line through the tumor.
    """
    if per_seq_widths is None:
        per_seq_widths = load_per_sequence_bin_widths()

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

        # 2. Fixed binWidth vs. this sequence's SNR-derived per-sequence binWidth
        per_seq_bw = per_seq_widths.get(seq)
        widths_to_plot = [('Fixed', fixed_bw, 'tab:blue')]
        if per_seq_bw is not None:
            widths_to_plot.append(('Per-seq (SNR)', per_seq_bw, 'tab:orange'))
        else:
            print(f"  NOTE: no per-sequence bin width found for '{seq}' in radiomics_params.yaml -- "
                  "showing fixed binWidth only for this sequence.")

        for label, bw, color in widths_to_plot:
            binned = np.zeros_like(arr, dtype=np.float32)
            # Perform binning
            binned[mask_arr > 0] = np.floor((arr[mask_arr > 0] - tumor_min) / bw)

            prof = binned[mid_z, mid_row, :]
            # Normalize binned to 0-1 to match original scale
            if prof.max() > 0:
                prof = prof / prof.max()

            ax_plot.plot(x_coords, prof, label=f'{label} (BW={bw})', color=color)

        ax_plot.set_xlabel('Physical Distance (mm)')
        if c == 0: ax_plot.set_ylabel('Norm. Intensity/Level')
        ax_plot.grid(True, alpha=0.3)
        ax_plot.legend(fontsize=7, loc='best')

    plt.suptitle(f"Texture analysis: fixed (BW={fixed_bw}) vs. per-sequence SNR-derived bin width", fontsize=14)
    plt.show()

# Execution
patient_dir = "/Volumes/T7 Shield/Restrepo/HCC_Retrospective_radiomics/Total_segmentations/SA1/P017"
target_seqs = ['adc', 't2', 'hbp', 'ap','pre', 'pvp', 'tp']
run_binning_analysis_and_plot(patient_dir, target_seqs, fixed_bw=5)
