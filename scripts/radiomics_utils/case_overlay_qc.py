"""
Per-patient tumor-segmentation QC overlay, for visual sanity-checking a rank1 combo's
lock-set predictions against the actual MRI + segmentation.

Recreates the kind of deliverable previously produced by an ad-hoc script that is no
longer present anywhere in the repo or on disk (it predated this file and was never
committed) — this is a from-scratch reimplementation, not a recovery of the original.

Design notes / things that differ deliberately from what the lost script did:
  - Image/mask are read straight from the source NRRDs used by extraction.py
    (sitk.ReadImage -> sitk.GetArrayFromImage, no flips/rotations applied), so mask-to-image
    registration here is only ever as good as the source files themselves.
  - Display convention is plain matplotlib `imshow` defaults (row 0 at top, no flips).
    A prior version of this deliverable (rank1_pre_ap_pvp_tp_hbp_case_P085_tumor.png,
    tils_2_vs_0-1/binwidth_perseq_nowavelet) displayed the same, correctly-registered
    data rotated 180 degrees relative to this convention — confirmed by an independent
    from-scratch render that the mask-to-image alignment itself was fine either way; only
    the display orientation differed. That old script is gone, so there was nothing to fix
    in the current pipeline — this file just picks (and documents) one convention going
    forward instead of silently guessing at the old one.
  - Shows every sequence in the combo by default (the old deliverable showed a 3-of-5
    subset for P085's combo with no discoverable selection rule) — pass --sequences to
    restrict this if a figure gets too wide.

Per-patient title metadata (TILs_resection, lock-set membership, model P(high TILs)) is
pulled from `rank1_<combo>_lock_set_feature_alignment.csv` in the sweep's summary_results/
folder, joined by PatientID against the sequences list in gridsearch_train_results.csv.

Usage:
    python case_overlay_qc.py \
        --summary-dir "<.../binwidth_perseq_nowavelet/summary_results>" \
        --combo pre_ap_pvp_tp_hbp \
        --patients P085 P019 \
        --output-dir <somewhere NOT in the repo>

    # or, to render every patient in that combo's lock set:
    python case_overlay_qc.py --summary-dir ... --combo pre_ap_pvp_tp_hbp --all --output-dir ...
"""
import argparse
import ast
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk

SEQUENCES_WITHOUT_N4 = {"adc", "dwi_b1", "dwi_b2"}
IMAGE_BASE_DIR = "/Volumes/T7 Shield/Restrepo/HCC_Retrospective_radiomics/Total_segmentations"
PATIENT_COHORT_SUBDIRS = ["SA1", "SA2"]  # searched in this order
BBOX_MARGIN_VOXELS = 8  # matches the old deliverable's stated "+8 voxel margin"


def find_patient_dir(pid, base_dir=IMAGE_BASE_DIR, subdirs=PATIENT_COHORT_SUBDIRS):
    for sub in subdirs:
        candidate = os.path.join(base_dir, sub, pid)
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(f"Could not find a patient directory for {pid} under {subdirs} in {base_dir}")


def load_sequence_arrays(patient_dir, pid, seq):
    img_suffix = "" if seq in SEQUENCES_WITHOUT_N4 else "_N4"
    img_path = os.path.join(patient_dir, f"{pid}_{seq}{img_suffix}.nrrd")
    mask_path = os.path.join(patient_dir, f"{pid}_{seq}.seg.nrrd")
    if not (os.path.exists(img_path) and os.path.exists(mask_path)):
        return None
    img = sitk.ReadImage(img_path)
    seg = sitk.ReadImage(mask_path)
    if img.GetSize() != seg.GetSize():
        raise ValueError(f"{pid}/{seq}: image/mask size mismatch {img.GetSize()} vs {seg.GetSize()}")
    return sitk.GetArrayFromImage(img), sitk.GetArrayFromImage(seg) > 0, img_suffix


def largest_area_slice_crop(img_arr, mask_arr, margin=BBOX_MARGIN_VOXELS):
    areas = mask_arr.sum(axis=(1, 2))
    if areas.max() == 0:
        return None
    z = int(np.argmax(areas))
    ys, xs = np.where(mask_arr[z])
    y0, y1 = max(ys.min() - margin, 0), min(ys.max() + margin + 1, img_arr.shape[1])
    x0, x1 = max(xs.min() - margin, 0), min(xs.max() + margin + 1, img_arr.shape[2])
    return img_arr[z, y0:y1, x0:x1], mask_arr[z, y0:y1, x0:x1], int(areas[z])


def load_combo_sequences(gridsearch_csv, combo):
    df = pd.read_csv(gridsearch_csv)
    row = df[df["sequences_str"] == combo]
    if row.empty:
        raise ValueError(f"Combo '{combo}' not found in {gridsearch_csv} (sequences_str column)")
    return ast.literal_eval(row.iloc[0]["sequences"])


def render_case(pid, meta_row, sequences, patient_dir, out_path, seq_suffix_note):
    panels = []
    for seq in sequences:
        loaded = load_sequence_arrays(patient_dir, pid, seq)
        if loaded is None:
            continue
        img_arr, mask_arr, img_suffix = loaded
        cropped = largest_area_slice_crop(img_arr, mask_arr)
        if cropped is None:
            continue
        crop, mcrop, area = cropped
        panels.append((seq, crop, mcrop, area, img_suffix))

    if not panels:
        print(f"[{pid}] no sequences with a usable tumor mask found — skipping")
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5))
    if len(panels) == 1:
        axes = [axes]

    for ax, (seq, crop, mcrop, area, img_suffix) in zip(axes, panels):
        ax.imshow(crop, cmap="gray")
        ax.contour(mcrop, levels=[0.5], colors="red")
        label = "N4" if img_suffix == "_N4" else "raw"
        ax.set_title(f"{seq.upper()} ({label})  |  tumor area={area} px")

    lock_note = "held-out lock set" if meta_row is not None else "dev set (no lock-set prediction)"
    if meta_row is not None:
        proba_note = f", model P(high TILs)={meta_row['model_proba_high_TILs']:.2f}"
        tils_note = f"TILs_resection={meta_row['TILs_resection']:g}"
    else:
        proba_note = ""
        tils_note = "TILs_resection=unknown"

    fig.suptitle(
        f"{pid} ({tils_note}, {lock_note}{proba_note})\n"
        f"Cropped to tumor bounding box (+{BBOX_MARGIN_VOXELS} voxel margin), "
        f"largest-area slice per sequence, segmentation in red{seq_suffix_note}",
        fontsize=11,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[{pid}] saved {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary-dir", required=True, help="e.g. .../binwidth_perseq_nowavelet/summary_results")
    ap.add_argument("--combo", required=True, help="e.g. pre_ap_pvp_tp_hbp (matches sequences_str)")
    ap.add_argument("--gridsearch-csv", default=None,
                     help="defaults to <summary-dir>/gridsearch_train_results.csv")
    ap.add_argument("--patients", nargs="*", default=None, help="specific PatientIDs to render")
    ap.add_argument("--all", action="store_true", help="render every patient in the lock-set alignment CSV")
    ap.add_argument("--sequences", nargs="*", default=None,
                     help="restrict to a subset of the combo's sequences (default: all)")
    ap.add_argument("--output-dir", required=True,
                     help="MUST be outside the git repo — e.g. a folder under the OneDrive Results tree")
    args = ap.parse_args()

    if os.path.abspath(args.output_dir).startswith(os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))):
        raise SystemExit("Refusing to write output inside the repo. Pass an --output-dir outside it.")

    os.makedirs(args.output_dir, exist_ok=True)

    gridsearch_csv = args.gridsearch_csv or os.path.join(args.summary_dir, "gridsearch_train_results.csv")
    sequences = load_combo_sequences(gridsearch_csv, args.combo)
    if args.sequences:
        sequences = [s for s in sequences if s in args.sequences]

    alignment_csv = os.path.join(args.summary_dir, f"rank1_{args.combo}_lock_set_feature_alignment.csv")
    meta_df = pd.read_csv(alignment_csv).set_index("PatientID") if os.path.exists(alignment_csv) else None
    if meta_df is None:
        print(f"Warning: no lock_set_feature_alignment.csv found at {alignment_csv} — titles will lack TILs/probability info")

    if args.all:
        if meta_df is None:
            raise SystemExit("--all requires the lock_set_feature_alignment.csv to enumerate patients")
        patients = list(meta_df.index)
    elif args.patients:
        patients = args.patients
    else:
        raise SystemExit("Pass --patients P### [P### ...] or --all")

    seq_suffix_note = f" (sequences: {', '.join(sequences)})" if args.sequences else ""

    for pid in patients:
        meta_row = meta_df.loc[pid] if (meta_df is not None and pid in meta_df.index) else None
        try:
            patient_dir = find_patient_dir(pid)
        except FileNotFoundError as e:
            print(f"[{pid}] {e} — skipping")
            continue
        out_path = os.path.join(args.output_dir, f"rank1_{args.combo}_case_{pid}_tumor.png")
        render_case(pid, meta_row, sequences, patient_dir, out_path, seq_suffix_note)


if __name__ == "__main__":
    main()
