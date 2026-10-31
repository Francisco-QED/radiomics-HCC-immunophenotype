# Annuli v3/v4 MacBook run

This package creates two new annulus experiments without modifying any existing project input or output:

- **Annuli v3:** 7 mm inward, 0 mm outward.
- **Annuli v4:** 0 mm inward, 7 mm outward, clipped to the liver mask.

SA1 uses SA1 imaging and SA2 uses SA2-post imaging. The model screen uses the established fixed 111-patient development subset from the 141-patient cohort. It does not evaluate the lock set.

## Write protection

The launcher reads the existing tumor segmentations, images, liver masks, database, and split files. It writes only to new folders and aborts if an intended mask, sheet, or result folder already exists.

Default new folders:

- `/Volumes/LaCie2/peritumor_roisv3_7in_0out`
- `/Volumes/LaCie2/peritumor_roisv4_0in_7out`
- `Results/Radiomics_sheets/Annuli_v3`
- `Results/Radiomics_sheets/Annuli_v4`
- timestamped `Results/main_pipeline_gpt/annuli_v3_*` and `annuli_v4_*` folders

## Geometry safeguards

Mask construction is slice-wise and uses physical in-plane millimeters. A width of zero is strict: v3 has no exterior voxels and v4 has no tumor voxels. V4 exterior voxels are clipped to the liver mask.

Each mask run produces `geometry_audit.csv` and `geometry_audit_summary.json`. V3 records the fraction of the tumor occupied by the inner annulus and flags cases where the 7-mm rim becomes at least 95% or all of the tumor. V4 fails automatically if any tumor voxel enters the exterior mask.

The standard v4 screen retains shape features for direct comparison with earlier annulus analyses. A second v4 screen excludes all `original_shape` features because the exterior ring contour inherits tumor size and shape even when its voxel overlap with the tumor is exactly zero.

## Recommended two-machine workflow

The masks and radiomics extraction require the T7 and LaCie2 source data. Run those two stages on the iMac, where the volumes are already mounted:

```bash
bash scripts/radiomics_utils/run_annuli_v3_v4_macbook.sh masks both
bash scripts/radiomics_utils/run_annuli_v3_v4_macbook.sh extract both
```

The extraction stage writes the new `Annuli_v3` and `Annuli_v4` CSV folders to OneDrive. After those folders finish syncing, run only the model stage on the MacBook:

```bash
bash scripts/radiomics_utils/run_annuli_v3_v4_macbook.sh models both
```

The model stage needs no access to T7, LaCie2, or the masks. It reads the synchronized CSV sheets, project split/database files, and local GitHub checkout, and writes new timestamped result folders to OneDrive.

## Prerequisites for running mask generation or extraction

Connect the T7 Shield and LaCie2 volumes. Ensure these paths exist, or override them with environment variables:

```bash
export RADIOMICS_SEG_ROOT="/Volumes/T7 Shield/Restrepo/HCC_Retrospective_radiomics/Total_segmentations"
export PERI_CLEAN_ROOT="/Volumes/LaCie2/clean segs 10-23-26"
export RADIOMICS_PROJECT_ROOT="$HOME/Library/CloudStorage/OneDrive-TheMountSinaiHospital/Regeneron retrospective radiomics"
export SLICER_BIN="/Applications/Slicer.app/Contents/MacOS/Slicer"
```

Point the launcher to Python environments that already contain the relevant packages:

```bash
export EXTRACT_PY="/absolute/path/to/python-with-pyradiomics"
export TIL_PY="/absolute/path/to/python-with-scikit-learn"
export SURVIVAL_PY="/absolute/path/to/python-with-scikit-survival"
```

`EXTRACT_PY` needs pandas, PyYAML, SimpleITK, SciPy, and PyRadiomics. `TIL_PY` needs scikit-learn >=1.8 plus the standard scientific Python packages. `SURVIVAL_PY` additionally needs scikit-survival.

## Running every stage on one machine

This option requires T7 and LaCie2 to be mounted on that same machine. From the repository root on branch `v2-stability-pipeline`:

```bash
caffeinate -i bash scripts/radiomics_utils/run_annuli_v3_v4_macbook.sh all both \
  2>&1 | tee "$HOME/annuli_v3_v4_$(date +%Y%m%d_%H%M%S).log"
```

Stages can be run separately:

```bash
bash scripts/radiomics_utils/run_annuli_v3_v4_macbook.sh masks both
bash scripts/radiomics_utils/run_annuli_v3_v4_macbook.sh extract both
bash scripts/radiomics_utils/run_annuli_v3_v4_macbook.sh models both
```

The initial model pass is a one-repeat screen of 14 prespecified sequence sets across ridge, lasso, and elastic-net for TILs >=2 and recurrence. After comparing v3 and v4 with v2, a small prespecified candidate set can receive repeated-CV confirmation. Lock evaluation should occur only after fixing that candidate and is not part of this launcher.
