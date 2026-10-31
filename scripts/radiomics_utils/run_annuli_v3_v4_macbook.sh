#!/usr/bin/env bash
set -euo pipefail

# Usage: bash run_annuli_v3_v4_macbook.sh [masks|extract|models|all] [v3|v4|both]
MODE="${1:-all}"
WHICH="${2:-both}"
case "$MODE" in masks|extract|models|all) ;; *) echo "invalid mode: $MODE" >&2; exit 2 ;; esac
case "$WHICH" in v3|v4|both) ;; *) echo "invalid variant: $WHICH" >&2; exit 2 ;; esac

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
FINAL_DIR="$REPO_ROOT/scripts/final_pipeline"
PROJECT_ROOT="${RADIOMICS_PROJECT_ROOT:-$HOME/Library/CloudStorage/OneDrive-TheMountSinaiHospital/Regeneron retrospective radiomics}"
SEG_ROOT="${RADIOMICS_SEG_ROOT:-/Volumes/T7 Shield/Restrepo/HCC_Retrospective_radiomics/Total_segmentations}"
CLEAN_ROOT="${PERI_CLEAN_ROOT:-/Volumes/LaCie2/clean segs 10-23-26}"
MASK_BASE="${ANNULUS_MASK_BASE:-/Volumes/LaCie2}"
SHEETS_BASE="${ANNULUS_SHEETS_BASE:-$PROJECT_ROOT/Results/Radiomics_sheets}"
RESULTS_BASE="${ANNULUS_RESULTS_BASE:-$PROJECT_ROOT/Results/main_pipeline_gpt}"
SLICER_BIN="${SLICER_BIN:-/Applications/Slicer.app/Contents/MacOS/Slicer}"
EXTRACT_PY="${EXTRACT_PY:-python3}"
TIL_PY="${TIL_PY:-python3}"
SURVIVAL_PY="${SURVIVAL_PY:-$REPO_ROOT/.venv-survival/bin/python}"
EXTRACT_JOBS="${EXTRACT_JOBS:-4}"
TIL_JOBS="${TIL_JOBS:-4}"
SURVIVAL_JOBS="${SURVIVAL_JOBS:-2}"
SETS=(tp pvp dwi_b2 hbp tp+pvp tp+dwi_b2 tp+hbp pvp+dwi_b2 pvp+hbp dwi_b2+hbp tp+pvp+dwi_b2 tp+pvp+hbp tp+dwi_b2+hbp pvp+dwi_b2+hbp)
FAMILIES=(ridge lasso elastic_net)

variants=()
[[ "$WHICH" == v3 || "$WHICH" == both ]] && variants+=(v3)
[[ "$WHICH" == v4 || "$WHICH" == both ]] && variants+=(v4)

require_path() { [[ -e "$1" ]] || { echo "missing required path: $1" >&2; exit 1; }; }
if [[ "$MODE" == masks || "$MODE" == all ]]; then
  require_path "$SLICER_BIN"; require_path "$SEG_ROOT"; require_path "$CLEAN_ROOT"
fi
require_path "$PROJECT_ROOT"; require_path "$REPO_ROOT"

geometry() {
  if [[ "$1" == v3 ]]; then echo "0 7 7in_0out"; else echo "7 0 0in_7out"; fi
}

make_masks() {
  local variant="$1" out_mm in_mm tag mask_root
  read -r out_mm in_mm tag <<<"$(geometry "$variant")"
  mask_root="$MASK_BASE/peritumor_rois${variant}_${tag}"
  if [[ -e "$mask_root" ]]; then echo "refusing to write into existing mask folder: $mask_root" >&2; exit 1; fi
  mkdir "$mask_root"
  echo "[$variant] generating masks in $mask_root"
  PERI_OUT="$mask_root" PERI_R_OUT="$out_mm" PERI_R_IN="$in_mm" PERI_COHORTS="SA1,SA2_post" PERI_SEG_ROOT="$SEG_ROOT" PERI_CLEAN_ROOT="$CLEAN_ROOT" \
    "$SLICER_BIN" --no-main-window --python-script "$SCRIPT_DIR/peritumor_annulus_slicer.py" --exit-after-startup
  for reader in Reader1 Reader2; do
    PERI_OUT="$mask_root" PERI_R_OUT="$out_mm" PERI_R_IN="$in_mm" PERI_READER="$reader" PERI_SEG_ROOT="$SEG_ROOT" PERI_CLEAN_ROOT="$CLEAN_ROOT" \
      "$SLICER_BIN" --no-main-window --python-script "$SCRIPT_DIR/peritumor_annulus_slicer.py" --exit-after-startup
  done
  "$EXTRACT_PY" "$SCRIPT_DIR/audit_annulus_geometry.py" --mask-root "$mask_root" --variant "$variant"
}

extract_sheets() {
  local variant="$1" out_mm in_mm tag mask_root sheets_root combined
  read -r out_mm in_mm tag <<<"$(geometry "$variant")"
  mask_root="$MASK_BASE/peritumor_rois${variant}_${tag}"
  sheets_root="$SHEETS_BASE/Annuli_${variant}"
  combined="$sheets_root/SA1_2_post/R-SA1_2-post_I-sitkBSpline_S-1mm_W-perseq.csv"
  require_path "$mask_root/geometry_audit_summary.json"
  if [[ -e "$sheets_root" ]]; then echo "refusing to write into existing sheet folder: $sheets_root" >&2; exit 1; fi
  echo "[$variant] extracting radiomics to $sheets_root"
  for target in SA1 SA2-post Reader1 Reader2; do
    RADIOMICS_SEG_ROOT="$SEG_ROOT" "$EXTRACT_PY" "$SCRIPT_DIR/annuli_extraction.py" --target "$target" --out-root "$sheets_root" \
      --jobs "$EXTRACT_JOBS" --out-mm "$out_mm" --in-mm "$in_mm" --mask-source supplied --mask-root "$mask_root" --binwidth perseq
  done
  "$EXTRACT_PY" "$SCRIPT_DIR/combine_annuli_sheets.py" --sheets-root "$sheets_root" --binwidth perseq
}

run_models() {
  local variant="$1" out_mm in_mm tag sheets_root out_root
  read -r out_mm in_mm tag <<<"$(geometry "$variant")"
  sheets_root="$SHEETS_BASE/Annuli_${variant}"
  require_path "$sheets_root/SA1_2_post/R-SA1_2-post_I-sitkBSpline_S-1mm_W-perseq.csv"
  require_path "$SURVIVAL_PY"
  out_root="$RESULTS_BASE/annuli_${variant}_${tag}_post_$(date +%Y%m%d_%H%M%S)"
  if [[ -e "$out_root" ]]; then echo "refusing to write into existing results folder: $out_root" >&2; exit 1; fi
  mkdir "$out_root"
  echo "[$variant] development-only model screen in $out_root"
  RADIOMICS_PROJECT_ROOT="$PROJECT_ROOT" RADIOMICS_SHEETS_DIR="$sheets_root" "$TIL_PY" "$FINAL_DIR/repeated_nested_families.py" \
    --sheet post --sets "${SETS[@]}" --families "${FAMILIES[@]}" --repeats 1 --jobs "$TIL_JOBS" \
    --output-parent "$out_root/tils_ge2_screen" --label "tilsGE2_annuli_${variant}_screen" --endpoint tils_ge2 --strata outcome --icc fold --gpt-dir "$FINAL_DIR/vendored_functions"
  RADIOMICS_PROJECT_ROOT="$PROJECT_ROOT" RADIOMICS_SHEETS_DIR="$sheets_root" "$SURVIVAL_PY" "$FINAL_DIR/repeated_survival_families.py" \
    --gpt-dir "$FINAL_DIR/vendored_functions" --sheet post --sets "${SETS[@]}" --families "${FAMILIES[@]}" --clinical none \
    --repeats 1 --jobs "$SURVIVAL_JOBS" --output-parent "$out_root/recurrence_screen" --label "recurrence_annuli_${variant}_screen"
  if [[ "$variant" == v4 ]]; then
    echo "[v4] running exterior-annulus no-shape negative-control screens"
    RADIOMICS_PROJECT_ROOT="$PROJECT_ROOT" RADIOMICS_SHEETS_DIR="$sheets_root" "$TIL_PY" "$FINAL_DIR/repeated_nested_families.py" \
      --sheet post --sets "${SETS[@]}" --families "${FAMILIES[@]}" --repeats 1 --jobs "$TIL_JOBS" --exclude-shape \
      --output-parent "$out_root/tils_ge2_no_shape_screen" --label "tilsGE2_annuli_v4_no_shape" --endpoint tils_ge2 --strata outcome --icc fold --gpt-dir "$FINAL_DIR/vendored_functions"
    RADIOMICS_PROJECT_ROOT="$PROJECT_ROOT" RADIOMICS_SHEETS_DIR="$sheets_root" "$SURVIVAL_PY" "$FINAL_DIR/repeated_survival_families.py" \
      --gpt-dir "$FINAL_DIR/vendored_functions" --sheet post --sets "${SETS[@]}" --families "${FAMILIES[@]}" --clinical none --exclude-shape \
      --repeats 1 --jobs "$SURVIVAL_JOBS" --output-parent "$out_root/recurrence_no_shape_screen" --label "recurrence_annuli_v4_no_shape"
  fi
}

for variant in "${variants[@]}"; do
  if [[ "$MODE" == masks || "$MODE" == all ]]; then make_masks "$variant"; fi
  if [[ "$MODE" == extract || "$MODE" == all ]]; then extract_sheets "$variant"; fi
  if [[ "$MODE" == models || "$MODE" == all ]]; then run_models "$variant"; fi
done
echo "ALL REQUESTED STAGES COMPLETE"
