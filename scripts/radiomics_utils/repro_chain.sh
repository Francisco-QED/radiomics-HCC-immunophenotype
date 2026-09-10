#!/bin/bash
# Full-chain reproduction driver for tils_2_vs_0-1/radiomics/binwidth_perseq_nowavelet -> ...2
set -e
cd "/Users/Francisco/Documents/GitHub/radiomics-HCC-immunophenotype/scripts/radiomics_utils"
PY=/usr/bin/python3
BW_LABEL="binwidth_perseq_nowavelet2"
RESULTS_DIR="/Users/Francisco/Library/CloudStorage/OneDrive-TheMountSinaiHospital/Regeneron retrospective radiomics/Results/main_pipeline/immunophenotype/tils_2_vs_0-1/radiomics/${BW_LABEL}"
SUMMARY="${RESULTS_DIR}/summary_results"

echo "=== [1/5] partition (7 combos) ==="
$PY run_seven_combo_partition.py --cutoff 2 --binwidth-label "$BW_LABEL"

echo "=== [2/5] gridsearch + train ==="
$PY gridsearch_train_sweep.py --cutoff 2 --binwidth-label "$BW_LABEL" --top-n 7

echo "=== [3/5] build_sweep_diagnostics (feature consistency + ranking) ==="
$PY build_sweep_diagnostics.py --radiomics-dir "$RESULTS_DIR" --top-n 7

echo "=== [4/5] publish_rank_deliverables (ROC copy, SHAP, rank1 lock-set alignment) ==="
$PY publish_rank_deliverables.py --radiomics-dir "$RESULTS_DIR" --top-n 7

echo "=== [5/5] case overlay for rank1's combo ==="
RANK1_FILE=$(ls "$SUMMARY"/results_*_rank1_*.csv | head -1)
RANK1_COMBO=$(basename "$RANK1_FILE" | sed -E 's/^results_.*_rank1_(.+)\.csv$/\1/')
echo "rank1 combo detected: $RANK1_COMBO"
$PY case_overlay_qc.py --summary-dir "$SUMMARY" --combo "$RANK1_COMBO" --patients P085 --output-dir "$SUMMARY"

echo "=== DONE ==="
