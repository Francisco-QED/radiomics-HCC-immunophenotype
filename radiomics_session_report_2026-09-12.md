# Radiomics Pipeline Session Report - 2026-09-12

## Scope

This session focused on exploratory model discovery for the HCC immunophenotype radiomics pipeline, especially improving OOF AUC while keeping feature selection fold-internal and avoiding leakage. We kept analysis outputs outside the repository.

## Main Work Completed

1. Reviewed prior detailed-sweep results and separated candidates by TIL cutoff.
2. Ranked candidate models by both OOF AUC and feature recurrence/prevalence.
3. Created cutoff-separated top-5 summaries from the existing detailed sweep.
4. Ran a new broad binary signal screen using 4-6 MRI sequence combinations.
5. Confirmed the large sweep completed successfully and saved all outputs.

## Detailed-Sweep Summary

Existing detailed-sweep outputs were organized into a cutoff-separated top-5 report.

Location:

`/Users/Francisco/Library/CloudStorage/OneDrive-TheMountSinaiHospital/Regeneron retrospective radiomics/Results/main_pipeline/immunophenotype/signal_screen_v2/detailed_sweep_k5_25_nowavelet_nologsigma/summary_results/top5_by_cutoff`

Files created:

- `top5_by_cutoff_model_summary.csv`
- `top5_by_cutoff_feature_recurrence.csv`
- `top5_by_cutoff_report.md`

Key interpretation:

- Cutoff 3 best detailed candidate: `pre_tp_dwi_b2`, SVM linear, K=15, OOF AUC 0.6678.
- Cutoff 2 best detailed candidate: `pvp_tp`, elastic net, K=25, OOF AUC 0.6520.
- `pvp_tp` had stronger feature recurrence than the top cutoff-3 model.
- Feature recurrence was descriptive across OOF folds, not formal Meinshausen-Buhlmann stability selection.

## Large 4-6 Sequence Sweep

A new broad signal screen was run for all 4-, 5-, and 6-sequence combinations using:

- Cutoffs: 3 and 2
- Sequence pool: `pre`, `ap`, `pvp`, `tp`, `hbp`, `t2`, `dwi_b1`, `dwi_b2`, `adc`
- Models: ridge, elastic net, linear SVM
- Fold-internal AUC filter
- K values: 25, 50, 100
- Excluded feature families: `wavelet`, `log-sigma`
- No lock-set evaluation

Output location:

`/Users/Francisco/Library/CloudStorage/OneDrive-TheMountSinaiHospital/Regeneron retrospective radiomics/Results/main_pipeline/immunophenotype/signal_screen_v2/large_combos_4to6_nowavelet_nologsigma_aucfilter`

Summary file:

`summary_results/signal_screen_summary.csv`

The run completed:

- Total runs: 672
- Successful: 672
- Failed: 0

Each combination folder contains:

- `y_dev.csv`
- `y_lock.csv`
- `screen_feature_pool.csv`
- `nested_oof_signal_screen.csv`
- `nested_oof_signal_screen_folds.csv`
- `nested_oof_signal_screen_predictions.csv`

## Best Large-Sweep Results

Top overall results:

| Rank | Cutoff | Combo | Model | K | OOF AUC |
|---:|---:|---|---|---:|---:|
| 1 | 3 | `pre_tp_dwi_b1_dwi_b2` | SVM linear | 25 | 0.6653 |
| 2 | 2 | `ap_pvp_tp_hbp_dwi_b2_adc` | ridge | 25 | 0.6481 |
| 3 | 3 | `pre_ap_tp_dwi_b1_dwi_b2` | SVM linear | 25 | 0.6462 |
| 4 | 2 | `ap_tp_dwi_b2_adc` | elastic net | 25 | 0.6347 |
| 5 | 3 | `pvp_hbp_dwi_b1_adc` | SVM linear | 100 | 0.6334 |
| 6 | 3 | `ap_pvp_hbp_dwi_b1` | SVM linear | 100 | 0.6326 |
| 7 | 3 | `pre_ap_tp_dwi_b1` | SVM linear | 25 | 0.6297 |
| 8 | 2 | `ap_pvp_tp_hbp` | ridge | 25 | 0.6296 |

Top cutoff-3 large-sweep model:

- `pre_tp_dwi_b1_dwi_b2`
- SVM linear
- K=25
- OOF AUC 0.6653
- Bootstrap CI 0.5567-0.7768

Top cutoff-2 large-sweep model:

- `ap_pvp_tp_hbp_dwi_b2_adc`
- Ridge
- K=25
- OOF AUC 0.6481
- Bootstrap CI 0.5026-0.7822

## Interpretation

The 4-6 sequence sweep found models in the same performance range as the best earlier smaller-combo models, but did not dramatically exceed them.

The broader combinations helped cutoff 2 more than cutoff 3. The best cutoff-2 large-combo model reached OOF AUC 0.6481, close to the best cutoff-3 models.

The best cutoff-3 large-combo model, `pre_tp_dwi_b1_dwi_b2`, was essentially tied with the earlier smaller-combo detailed winner, `pre_tp_dwi_b2`.

## Suggested Next Steps

1. Take the top 5 models per cutoff from the 4-6 sequence sweep.
2. Run the same detailed recurrence/feature-prevalence analysis on those candidates.
3. Compare small-combo and large-combo candidates side by side.
4. Choose a short list for final lock-set evaluation.
5. Keep lock-set AUC separate from exploratory OOF model selection.

