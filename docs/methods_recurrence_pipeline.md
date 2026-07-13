# Methods: HCC Recurrence Survival Pipeline

Covers `scripts/radiomics_utils/recurrence_pipeline.py` (stages: `partition` → `gridsearch` →
`train`), orchestrated across sequence combinations by `recurrence_sweep.py` /
`recurrence_gridsearch_train_sweep.py`. Methods only — results are reported separately once the
current sweep completes. Feature extraction (Section 1) is shared with the classification
pipeline (`docs/methods_classification_pipeline.md`) and repeated here for completeness.

## 1. Feature extraction

Identical extraction as the classification pipeline (same PyRadiomics run, same source CSVs):
9 sequences (`pre`, `ap`, `pvp`, `tp`, `hbp`, `t2`, `dwi_b1`, `dwi_b2`, `adc`), isotropic 1mm
resampling, per-sequence noise-calibrated bin widths, N4 correction on all sequences except
`adc`/`dwi_b1`/`dwi_b2`, Original + LoG (σ=2/3/4/5mm) + Wavelet (8 subbands) filters, 6 feature
classes (shape/firstorder/GLCM/GLRLM/GLSZM/GLDM) — ~11,007 raw candidate features per patient.

**Wavelet exclusion**: same `filter_feature_families()` mechanism and default
(`--exclude-feature-families=['wavelet']`) as the classification pipeline. Empirically verified on
the flagship recurrence combo (`ap_tp_dwi_b1_adc`): dev-only nested-CV OOF C-index 0.505→0.532,
consensus feature count 2→2 (one feature carried over unchanged) — no performance cost.

## 2. Sequence combination design

Same 7 pre-specified, LI-RADS-motivated combinations as the classification pipeline (no survival-
specific changes to the combo set) — see the table in `docs/methods_classification_pipeline.md`
Section 2.

## 3. Dev/lock partitioning

Stratified split (seed=`PARTITION_SEED`=997, `test_size`=30) computed once and shared across all
sequence combinations, with the same ICC-patient forced-dev-inclusion rule as the classification
pipeline. **Independent of the classification pipeline's own split** — different patient pool
(143 recurrence-evaluable patients vs. ~135–140 for TILs) and different stratification variable
(event status here vs. TILs cutoff there), so dev/lock membership is not guaranteed to match
between the two pipelines for the same patient.

**Outcome**: time-to-recurrence as a right-censored survival endpoint (`event`, `time_days`),
handled natively by Cox/survival-forest methods rather than dichotomized — no cutoff dimension to
sweep (unlike the classification pipeline's two TILs cutoffs).

## 4. Feature filtering pipeline

Same overall structure as the classification pipeline — everything past sequence/family selection
happens inside each of 5 outer cross-validation folds, fit on that fold's training partition only:

1. **Global cleanup** (dev-set-wide): identical CV-based filter to the classification pipeline
   (`--cv-threshold`, default 0.1).
2. **Interobserver stability (ICC) filter**: identical ICC(3,1) mechanism and threshold
   (`--icc-threshold`, default 0.85).
3. **Per-fold univariate screen — Cox, not AUC**: univariate `CoxPHFitter` (penalizer=0, `lifelines`)
   fit per feature against time/event, yielding log(HR), HR, Wald p-value, and univariate
   C-index; fits that fail numerically or return implausible HR (>10⁴ or <10⁻⁴) are dropped.
   Effect-size screen keeps `|log(HR)| > log(--hr-cutoff)` (default HR cutoff 1.5, symmetric —
   catches both protective and harmful directions).
4. **FDR within the screened pool**: Benjamini-Hochberg applied only to HR-screened survivors,
   threshold `--q-cutoff` (default q<0.20) — same "independent filtering" rationale as the
   classification pipeline.
5. **Spearman decorrelation**: identical mechanism (|ρ|>0.85, keep the lower-p-value member of
   each correlated pair).
6. **LASSO-Cox**: `CoxnetSurvivalAnalysis` (`l1_ratio=1.0`, pure L1). Unlike the classification
   pipeline's fixed `Cs` grid, the regularization path is **data-derived per fold**: `alpha_max`
   is probed fresh from the fold's own data (via the library's auto-path generation), then a
   20-point log-spaced grid spanning 5 orders of magnitude below `alpha_max` is searched by 3-fold
   inner CV (C-index-scored) to pick the best alpha. If the selected alpha diverges on the full
   fold data, the procedure steps up to the next larger (more regularized) candidate rather than
   failing the fold. Nonzero coefficients ranked by magnitude, capped at
   `--target-feature-count` (default **10**, matching the classification pipeline — no observed
   combo has come close to either cap, so this is a consistency choice, not a response to a
   binding constraint).

## 5. Consensus feature selection

Identical majority-vote mechanism to the classification pipeline: a feature is retained if
selected in ≥ `--min-fold-n` (default 3) of 5 outer folds with a stable coefficient sign across
those folds. Pipeline raises explicitly if zero features meet this bar.

## 6. Nested cross-validation diagnostics

Out-of-fold risk predictions pooled into a leakage-free dev-only OOF C-index estimate
(`concordance_index_censored`), with bootstrap 95% CI (1,000 resamples). Never touches the lock
set.

## 7. Model optimization (gridsearch stage)

Two model families:

| Model | Tuned | Notes |
|---|---|---|
| LASSO-Cox | — (no separate gridsearch stage) | alpha selected by the same data-derived inner-CV procedure as Section 4, re-run on the full dev set at train time (not carried over from any one outer fold) |
| Random Survival Forest (RSF) | min_samples_split ∈ {5,10,20}, min_samples_leaf ∈ {3,5,10,15}, max_features ∈ {sqrt, 0.3, 0.5} | `n_estimators` fixed at 300; `GridSearchCV` with pre-generated `StratifiedKFold` splits (stratified on event status, `cv=5`) passed explicitly to work around sksurv's structured-array stratification limitation |

## 8. Model testing (train stage)

Both models refit on the full dev set, evaluated once on the held-out lock set:
- **C-index + 95% CI**: `concordance_index_censored` on lock-set risk scores, bootstrap CI
  (1,000 resamples).
- **Time-dependent AUC**: `cumulative_dynamic_auc` (sksurv) at 1/2/3/5 years (365/730/1095/1825
  days), restricted to timepoints strictly inside the observed lock-set event range.
- **Kaplan-Meier curves + log-rank test**: lock-set patients split into high/low risk at the
  **median** predicted risk score; KM curves plotted per group, group difference tested via
  `lifelines.logrank_test`. Skipped (with a note, not a crash) if the median split produces an
  empty risk group.
- **SHAP** (optional, `--shap`): not yet integrated into the pipeline itself for survival models —
  `shap.TreeExplainer` doesn't support `sksurv.ensemble.RandomSurvivalForest`
  (`InvalidModelError`), so SHAP for the RSF model is currently produced via a standalone script
  using `shap.KernelExplainer(model.predict, background_sample)` rather than the pipeline's
  built-in `--shap` flag.
