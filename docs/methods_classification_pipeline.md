# Methods: HCC TILs Classification Pipeline

Covers `scripts/radiomics_utils/main_pipeline.py` (stages: `partition` → `gridsearch` → `train`),
orchestrated across sequence combinations by `sequence_sweep.py` / `gridsearch_train_sweep.py`.
Methods only — results are reported separately once the current sweep completes.

## 1. Feature extraction

Radiomics features are extracted with PyRadiomics from 9 MRI sequences: `pre`, `ap`, `pvp`, `tp`,
`hbp`, `t2`, `dwi_b1`, `dwi_b2`, `adc` (`scripts/radiomics_utils/extraction.py`,
`scripts/radiomics_params.yaml`).

- **Preprocessing**: resampled to isotropic 1×1×1mm (`sitkBSpline` interpolation); intensity
  normalized (`normalize: true`, `normalizeScale: 100`). N4 bias-field correction applied to all
  sequences except `adc`, `dwi_b1`, `dwi_b2` (raw signal used for these — ADC is already a derived
  quantitative map, and low-SNR/EPI-distorted raw diffusion images are considered unreliable N4
  candidates).
- **Discretization**: per-sequence fixed bin width, calibrated from the 2σ noise-based IBSI rule
  on liver-patch SNR measurements (`snr_binwidth.py`): `pre`/`ap`/`pvp`/`tp` = 25,
  `hbp`/`dwi_b1`/`dwi_b2` = 40, `t2` = 15, `adc` = 60.
- **Filters**: Original, Laplacian-of-Gaussian (LoG, σ = 2/3/4/5mm), and Wavelet (8 subbands:
  LLL/LLH/LHL/LHH/HLL/HLH/HHL/HHH) — 13 image types per sequence.
- **Feature classes**: shape (Original image only, ~14 features), firstorder (18), GLCM (21,
  SumAverage excluded as redundant with JointAverage), GLRLM (16), GLSZM (16), GLDM (14) — 85
  intensity/texture features per non-shape image type.
- **Yield**: ~1,223 features/sequence (107 Original incl. shape, 372 LoG, 744 Wavelet) ×
  9 sequences ≈ 11,007 raw candidate features across the full cohort (n=140).

**Wavelet exclusion.** `filter_feature_families()` drops all wavelet-derived columns before any
downstream selection step (`--exclude-feature-families`, default `['wavelet']`). Wavelet accounts
for ~61% of raw columns and is the filter family most consistently flagged in radiomics
reproducibility literature as sensitive to acquisition/reconstruction variability; empirical
comparison on this cohort (dev-only nested-CV, wavelet-included vs. excluded, held-out lock set
untouched) showed no performance cost and comparable-or-larger consensus feature sets. LoG was
checked against the lesion-size distribution (median equivalent diameter ≈35mm; only 14.3% of
lesions <20mm) and retained at all 4 sigma values — the "large kernel relative to small lesion"
concern only affects a minority of cases.

## 2. Sequence combination design

Rather than an exhaustive combinatorial sweep, 7 sequence combinations were pre-specified from
LI-RADS diagnostic criteria (arterial hyperenhancement, portal-venous/delayed washout,
hepatobiliary-phase hypointensity, diffusion restriction), so each combination has a stated
rationale independent of any observed result:

| # | Combo | Sequences | Tests |
|---|---|---|---|
| 1 | Base | ap, pvp, hbp, dwi_b1, adc | LI-RADS core: AP hyperenhancement + PVP washout + HBP hypointensity + diffusion |
| 2 | Base + T2 | ap, pvp, hbp, dwi_b1, adc, t2 | Does T2 hyperintensity add anything |
| 3 | Base, pvp→tp | ap, tp, hbp, dwi_b1, adc | Washout read on delayed/transitional phase instead of portal venous |
| 4 | Base, hbp→tp | ap, pvp, tp, dwi_b1, adc | Effect of HBP specifically |
| 5 | Base + pre | pre, ap, pvp, hbp, dwi_b1, adc | Does the unenhanced baseline add anything (6 sequences) |
| 6 | Base, b1→b2 | ap, pvp, hbp, dwi_b2, adc | b-value choice (technical, not biological) |
| 7 | T1-only | pre, ap, pvp, tp, hbp | No diffusion sequences at all — a structurally distinct design, not a single-variable contrast against the base |

## 3. Dev/lock partitioning

Stratified train/test split (`train_test_split`, `stratify=y_bin`, seed=`PARTITION_SEED`=997,
`test_size`=30), computed once per outcome cutoff and shared across all sequence combinations
(the split doesn't depend on which features are used). Patients with double-reader segmentations
(inter-reader ICC cases) are force-included in the dev set before the stratified split runs on
the remainder, so the lock set never contains an ICC patient. Row order is sorted by PatientID
before splitting so the partition depends only on patient identity + seed, not on incidental CSV
row order.

**Outcome**: `TILs_resection` (ordinal 0–3) dichotomized at two cutoffs — `y_bin = (TILs_resection
>= cutoff)`. Cutoff=2 → 0–1 vs. 2–3; cutoff=3 → 0–2 vs. 3.

## 4. Feature filtering pipeline

Everything past sequence/family selection happens **inside each of 5 outer cross-validation
folds**, fit on that fold's training partition only, to prevent leakage into validation:

1. **Global cleanup** (dev-set-wide, before fold splitting): numeric columns only; drop columns
   missing in >10% of patients; drop near-constant columns (`std <= 1e-8`) and low-variation
   columns (coefficient of variation `std/|mean|` at or below `--cv-threshold`, default 0.1),
   with zero-centered features like `ClusterShade` exempted from the CV check (mean≈0 there isn't
   evidence of low variance).
2. **Interobserver stability (ICC) filter**: two-way mixed ICC(3,1), computed vectorized between
   Reader1 and Reader2 segmentations of the same patients, keeping features with ICC ≥
   `--icc-threshold` (default 0.85). Clinical/semantic features (when `--feature-set both`) bypass
   this filter — no double-reader segmentation exists for them.
3. **Per-fold univariate effect-size screen**: Mann-Whitney U test + univariate AUC (unflipped)
   per feature. Kept if `AUC >= --metric-cutoff` (default 0.65) **or** `AUC <= 1 - metric-cutoff`
   (symmetric, catches inversely-associated features).
4. **FDR within the screened pool** (not across the full candidate pool): Benjamini-Hochberg
   applied only to features that passed step 3, threshold `--p-value-cutoff` (default q<0.2).
   Screening on effect size first and correcting only among survivors is standard "independent
   filtering" practice (cf. DESeq2) — applying FDR across the full multi-thousand-feature pool
   would bury real signal, since most of that pool is near-duplicate transforms of the same
   underlying features.
5. **Spearman decorrelation**: pairwise |Spearman ρ| > 0.85 → drop one of each pair, keeping the
   member with the lower univariate p-value.
6. **LASSO**: `LogisticRegressionCV` (L1, `liblinear`, `Cs=np.logspace(-3, 1, 15)`,
   `class_weight='balanced'`, inner CV = `StratifiedKFold` with up to 5 splits scored on AUC) fit
   on the decorrelated pool; C is cross-validated rather than fixed (an untuned regularization
   strength was previously the dominant driver of fold-to-fold selection instability). Nonzero
   coefficients ranked by magnitude, capped at `--lasso-target-count` (default 100).

## 5. Consensus feature selection

Across the 5 outer folds, a feature is retained in the final **consensus set** if it was selected
(nonzero LASSO coefficient) in at least `--min-fold-n` (default 3) of 5 folds **and** its
coefficient sign is stable across every fold in which it was selected (a feature selected with
positive weight in some folds and negative in others is treated as noise, not a stable effect).
Consensus features are ranked by selection frequency then mean absolute coefficient, capped at
`--target-feature-count` (default 10). If zero features meet this bar, the pipeline raises
explicitly rather than silently returning an empty/unusable model.

## 6. Nested cross-validation diagnostics

Out-of-fold predictions from step 4/6 above (each patient predicted only by the model from the
fold where they were held out) are pooled into a per-model honest, leakage-free dev-only AUC
estimate, with a bootstrap 95% CI (1,000 resamples, `bootstrap_ci()`) and the Youden's-J-optimal
decision threshold. This never touches the lock set, so it can be inspected repeatedly without
eroding its integrity as a held-out set.

## 7. Model optimization (gridsearch stage)

The final consensus feature set (fit on the full dev set: KNN imputation, k=5, then
`StandardScaler`) is used to tune 8 candidate models via `GridSearchCV` (`cv=5`,
`scoring='roc_auc'`, `n_jobs=6`, class-imbalance handled via `class_weight='balanced'` /
`scale_pos_weight`):

| Model | Tuned hyperparameters | Notes |
|---|---|---|
| Logistic regression | C ∈ {0.001, 0.01, 0.1, 1, 10} | L2, `lbfgs` |
| SVM (linear) | C ∈ {0.001, 0.01, 0.1, 1, 10} | |
| SVM (RBF) | C ∈ {1, 10, 100, 1000}, γ ∈ {scale, auto, 0.01} | |
| Random forest | max_depth ∈ {2..6}, min_samples_leaf ∈ {1,2,5,10} | n_estimators fixed at 300 (AUC flat past ~200 trees on this dev set) |
| Extra trees | max_depth ∈ {2..6}, min_samples_leaf ∈ {1,2,5,10} | n_estimators fixed at 300 |
| XGBoost | n_estimators ∈ {25..300}, max_depth ∈ {1,2,3}, learning_rate ∈ {0.01..0.3}, subsample ∈ {0.6,0.8,1.0} | shallow depth to limit overfitting on small n |
| Gaussian process | — (no grid) | RBF kernel length-scale fit internally via marginal-likelihood maximization at `.fit()`; unweighted (no `class_weight` support) |
| TabPFN | — (no grid) | CPU, `torch.set_num_threads(1)` (multi-thread default caused multi-minute hangs on this small dataset) |

## 8. Model testing (train stage)

Each tuned model is refit on the full dev set and evaluated once on the held-out lock set:
- **Point estimate + 95% CI**: AUC, sensitivity, and specificity via the same 1,000-resample
  bootstrap procedure as the nested-CV diagnostics (directly comparable numbers).
  Youden-threshold-calibrated sensitivity/specificity reported alongside default-threshold values.
- **Best model selection**: the model with the highest raw (unrounded) lock-set AUC among
  successful fits — resolved from the underlying float, not the 2-decimal display string, to
  avoid tie-breaking on display precision.
- **ROC comparison plots**: one across all 8 models, one excluding `svm_linear` (a near-degenerate
  fit in several combos — sensitivity/specificity pinned at 1.0/0.0 — that otherwise dominates the
  shared axes).
- **SHAP** (optional, `--shap`, run separately post-sweep rather than inside the sweep itself —
  `KernelExplainer` for GP/TabPFN takes hours per combo): `TreeExplainer` for tree models,
  `LinearExplainer` for logistic, `KernelExplainer` (50-sample background) for SVM/GP/TabPFN.
