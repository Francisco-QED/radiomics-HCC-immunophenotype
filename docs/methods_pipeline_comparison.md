# Methods: Classification vs. Recurrence Pipeline — Side-by-Side Comparison

Both pipelines share the same overall architecture (extraction → QC filters → per-fold
univariate screen → FDR → decorrelation → LASSO → consensus voting → nested-CV diagnostics →
gridsearch → lock-set testing) and the same 7 sequence combinations. This table lays out the
pipeline stage by stage, so the points where they're identical and the points where they diverge
(the univariate test/metric/cutoff, and LASSO's regularization strategy, in particular) are
visible at a glance. Full prose write-ups: `docs/methods_classification_pipeline.md`,
`docs/methods_recurrence_pipeline.md`.

| Step | Classification (TILs) | Recurrence (Survival) | Same? |
|---|---|---|---|
| **Feature extraction** | PyRadiomics, 9 sequences, Original+LoG(σ=2/3/4/5)+Wavelet filters, wavelet excluded by default | Identical — same source CSVs | ✅ Identical |
| **Sequence combinations** | 7 pre-specified, LI-RADS-motivated combos | Same 7 combos | ✅ Identical |
| **Dev/lock split** | Stratified on `y_bin` (TILs cutoff), seed=997, `test_size`=30, ICC patients forced into dev | Stratified on event status, same seed/test_size/ICC rule | ⚠️ Same mechanism, **different patient pool** (143 recurrence-evaluable vs. ~135–140 TILs-evaluable) — dev/lock membership not guaranteed to match between pipelines for the same patient |
| **QC filters** (cleanup + interobserver ICC) | CV filter (`--cv-threshold`=0.1) + ICC(3,1)≥0.85 between readers | Identical filters, same defaults | ✅ Identical |
| **Univariate screen** (test, metric, cutoff) | Mann-Whitney U + univariate AUC; keep AUC≥0.65 or ≤0.35 (symmetric) | Univariate Cox PH; keep magnitude of log(HR) above log(1.5) (symmetric) | ❌ Different test family (rank-sum/logistic vs. Cox), forced by binary vs. time-to-event outcomes — same symmetric, two-sided design in both |
| **FDR correction** | Benjamini-Hochberg within the screened pool, q<**0.2** | Identical mechanism, q<**0.2** | ✅ Identical, including the numeric threshold |
| **Decorrelation** | Absolute Spearman correlation > 0.85, keep lower-p-value member of each pair | Identical mechanism and threshold | ✅ Identical |
| **LASSO** | `LogisticRegressionCV`, fixed `Cs` grid, inner CV scored on AUC | `CoxnetSurvivalAnalysis`, **data-derived per-fold** alpha grid, inner CV scored on C-index | ❌ Different regularization-path strategy — Coxnet's original fixed range missed the fold's real signal, forcing the adaptive version |
| **Consensus selection** | Selected in ≥3 of 5 folds, sign-stable, capped at 10 features | Identical mechanism and cap | ✅ Fully unified |
| **Nested-CV diagnostics** | OOF AUC, bootstrap 95% CI | OOF C-index, bootstrap 95% CI | ⚠️ Same bootstrap machinery, different metric |
| **Gridsearch & lock-set testing** | 8 classifiers tuned; AUC/sensitivity/specificity + ROC curves on lock set | RSF (+LASSO-Cox, no separate gridsearch) tuned; C-index/time-dependent AUC + KM curves/log-rank on lock set | ❌ Different model families and evaluation metrics, following from the outcome type |

## Summary of what actually differs

Everything **through decorrelation is either identical or a like-for-like adaptation** of the same
symmetric two-sided screen-then-FDR design (same 0.2 FDR threshold in both). The real divergences:
1. The univariate test itself (rank-sum/logistic vs. Cox) — forced by the binary-vs-time-to-event
   outcome types, not a methodological inconsistency.
2. LASSO's regularization-path strategy (fixed grid vs. per-fold-adaptive grid).
3. Everything downstream of consensus selection (gridsearch/testing) necessarily differs because
   the two endpoints require different model families and evaluation metrics.
