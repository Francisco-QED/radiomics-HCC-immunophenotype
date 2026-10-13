##### ------------ Clinical-only benchmark for prediction of immunophenotype in HCC -------------- #####
# Companion to main_pipeline.py: same dev/lock split (same PARTITION_SEED), same downstream
# gridsearch/train machinery, imported directly from main_pipeline.py rather than duplicated. The
# ONLY thing that's actually different here is feature selection: with only ~12 hand-curated clinical
# features (vs. thousands of radiomics features), the LASSO/Spearman/FDR pruning built to fight
# multiple-testing at high dimensionality is unnecessary. Instead this does an exhaustive search over
# every non-empty feature subset (2^12 - 1 = 4095, trivial at this scale), scored by 5-fold CV AUC
# with a fixed, untuned L2 logistic regression, and picks the smallest subset within 1 SE of the best
# (the "1-SE rule" from Hastie & Tibshirani) so a marginally higher AUC doesn't buy a bigger feature
# set than the data can actually justify.
#
# Pipeline overview (run one stage at a time via --stage):
#   --stage partition  : partition dev/lock (same seed as main_pipeline), exhaustive subset search,
#                         nested-CV diagnostics, save processed dev/lock artifacts.
#   --stage gridsearch : identical to main_pipeline's gridsearch stage (reused directly).
#   --stage train      : identical to main_pipeline's train stage (reused directly).
#
# Example:
#   python clinical_only_pipeline.py --stage partition
#   python clinical_only_pipeline.py --stage gridsearch
#   python clinical_only_pipeline.py --stage train --bootstrap 2000 --shap

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pickle
import argparse
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import main_pipeline as mp

RANDOM_STATE = mp.RANDOM_STATE
PARTITION_SEED = mp.PARTITION_SEED
ALL_MODELS = mp.ALL_MODELS

DEFAULT_STAGE = 'train'  # partition/gridsearch/train -- gridsearch is the most common, so default to that


# =====================================================================================================
# Exhaustive feature-subset search (replaces LASSO/Spearman/FDR pruning -- see module docstring)
# =====================================================================================================

# 5-fold CV AUC for one candidate feature subset, using a fixed, untuned L2 logistic regression.
# Untuned on purpose: hyperparameter tuning happens once, later, on the winning subset only (via
# main_pipeline.tune_model) -- tuning inside the subset search would confound "is this subset good"
# with "did this subset get lucky with its hyperparameters" across 4095 candidates.
def _cv_auc_for_subset(X_dev, y_dev, feature_subset, cv_splits, random_state):
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    fold_aucs = []
    cols = list(feature_subset)

    for train_idx, val_idx in skf.split(X_dev, y_dev):
        X_tr, y_tr = X_dev.iloc[train_idx][cols], y_dev.iloc[train_idx]
        X_val, y_val = X_dev.iloc[val_idx][cols], y_dev.iloc[val_idx]

        imputer = KNNImputer(n_neighbors=5)
        X_tr_imp = imputer.fit_transform(X_tr)
        X_val_imp = imputer.transform(X_val)
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr_imp)
        X_val_scaled = scaler.transform(X_val_imp)

        model = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=1000,
                                    class_weight='balanced', random_state=random_state)
        model.fit(X_tr_scaled, y_tr)
        fold_aucs.append(roc_auc_score(y_val, model.predict_proba(X_val_scaled)[:, 1]))

    return np.array(fold_aucs)


# Scores every non-empty feature subset, then applies the 1-SE rule: among subsets whose mean CV AUC
# is within one standard error of the best, pick the smallest. Returns (chosen_features, results_df)
# where results_df is the full ranked search (saved to disk for transparency/audit).
def select_best_clinical_subset(X_dev, y_dev, cv_splits=5, random_state=RANDOM_STATE):
    features = list(X_dev.columns)
    all_subsets = [c for r in range(1, len(features) + 1) for c in combinations(features, r)]
    print(f"Exhaustive subset search: {len(all_subsets)} subsets of {len(features)} clinical features...")

    results = []
    for subset in tqdm(all_subsets, desc="Subset search"):
        fold_aucs = _cv_auc_for_subset(X_dev, y_dev, subset, cv_splits, random_state)
        results.append({
            'features': ', '.join(subset),
            'n_features': len(subset),
            'mean_auc': fold_aucs.mean(),
            'se_auc': fold_aucs.std(ddof=1) / np.sqrt(len(fold_aucs)),
        })

    results_df = pd.DataFrame(results).sort_values('mean_auc', ascending=False).reset_index(drop=True)
    best_auc = results_df.loc[0, 'mean_auc']
    best_se = results_df.loc[0, 'se_auc']
    within_1se = results_df[results_df['mean_auc'] >= best_auc - best_se]
    chosen = within_1se.sort_values('n_features').iloc[0]
    chosen_features = [f.strip() for f in chosen['features'].split(',')]

    print(f"Best subset by raw CV AUC: {results_df.loc[0, 'features']} (AUC={best_auc:.3f})")
    print(f"Chosen subset (1-SE rule, smallest within 1 SE of best): {chosen['features']} "
          f"(AUC={chosen['mean_auc']:.3f}, n={chosen['n_features']})")

    return chosen_features, results_df


# =====================================================================================================
# Lightweight nested CV on the winning subset (no fold-specific selection -- the subset is already
# fixed from run_partition_stage's search above; this just tunes+fits each model per fold to get an
# honest, leakage-free OOF AUC, same format as main_pipeline's nested_cv_diagnostics.csv).
# =====================================================================================================

def run_nested_cv(X_dev, y_dev, model_names=None, tuning_cv=5, random_state=RANDOM_STATE):
    if model_names is None:
        model_names = ALL_MODELS

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    oof_predictions = pd.DataFrame(np.nan, index=X_dev.index, columns=model_names)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_dev, y_dev)):
        X_tr, y_tr = X_dev.iloc[train_idx], y_dev.iloc[train_idx]
        X_val = X_dev.iloc[val_idx]

        imputer = KNNImputer(n_neighbors=5)
        X_tr_imp = pd.DataFrame(imputer.fit_transform(X_tr), columns=X_tr.columns, index=X_tr.index)
        X_val_imp = pd.DataFrame(imputer.transform(X_val), columns=X_val.columns, index=X_val.index)
        scaler = StandardScaler()
        X_tr_scaled = pd.DataFrame(scaler.fit_transform(X_tr_imp), columns=X_tr.columns, index=X_tr.index)
        X_val_scaled = pd.DataFrame(scaler.transform(X_val_imp), columns=X_val.columns, index=X_val.index)

        for model_name in model_names:
            model = mp.tune_model(X_tr_scaled, y_tr, model_name, cv=tuning_cv, random_state=random_state)
            oof_predictions.loc[X_val.index, model_name] = model.predict_proba(X_val_scaled)[:, 1]

        print(f"Fold {fold + 1} complete.")

    return oof_predictions


# =====================================================================================================
# Partition stage
# =====================================================================================================

def run_partition_stage(args):
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    df_clinical_feats = mp.load_clinical_features(args.clinical_data_path)

    X_dev, y_dev, X_lock, y_lock, df_clinical_dev, df_clinical_lock = mp.partition_with_constraints(
        df_clinical_feats,
        args.outcomes_path,
        args.icc_path,
        clinical_data_path=args.clinical_data_path,
        label_col=args.label_col,
        cutoff=args.cutoff,
        test_size=args.test_size,
        output_dir=args.output_dir,
        random_state=PARTITION_SEED,
    )

    mp.compare_dev_test(df_clinical_dev, df_clinical_lock, args.compare_variables,
                         output_dir=args.output_dir, outcome_col=args.compare_outcome_col)

    # cv_threshold=0.0: the clinical feature set is already hand-curated for variance (see
    # main_pipeline.load_clinical_features docstring), so the near-zero-variance filter built for
    # radiomics feature scale weirdness has nothing to do here. This step still does the useful
    # generic hygiene: numeric coercion, inf->NaN, drop fully-empty rows/columns.
    X_dev_clean = mp.global_cleanup(X_dev, id_col='PatientID', cv_threshold=0.0)
    y_dev_filtered = y_dev.loc[X_dev_clean.index]
    print(f"After cleanup: X_dev has {X_dev_clean.shape[0]} samples, {X_dev_clean.shape[1]} features")

    consensus_features, subset_search_results = select_best_clinical_subset(
        X_dev_clean, y_dev_filtered, cv_splits=args.cv, random_state=RANDOM_STATE)
    subset_search_results.to_csv(f"{args.output_dir}/subset_search_results.csv", index=False)
    pd.Series(consensus_features, name='feature').to_csv(f"{args.output_dir}/consensus_features.csv", index=False)
    print(f"Consensus features identified: {len(consensus_features)}")
    print(consensus_features)

    X_dev_subset = X_dev_clean[consensus_features]
    X_lock_subset = X_lock[consensus_features]

    # Global scorecard (univariate AUC/OR/p-value per feature), for reporting/interpretability --
    # same artifact main_pipeline produces, not used to filter consensus_features here.
    diag_imputer = KNNImputer(n_neighbors=5)
    X_dev_imp_diag = pd.DataFrame(diag_imputer.fit_transform(X_dev_subset), columns=consensus_features, index=X_dev_subset.index)
    diag_scaler = StandardScaler()
    X_dev_scaled_diag = pd.DataFrame(diag_scaler.fit_transform(X_dev_imp_diag), columns=consensus_features, index=X_dev_subset.index)
    global_scorecard = mp.calculate_feature_metrics(X_dev_scaled_diag, y_dev_filtered)
    global_scorecard.to_csv(f"{args.output_dir}/univariate_metrics.csv")

    # Honest, dev-only nested CV estimate -- never touches X_lock/y_lock.
    oof_predictions = run_nested_cv(X_dev_subset, y_dev_filtered, model_names=ALL_MODELS,
                                     tuning_cv=args.cv, random_state=RANDOM_STATE)
    nested_cv_summary = mp.summarize_nested_cv(oof_predictions, y_dev_filtered, random_state=RANDOM_STATE)
    nested_cv_summary.to_csv(f"{args.output_dir}/nested_cv_diagnostics.csv", index=False)
    nested_cv_summary[['model', 'youden_threshold']].to_csv(f"{args.output_dir}/calibrated_thresholds.csv", index=False)

    print(f"\n{'=' * 100}")
    print("DEV-ONLY NESTED CV DIAGNOSTICS (honest, leakage-free estimate -- does not touch the lock set)")
    print(f"{'=' * 100}")
    print(nested_cv_summary.to_string(index=False))
    print(f"{'=' * 100}\n")

    mp.run_feature_diagnostics(X_dev_subset, y_dev_filtered, X_lock, y_lock, consensus_features, args.output_dir)

    # Final dev/lock preprocessing: fit imputer/scaler on dev ONLY, transform lock -- same artifact
    # names/layout as main_pipeline so the gridsearch/train stages work completely unmodified.
    imputer = KNNImputer(n_neighbors=5)
    scaler = StandardScaler()

    X_dev_impute = pd.DataFrame(imputer.fit_transform(X_dev_subset), columns=consensus_features, index=X_dev_subset.index)
    X_dev_final = pd.DataFrame(scaler.fit_transform(X_dev_impute), columns=consensus_features, index=X_dev_subset.index)
    X_lock_impute = pd.DataFrame(imputer.transform(X_lock_subset), columns=consensus_features, index=X_lock_subset.index)
    X_lock_final = pd.DataFrame(scaler.transform(X_lock_impute), columns=consensus_features, index=X_lock_subset.index)

    X_dev_final.to_csv(f"{args.output_dir}/X_dev_processed.csv")
    X_lock_final.to_csv(f"{args.output_dir}/X_lock_processed.csv")
    pd.Series(y_dev_filtered, name='y_dev').to_csv(f"{args.output_dir}/y_dev.csv")
    pd.Series(y_lock, name='y_lock').to_csv(f"{args.output_dir}/y_lock.csv")
    pd.DataFrame({'n_features_in': [imputer.n_features_in_]}).to_csv(f"{args.output_dir}/imputer_n_features.csv", index=False)

    with open(f"{args.output_dir}/imputer.pkl", 'wb') as f:
        pickle.dump(imputer, f)
    with open(f"{args.output_dir}/scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)

    print(f"Artifacts saved to {args.output_dir}")
    print(f"Processed Development Set: {X_dev_final.shape}")
    print(f"Processed Lock Set: {X_lock_final.shape}")
    print(f"Consensus Features: {len(consensus_features)}")


# =====================================================================================================
# CLI -- mirrors main_pipeline.py's argument names/defaults so paths and conventions line up.
# =====================================================================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Clinical-only benchmark: same dev/lock split as main_pipeline.py, exhaustive '
                    'feature-subset search instead of LASSO pruning, then the same gridsearch/train stages.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--stage', choices=['partition', 'gridsearch', 'train'], default=DEFAULT_STAGE)

    # --- partition stage ---
    parser.add_argument('--path-r1', type=str,
        default='/Volumes/T7 Shield/Restrepo/HCC_Retrospective_radiomics/Results/Radiomics_sheets/inter-reader/R-Reader1_I-sitkBSpline_S-1mm_W-5.csv',
        help="Only used as the source of which PatientIDs have inter-reader (ICC) segmentations, "
             "so those patients are forced into dev here too -- consistent with main_pipeline.")
    parser.add_argument('--outcomes-path', type=str,
        default='/Volumes/T7 Shield/Restrepo/HCC_Retrospective_radiomics/databases/immunophenotype_outcomes.xlsx')
    parser.add_argument('--clinical-data-path', type=str,
        default='/Volumes/T7 Shield/Restrepo/HCC_Retrospective_radiomics/databases/SA1-2_clinical_imaging.xlsx')
    parser.add_argument('--icc-path', type=str, default=None,
        help="Defaults to --path-r1.")
    parser.add_argument('--label-col', type=str, default='TILs_resection')
    parser.add_argument('--cutoff', type=float, default=2)
    parser.add_argument('--test-size', type=int, default=30)
    parser.add_argument('--cv', type=int, default=5,
        help="Used both for the exhaustive subset search's CV AUC and for per-fold model tuning.")
    parser.add_argument('--compare-variables', nargs='+', default=[
        'Age', 'Gender', 'BMI', 'Cirrhosis', 'Capsule (on PVP yes/no)', 'Mosaic appearance (yes/no)',
        'Macrovascular invasion (yes/no)', 'Maximum tumor size on T1 portal venous phase (PVP)',
        'TILs_resection', 'TLS_resection',
    ])
    parser.add_argument('--compare-outcome-col', type=str, default='TILs_resection')

    # --- gridsearch stage (passed straight through to main_pipeline.run_gridsearch_stage) ---
    parser.add_argument('--input-dir', type=str, default=None)
    parser.add_argument('--scoring', type=str, default='roc_auc')
    parser.add_argument('--n-jobs', type=int, default=6)

    # --- train stage (passed straight through to main_pipeline.run_train_stage) ---
    parser.add_argument('--input-data-dir', type=str, default=None)
    parser.add_argument('--input-params-dir', type=str, default=None)
    parser.add_argument('--bootstrap', type=int, default=1000)
    parser.add_argument('--threshold', type=float, default=mp.TARGET_THRESHOLD)
    parser.add_argument('--shap', action='store_true')

    # --- shared ---
    parser.add_argument('--models', nargs='+', default=None)
    parser.add_argument('--output-dir', type=str, default=None)

    args = parser.parse_args()

    if args.icc_path is None:
        args.icc_path = args.path_r1

    cutoff_val = int(args.cutoff) if args.cutoff == int(args.cutoff) else args.cutoff
    neg_max = int(args.cutoff) - 1
    cutoff_label = f"tils_{cutoff_val}_vs_0-{neg_max}"
    # Outputs go to OneDrive; source data (clinical/outcomes spreadsheets, ICC reference CSV) stays
    # on T7 Shield -- see --clinical-data-path/--outcomes-path/--path-r1 defaults above.
    _base = (f'/Users/Francisco/Library/CloudStorage/OneDrive-TheMountSinaiHospital/'
             f'Regeneron retrospective radiomics/Results/main_pipeline/immunophenotype/'
             f'{cutoff_label}/clinical')

    if args.output_dir is None:
        if args.stage == 'partition':
            args.output_dir = f"{_base}/clean_partition_select"
        elif args.stage == 'gridsearch':
            args.output_dir = f"{_base}/optimized_params"
        elif args.stage == 'train':
            args.output_dir = f"{_base}/trained_model_outputs"

    if args.input_dir is None:
        args.input_dir = f"{_base}/clean_partition_select"
    if args.input_data_dir is None:
        args.input_data_dir = f"{_base}/clean_partition_select"
    if args.input_params_dir is None:
        args.input_params_dir = f"{_base}/optimized_params"

    return args


if __name__ == "__main__":
    args = parse_arguments()

    print("\n" + "=" * 100)
    print(f"CLINICAL-ONLY BENCHMARK - STAGE: {args.stage.upper()}")
    print(f"Output directory: {args.output_dir}")
    if args.models:
        print(f"Models: {', '.join(args.models)}")
    print("=" * 100)

    if args.stage == 'partition':
        run_partition_stage(args)
    elif args.stage == 'gridsearch':
        mp.run_gridsearch_stage(args)
    elif args.stage == 'train':
        mp.run_train_stage(args)

    print(f"{'=' * 100}")
    print(f"STAGE '{args.stage}' COMPLETE")
    print(f"{'=' * 100}\n")
