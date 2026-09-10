"""
Fast dev-only small-combination screen for ordinal TILs_resection modeling.

This is the ordinal companion to run_signal_screen_v2.py:
  - Uses every 1-, 2-, and 3-sequence combination from the sequence pool.
  - Excludes wavelet/log-sigma by default.
  - Performs fold-internal univariate feature filtering by absolute Spearman correlation
    with the ordinal TILs label.
  - Fits simple regression-style models to predict the ordinal score.
  - Reports ordinal metrics plus binary AUCs derived from the predicted ordinal score:
      TILs >= 2 vs 0-1 and TILs >= 3 vs 0-2.
  - Does not evaluate the lock set.
"""

import argparse
import itertools
import json
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from sklearn.impute import KNNImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from main_pipeline_v2 import RANDOM_STATE, load_dev_lock_clean


DEFAULT_SEQUENCE_POOL = ["pre", "ap", "pvp", "tp", "hbp", "t2", "dwi_b1", "dwi_b2", "adc"]


def make_combos(sequence_pool, max_sequences):
    combos = []
    for n in range(1, max_sequences + 1):
        combos.extend([list(c) for c in itertools.combinations(sequence_pool, n)])
    return combos


def results_root(analysis_label):
    return (
        Path.home() / "Library" / "CloudStorage" / "OneDrive-TheMountSinaiHospital"
        / "Regeneron retrospective radiomics" / "Results" / "main_pipeline"
        / "immunophenotype" / "signal_screen_ordinal_v2" / analysis_label
    )


def load_ordinal_labels(outcomes_path, label_col):
    df = pd.read_excel(outcomes_path)
    y = pd.to_numeric(df[label_col], errors="coerce")
    out = pd.Series(y.values, index=df["PatientID"].values, name=label_col)
    return out.dropna()


def spearman_prefilter_features(X_tr, y_tr, top_k):
    imputer = KNNImputer(n_neighbors=5)
    X_imp = pd.DataFrame(imputer.fit_transform(X_tr), index=X_tr.index, columns=X_tr.columns)
    scores = {}
    y = pd.Series(y_tr, index=X_tr.index)
    for col in X_imp.columns:
        values = X_imp[col]
        if values.std(ddof=0) == 0:
            scores[col] = 0.0
            continue
        rho = values.corr(y, method="spearman")
        scores[col] = abs(float(rho)) if pd.notna(rho) else 0.0
    ranked = pd.Series(scores).sort_values(ascending=False)
    return ranked.head(min(top_k, len(ranked))).index.tolist(), ranked


def make_model_grid(model_name):
    if model_name == "ridge":
        pipe = Pipeline([
            ("imputer", KNNImputer(n_neighbors=5)),
            ("scaler", StandardScaler()),
            ("model", Ridge(random_state=RANDOM_STATE)),
        ])
        grid = {"model__alpha": [0.1, 1.0, 10.0, 100.0]}
    elif model_name == "elasticnet":
        pipe = Pipeline([
            ("imputer", KNNImputer(n_neighbors=5)),
            ("scaler", StandardScaler()),
            ("model", ElasticNet(max_iter=10000, random_state=RANDOM_STATE)),
        ])
        grid = {
            "model__alpha": [0.001, 0.01, 0.1, 1.0],
            "model__l1_ratio": [0.1, 0.3, 0.5, 0.7],
        }
    elif model_name == "svr_linear":
        pipe = Pipeline([
            ("imputer", KNNImputer(n_neighbors=5)),
            ("scaler", StandardScaler()),
            ("model", SVR(kernel="linear")),
        ])
        grid = {"model__C": [0.01, 0.1, 1.0, 10.0]}
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return pipe, grid


def safe_auc(y_true_binary, score):
    if pd.Series(y_true_binary).nunique() < 2:
        return np.nan
    return roc_auc_score(y_true_binary, score)


def nested_oof_ordinal_screen(X, y, top_ks, model_names):
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    pred_cols = [(k, m) for k in top_ks for m in model_names]
    oof = pd.DataFrame(np.nan, index=X.index, columns=pd.MultiIndex.from_tuples(pred_cols))
    fold_rows = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_val = X.iloc[val_idx]
        inner = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

        for top_k in top_ks:
            features, ranked = spearman_prefilter_features(X_tr, y_tr, top_k)
            if not features:
                continue
            for model_name in model_names:
                pipe, grid = make_model_grid(model_name)
                gs = GridSearchCV(pipe, grid, scoring="neg_mean_absolute_error",
                                  cv=inner, n_jobs=1, refit=True)
                gs.fit(X_tr[features], y_tr)
                oof.loc[X_val.index, (top_k, model_name)] = gs.predict(X_val[features])
                fold_rows.append({
                    "fold": fold,
                    "top_k": top_k,
                    "model": model_name,
                    "n_features": len(features),
                    "best_inner_neg_mae": gs.best_score_,
                    "best_params": json.dumps(gs.best_params_),
                    "top_feature": ranked.index[0] if len(ranked) else "",
                    "top_feature_abs_spearman": float(ranked.iloc[0]) if len(ranked) else np.nan,
                })

    rows = []
    flat_oof = pd.DataFrame(index=X.index)
    for top_k, model_name in pred_cols:
        pred = oof[(top_k, model_name)]
        valid = pred.notna()
        if valid.sum() == 0:
            continue
        y_valid = y.loc[valid]
        pred_valid = pred.loc[valid]
        spearman = pred_valid.corr(y_valid, method="spearman")
        mae = mean_absolute_error(y_valid, pred_valid)
        auc_ge2 = safe_auc((y_valid >= 2).astype(int), pred_valid)
        auc_ge3 = safe_auc((y_valid >= 3).astype(int), pred_valid)
        col = f"k{top_k}_{model_name}"
        flat_oof[col] = pred
        rows.append({
            "top_k": top_k,
            "model": model_name,
            "spearman": spearman,
            "mae": mae,
            "auc_tils_ge2": auc_ge2,
            "auc_tils_ge3": auc_ge3,
            "n_oof": int(valid.sum()),
        })
    return pd.DataFrame(rows), pd.DataFrame(fold_rows), flat_oof


def make_pipeline_args(args, sequences, output_dir, clinical_output_dir):
    return SimpleNamespace(
        sequences=sequences,
        features_path=args.features_path,
        path_r1=args.path_r1,
        path_r2=args.path_r2,
        outcomes_path=args.outcomes_path,
        clinical_data_path=args.clinical_data_path,
        icc_path=args.icc_path or args.path_r1,
        label_col=args.label_col,
        cutoff=args.partition_cutoff,
        test_size=args.test_size,
        feature_set=args.feature_set,
        exclude_feature_families=args.exclude_feature_families,
        cv_threshold=args.cv_threshold,
        icc_threshold=args.icc_threshold,
        output_dir=str(output_dir),
        clinical_output_dir=str(clinical_output_dir),
    )


def run_one(args, root, sequences, y_ordinal_all):
    combo = "_".join(sequences)
    out_dir = root / "combinations" / combo / "dev"
    clinical_out = root / "summary_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    clinical_out.mkdir(parents=True, exist_ok=True)

    start = time.time()
    pipe_args = make_pipeline_args(args, sequences, out_dir, clinical_out)
    X_dev, _y_bin_dev, X_lock, _y_bin_lock = load_dev_lock_clean(pipe_args)
    y_dev = y_ordinal_all.loc[X_dev.index].dropna()
    X_dev = X_dev.loc[y_dev.index]
    y_lock = y_ordinal_all.reindex(X_lock.index).dropna()

    pd.Series(y_dev, name=args.label_col).to_csv(out_dir / "y_dev_ordinal.csv")
    pd.Series(y_lock, name=args.label_col).to_csv(out_dir / "y_lock_ordinal.csv")
    pd.Series(X_dev.columns, name="feature").to_csv(out_dir / "screen_feature_pool.csv", index=False)

    summary, folds, oof = nested_oof_ordinal_screen(X_dev, y_dev, args.top_ks, args.models)
    summary.to_csv(out_dir / "nested_oof_ordinal_signal_screen.csv", index=False)
    folds.to_csv(out_dir / "nested_oof_ordinal_signal_screen_folds.csv", index=False)
    oof.to_csv(out_dir / "nested_oof_ordinal_signal_screen_predictions.csv")

    row = {
        "sequences": combo,
        "n_sequences": len(sequences),
        "n_features_after_cleanup_icc": X_dev.shape[1],
        "n_dev": len(y_dev),
        "tils_counts_dev": json.dumps(y_dev.value_counts().sort_index().astype(int).to_dict()),
        "elapsed_sec": round(time.time() - start, 1),
        "output_dir": str(out_dir),
        "status": "success",
    }
    if len(summary):
        best = summary.sort_values(["spearman", "auc_tils_ge3"], ascending=False).iloc[0]
        row.update({
            "best_top_k": int(best["top_k"]),
            "best_model": best["model"],
            "best_spearman": round(float(best["spearman"]), 4),
            "best_mae": round(float(best["mae"]), 4),
            "best_auc_tils_ge2": round(float(best["auc_tils_ge2"]), 4) if pd.notna(best["auc_tils_ge2"]) else np.nan,
            "best_auc_tils_ge3": round(float(best["auc_tils_ge3"]), 4) if pd.notna(best["auc_tils_ge3"]) else np.nan,
        })
    return row


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-label", default="small_combos_leq3_nowavelet_nologsigma_spearmanfilter")
    parser.add_argument("--top-ks", nargs="+", type=int, default=[25, 50, 100])
    parser.add_argument("--models", nargs="+", choices=["ridge", "elasticnet", "svr_linear"],
                        default=["ridge", "elasticnet", "svr_linear"])
    parser.add_argument("--sequence-pool", nargs="+", default=DEFAULT_SEQUENCE_POOL)
    parser.add_argument("--max-sequences", type=int, default=3)
    parser.add_argument("--partition-cutoff", type=int, default=3,
                        help="Only used to create the same dev/lock split machinery; ordinal labels replace y_bin.")
    parser.add_argument("--feature-set", choices=["radiomics", "both"], default="radiomics")
    parser.add_argument("--exclude-feature-families", nargs="*", default=["wavelet", "log-sigma"])
    parser.add_argument("--features-path", type=str,
        default="/Volumes/T7 Shield/Restrepo/HCC_Retrospective_radiomics/Results/Radiomics_sheets/SA1_2/R-SA1_2_I-sitkBSpline_S-1mm_W-perseq.csv")
    parser.add_argument("--path-r1", type=str,
        default="/Volumes/T7 Shield/Restrepo/HCC_Retrospective_radiomics/Results/Radiomics_sheets/inter-reader/Reader1/R-Reader1_I-sitkBSpline_S-1mm_W-perseq.csv")
    parser.add_argument("--path-r2", type=str,
        default="/Volumes/T7 Shield/Restrepo/HCC_Retrospective_radiomics/Results/Radiomics_sheets/inter-reader/Reader2/R-Reader2_I-sitkBSpline_S-1mm_W-perseq.csv")
    parser.add_argument("--outcomes-path", type=str,
        default="/Volumes/T7 Shield/Restrepo/HCC_Retrospective_radiomics/databases/immunophenotype_outcomes.xlsx")
    parser.add_argument("--clinical-data-path", type=str,
        default="/Volumes/T7 Shield/Restrepo/HCC_Retrospective_radiomics/databases/SA1-2_clinical_imaging.xlsx")
    parser.add_argument("--icc-path", type=str, default=None)
    parser.add_argument("--label-col", type=str, default="TILs_resection")
    parser.add_argument("--test-size", type=int, default=30)
    parser.add_argument("--cv-threshold", type=float, default=0.1)
    parser.add_argument("--icc-threshold", type=float, default=0.85)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_arguments()
    root = results_root(args.analysis_label)
    combos = make_combos(args.sequence_pool, args.max_sequences)
    if args.overwrite and root.exists() and not args.dry_run:
        shutil.rmtree(root)
    summary_dir = root / "summary_results"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = summary_dir / "ordinal_signal_screen_summary.csv"

    print("Ordinal signal screen will write outside the repo:")
    print(f"  {root}")
    print(f"  top_ks={args.top_ks}")
    print(f"  models={args.models}")
    print(f"  sequence_pool={args.sequence_pool}")
    print(f"  max_sequences={args.max_sequences}")
    print(f"  exclude_feature_families={args.exclude_feature_families}")
    print(f"  partition_cutoff={args.partition_cutoff}")
    print(f"  overwrite={args.overwrite}")
    print(f"  n_combos={len(combos)}")
    for combo in combos:
        print(f"  {' '.join(combo)}")
    if args.dry_run:
        return

    y_ordinal_all = load_ordinal_labels(args.outcomes_path, args.label_col)
    rows = []
    for idx, sequences in enumerate(combos, start=1):
        print(f"\n[{idx}/{len(combos)}] {' '.join(sequences)}", flush=True)
        try:
            row = run_one(args, root, sequences, y_ordinal_all)
        except Exception as exc:
            row = {
                "sequences": "_".join(sequences),
                "n_sequences": len(sequences),
                "status": "failed",
                "error": repr(exc),
            }
        rows.append(row)
        pd.DataFrame(rows).to_csv(summary_csv, index=False)
        print(f"  -> {row}", flush=True)

    df = pd.DataFrame(rows)
    print("\nORDINAL SIGNAL SCREEN SUMMARY")
    if "best_spearman" in df.columns:
        print(df.sort_values(["best_spearman", "best_auc_tils_ge3"], ascending=False).head(30).to_string(index=False))
    else:
        print(df.to_string(index=False))
    print(f"\nSummary CSV: {summary_csv}")


if __name__ == "__main__":
    main()
