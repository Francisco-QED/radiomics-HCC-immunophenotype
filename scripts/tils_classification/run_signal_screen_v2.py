"""
Fast dev-only signal screen for radiomics immunophenotype modeling.

Purpose:
  - Search many sequence combinations and both binary TIL cutoffs quickly.
  - Favor OOF AUC over stability selection.
  - Reintroduce univariate AUC filtering only inside each outer training fold.
  - Exclude wavelet/log-sigma by default to reduce noise and speed up the screen.
  - Write all outputs to OneDrive, not the repository.

This is exploratory model discovery. It does not evaluate the lock set.
"""

import argparse
import itertools
import shutil
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from main_pipeline_v2 import RANDOM_STATE, bootstrap_ci, load_dev_lock_clean


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
        / "immunophenotype" / "signal_screen_v2" / analysis_label
    )


def auc_prefilter_features(X_tr, y_tr, top_k):
    imputer = KNNImputer(n_neighbors=5)
    X_imp = pd.DataFrame(imputer.fit_transform(X_tr), index=X_tr.index, columns=X_tr.columns)
    scores = {}
    y_arr = np.asarray(y_tr)
    for col in X_imp.columns:
        values = X_imp[col].to_numpy()
        if np.nanstd(values) == 0:
            scores[col] = 0.0
            continue
        try:
            auc = roc_auc_score(y_arr, values)
            scores[col] = abs(float(auc) - 0.5)
        except ValueError:
            scores[col] = 0.0
    ranked = pd.Series(scores).sort_values(ascending=False)
    return ranked.head(min(top_k, len(ranked))).index.tolist(), ranked


def make_model_grid(model_name):
    if model_name == "ridge":
        pipe = Pipeline([
            ("imputer", KNNImputer(n_neighbors=5)),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                penalty="l2", solver="lbfgs", class_weight="balanced",
                max_iter=5000, random_state=RANDOM_STATE,
            )),
        ])
        grid = {"model__C": [0.01, 0.1, 1.0, 10.0]}
    elif model_name == "elasticnet":
        pipe = Pipeline([
            ("imputer", KNNImputer(n_neighbors=5)),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                penalty="elasticnet", solver="saga", class_weight="balanced",
                max_iter=5000, random_state=RANDOM_STATE,
            )),
        ])
        grid = {"model__C": [0.01, 0.1, 1.0], "model__l1_ratio": [0.1, 0.3, 0.5]}
    elif model_name == "svm_linear":
        pipe = Pipeline([
            ("imputer", KNNImputer(n_neighbors=5)),
            ("scaler", StandardScaler()),
            ("model", SVC(kernel="linear", probability=True, class_weight="balanced",
                          random_state=RANDOM_STATE)),
        ])
        grid = {"model__C": [0.01, 0.1, 1.0]}
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return pipe, grid


def nested_oof_screen(X, y, top_ks, model_names):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    pred_cols = [(k, m) for k in top_ks for m in model_names]
    oof = pd.DataFrame(np.nan, index=X.index, columns=pd.MultiIndex.from_tuples(pred_cols))
    fold_rows = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_val = X.iloc[val_idx]
        safe_inner = min(3, int(y_tr.value_counts().min()))
        if safe_inner < 2:
            continue
        inner = StratifiedKFold(n_splits=safe_inner, shuffle=True, random_state=RANDOM_STATE)

        for top_k in top_ks:
            features, ranked = auc_prefilter_features(X_tr, y_tr, top_k)
            if not features:
                continue
            for model_name in model_names:
                pipe, grid = make_model_grid(model_name)
                gs = GridSearchCV(pipe, grid, scoring="roc_auc", cv=inner, n_jobs=1, refit=True)
                gs.fit(X_tr[features], y_tr)
                oof.loc[X_val.index, (top_k, model_name)] = gs.predict_proba(X_val[features])[:, 1]
                fold_rows.append({
                    "fold": fold,
                    "top_k": top_k,
                    "model": model_name,
                    "n_features": len(features),
                    "best_inner_auc": gs.best_score_,
                    "best_params": json.dumps(gs.best_params_),
                    "top_feature": ranked.index[0] if len(ranked) else "",
                    "top_feature_auc_distance": float(ranked.iloc[0]) if len(ranked) else np.nan,
                })

    rows = []
    flat_oof = pd.DataFrame(index=X.index)
    for top_k, model_name in pred_cols:
        proba = oof[(top_k, model_name)]
        valid = proba.notna()
        if valid.sum() == 0:
            continue
        auc = roc_auc_score(y.loc[valid], proba.loc[valid])
        ci = bootstrap_ci(y.loc[valid], proba.loc[valid], roc_auc_score, n_bootstrap=500, seed=RANDOM_STATE)
        col = f"k{top_k}_{model_name}"
        flat_oof[col] = proba
        rows.append({
            "top_k": top_k,
            "model": model_name,
            "oof_auc": auc,
            "auc_ci_lower": ci["ci_lower"],
            "auc_ci_upper": ci["ci_upper"],
            "n_oof": int(valid.sum()),
        })

    return pd.DataFrame(rows), pd.DataFrame(fold_rows), flat_oof


def make_pipeline_args(args, cutoff, sequences, output_dir, clinical_output_dir):
    return SimpleNamespace(
        sequences=sequences,
        features_path=args.features_path,
        path_r1=args.path_r1,
        path_r2=args.path_r2,
        outcomes_path=args.outcomes_path,
        clinical_data_path=args.clinical_data_path,
        icc_path=args.icc_path or args.path_r1,
        label_col=args.label_col,
        cutoff=cutoff,
        test_size=args.test_size,
        feature_set=args.feature_set,
        exclude_feature_families=args.exclude_feature_families,
        cv_threshold=args.cv_threshold,
        icc_threshold=args.icc_threshold,
        output_dir=str(output_dir),
        clinical_output_dir=str(clinical_output_dir),
    )


def run_one(args, root, cutoff, sequences):
    combo = "_".join(sequences)
    out_dir = root / f"cutoff_{cutoff}" / "combinations" / combo / "dev"
    clinical_out = root / f"cutoff_{cutoff}" / "summary_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    clinical_out.mkdir(parents=True, exist_ok=True)

    start = time.time()
    pipe_args = make_pipeline_args(args, cutoff, sequences, out_dir, clinical_out)
    X_dev, y_dev, X_lock, y_lock = load_dev_lock_clean(pipe_args)
    pd.Series(y_dev, name="y_dev").to_csv(out_dir / "y_dev.csv")
    pd.Series(y_lock, name="y_lock").to_csv(out_dir / "y_lock.csv")
    pd.Series(X_dev.columns, name="feature").to_csv(out_dir / "screen_feature_pool.csv", index=False)

    summary, folds, oof = nested_oof_screen(X_dev, y_dev, args.top_ks, args.models)
    summary.to_csv(out_dir / "nested_oof_signal_screen.csv", index=False)
    folds.to_csv(out_dir / "nested_oof_signal_screen_folds.csv", index=False)
    oof.to_csv(out_dir / "nested_oof_signal_screen_predictions.csv")

    row = {
        "cutoff": cutoff,
        "sequences": combo,
        "n_sequences": len(sequences),
        "n_features_after_cleanup_icc": X_dev.shape[1],
        "n_dev": len(y_dev),
        "n_dev_positive": int(y_dev.sum()),
        "elapsed_sec": round(time.time() - start, 1),
        "output_dir": str(out_dir),
        "status": "success",
    }
    if len(summary):
        best = summary.loc[summary["oof_auc"].idxmax()]
        row.update({
            "best_top_k": int(best["top_k"]),
            "best_model": best["model"],
            "best_oof_auc": round(float(best["oof_auc"]), 4),
            "best_auc_ci_lower": round(float(best["auc_ci_lower"]), 4),
            "best_auc_ci_upper": round(float(best["auc_ci_upper"]), 4),
        })
    return row


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-label", default="small_combos_leq3_nowavelet_nologsigma_aucfilter")
    parser.add_argument("--cutoffs", nargs="+", type=int, default=[3, 2])
    parser.add_argument("--top-ks", nargs="+", type=int, default=[25, 50, 100])
    parser.add_argument("--models", nargs="+", choices=["ridge", "elasticnet", "svm_linear"],
                        default=["ridge", "elasticnet", "svm_linear"])
    parser.add_argument("--sequence-pool", nargs="+", default=DEFAULT_SEQUENCE_POOL)
    parser.add_argument("--max-sequences", type=int, default=3)
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
    summary_csv = summary_dir / "signal_screen_summary.csv"

    print("Fast signal screen will write outside the repo:")
    print(f"  {root}")
    print(f"  cutoffs={args.cutoffs}")
    print(f"  top_ks={args.top_ks}")
    print(f"  models={args.models}")
    print(f"  sequence_pool={args.sequence_pool}")
    print(f"  max_sequences={args.max_sequences}")
    print(f"  exclude_feature_families={args.exclude_feature_families}")
    print(f"  overwrite={args.overwrite}")
    print(f"  n_combos={len(combos)}")
    for combo in combos:
        print(f"  {' '.join(combo)}")
    if args.dry_run:
        return

    rows = []
    total = len(args.cutoffs) * len(combos)
    idx = 0
    for cutoff in args.cutoffs:
        for sequences in combos:
            idx += 1
            print(f"\n[{idx}/{total}] cutoff={cutoff} {' '.join(sequences)}", flush=True)
            try:
                row = run_one(args, root, cutoff, sequences)
            except Exception as exc:
                row = {
                    "cutoff": cutoff,
                    "sequences": "_".join(sequences),
                    "n_sequences": len(sequences),
                    "status": "failed",
                    "error": repr(exc),
                }
            rows.append(row)
            pd.DataFrame(rows).to_csv(summary_csv, index=False)
            print(f"  -> {row}", flush=True)

    df = pd.DataFrame(rows)
    print("\nFAST SIGNAL SCREEN SUMMARY")
    if "best_oof_auc" in df.columns:
        print(df.sort_values("best_oof_auc", ascending=False).head(30).to_string(index=False))
    else:
        print(df.to_string(index=False))
    print(f"\nSummary CSV: {summary_csv}")


if __name__ == "__main__":
    main()
