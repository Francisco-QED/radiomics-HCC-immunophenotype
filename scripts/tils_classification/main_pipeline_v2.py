##### ------------ Radiomics immunophenotype pipeline v2: simplified stability selection -------------- #####
#
# Goals:
#   - Keep the original main_pipeline.py untouched.
#   - Remove supervised univariate AUC/Mann-Whitney/BH prefiltering from feature selection.
#   - Remove per-resample top-k coefficient truncation.
#   - Replace hard sign-stability filtering with descriptive conditional sign probabilities.
#   - Compare ridge, LASSO, and elastic-net logistic workflows by dev-only OOF AUC.
#   - Keep lock-set evaluation as a separate, explicit final stage.

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.svm import SVC

from xgboost import XGBClassifier

from main_pipeline import (
    CLINICAL_FEATURE_PREFIX,
    DEFAULT_L1_RATIOS,
    PARTITION_SEED,
    RANDOM_STATE,
    TARGET_THRESHOLD,
    bootstrap_ci,
    compute_correlation_clusters,
    compute_sensitivity,
    compute_specificity,
    filter_feature_families,
    global_cleanup,
    interobserver_stability,
    load_clinical_features,
    partition_with_constraints,
    preselect_sequences,
    stratified_resample,
)


PRIMARY_COMBOS = [
    ["ap", "pvp", "hbp", "dwi_b1", "adc"],
    ["ap", "pvp", "hbp", "dwi_b1", "adc", "t2"],
    ["ap", "tp", "hbp", "dwi_b1", "adc"],
    ["ap", "pvp", "hbp", "dwi_b2", "adc"],
]

SELECTOR_CS = np.logspace(-3, 1, 7)


def _positive_class_weight(y):
    n_pos = max(int((y == 1).sum()), 1)
    return int((y == 0).sum()) / n_pos


def make_selector(penalty, random_state=RANDOM_STATE, l1_ratios=None, cs=None):
    if l1_ratios is None:
        l1_ratios = DEFAULT_L1_RATIOS
    if cs is None:
        cs = SELECTOR_CS
    if penalty == "ridge":
        return LogisticRegressionCV(
            Cs=cs,
            penalty="l2",
            solver="lbfgs",
            cv=5,
            scoring="roc_auc",
            class_weight="balanced",
            max_iter=5000,
            random_state=random_state,
            n_jobs=1,
        )
    if penalty == "lasso":
        return LogisticRegressionCV(
            Cs=cs,
            penalty="l1",
            solver="liblinear",
            cv=5,
            scoring="roc_auc",
            class_weight="balanced",
            max_iter=2000,
            random_state=random_state,
        )
    if penalty == "elasticnet":
        return LogisticRegressionCV(
            Cs=cs,
            penalty="elasticnet",
            solver="saga",
            l1_ratios=l1_ratios,
            cv=5,
            scoring="roc_auc",
            class_weight="balanced",
            max_iter=7000,
            random_state=random_state,
            n_jobs=1,
        )
    raise ValueError(f"Unknown penalty: {penalty}")


def _fit_selector_on_matrix(X, y, penalty, random_state=RANDOM_STATE, l1_ratios=None, cs=None):
    safe_cv = min(5, int(y.value_counts().min()))
    if safe_cv < 2:
        raise RuntimeError("Need at least two patients in each class for selector CV.")
    selector = make_selector(penalty, random_state=random_state, l1_ratios=l1_ratios, cs=cs)
    selector.cv = StratifiedKFold(n_splits=safe_cv, shuffle=True, random_state=random_state)
    selector.fit(X, y)
    return selector


def fit_penalized_selector(X_tr, y_tr, penalty, random_state=RANDOM_STATE, l1_ratios=None, cs=None):
    imputer = KNNImputer(n_neighbors=5)
    scaler = StandardScaler()
    X_imp = pd.DataFrame(imputer.fit_transform(X_tr), columns=X_tr.columns, index=X_tr.index)
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imp), columns=X_tr.columns, index=X_tr.index)
    selector = _fit_selector_on_matrix(
        X_scaled, y_tr, penalty, random_state=random_state, l1_ratios=l1_ratios, cs=cs
    )
    coefs = pd.Series(selector.coef_[0], index=X_tr.columns)
    return coefs, imputer, scaler, selector


def cluster_representative_features(X, threshold=0.85):
    cluster_of, _orientation = compute_correlation_clusters(X, threshold=threshold)
    roots = []
    seen = set()
    for col in X.columns:
        root = cluster_of[col]
        if root not in seen:
            roots.append(root)
            seen.add(root)
    return roots


def run_cluster_stability_selection(X, y, penalty, n_resamples=200, resample_frac=0.75,
                                    stability_threshold=0.70, cluster_threshold=0.85,
                                    random_state=RANDOM_STATE, l1_ratios=None,
                                    selector_input="clusters", cs=None):
    if penalty == "ridge":
        raise ValueError("Ridge is a predictive baseline, not a sparse stability selector.")

    rng = np.random.default_rng(random_state)
    cluster_of, orientation = compute_correlation_clusters(X, threshold=cluster_threshold)
    cluster_members = {}
    for feat, cluster in cluster_of.items():
        cluster_members.setdefault(cluster, []).append(feat)
    selector_features = sorted(cluster_members) if selector_input == "clusters" else list(X.columns)
    X_selector = X[selector_features]

    cluster_hits = pd.Series(0, index=sorted(cluster_members), dtype=int)
    member_hits = pd.Series(0, index=X.columns, dtype=int)
    sign_records = {cluster: [] for cluster in cluster_members}
    coef_abs_records = {cluster: [] for cluster in cluster_members}
    n_valid = 0

    for _ in tqdm(range(n_resamples), desc=f"{penalty} stability"):
        X_rs, y_rs = stratified_resample(X_selector, y, resample_frac, rng, replace=False)
        try:
            coefs, *_ = fit_penalized_selector(
                X_rs, y_rs, penalty, random_state=random_state, l1_ratios=l1_ratios, cs=cs
            )
        except Exception:
            continue

        nonzero = coefs[coefs != 0]
        n_valid += 1
        hit_clusters = set()
        for feat, coef in nonzero.items():
            cluster = cluster_of[feat]
            member_hits.loc[feat] += 1
            sign_records[cluster].append(np.sign(coef) * orientation[feat])
            coef_abs_records[cluster].append(abs(coef))
            if cluster not in hit_clusters:
                cluster_hits.loc[cluster] += 1
                hit_clusters.add(cluster)

    if n_valid == 0:
        raise RuntimeError(f"Every {penalty} stability resample failed.")

    rows = []
    for cluster, members in cluster_members.items():
        if cluster_hits.loc[cluster] == 0:
            continue
        signs = np.asarray(sign_records[cluster], dtype=float)
        pos_prob = float((signs > 0).mean()) if len(signs) else np.nan
        neg_prob = float((signs < 0).mean()) if len(signs) else np.nan
        representative = member_hits.loc[members].idxmax()
        rows.append({
            "Cluster": cluster,
            "Representative_Feature": representative,
            "Cluster_Members": ";".join(members),
            "N_Members": len(members),
            "Selection_Frequency": float(cluster_hits.loc[cluster] / n_valid),
            "Selection_Count": int(cluster_hits.loc[cluster]),
            "N_Resamples_Valid": int(n_valid),
            "Positive_Sign_Probability": pos_prob,
            "Negative_Sign_Probability": neg_prob,
            "Dominant_Sign_Probability": float(np.nanmax([pos_prob, neg_prob])),
            "Mean_Abs_Coefficient_When_Selected": float(np.mean(coef_abs_records[cluster])),
            "Representative_Member_Selection_Count": int(member_hits.loc[representative]),
        })

    out = pd.DataFrame(rows).set_index("Cluster")
    return out.sort_values(
        ["Selection_Frequency", "Dominant_Sign_Probability", "Mean_Abs_Coefficient_When_Selected"],
        ascending=False,
    )


def choose_signature(stability_df, stability_threshold=0.70, target_feature_count=15):
    stable = stability_df[stability_df["Selection_Frequency"] >= stability_threshold].copy()
    stable = stable.head(target_feature_count)
    if stable.empty:
        stable = stability_df.head(target_feature_count).copy()
        stable["Exploratory_Top15_Fallback"] = True
    else:
        stable["Exploratory_Top15_Fallback"] = False
    return stable["Representative_Feature"].tolist(), stable


def _ridge_pipeline(random_state=RANDOM_STATE, cs=None):
    if cs is None:
        cs = SELECTOR_CS
    return Pipeline([
        ("imputer", KNNImputer(n_neighbors=5)),
        ("scaler", StandardScaler()),
        ("model", LogisticRegressionCV(
            Cs=cs,
            penalty="l2",
            solver="lbfgs",
            cv=5,
            scoring="roc_auc",
            class_weight="balanced",
            max_iter=5000,
            random_state=random_state,
            n_jobs=1,
        )),
    ])


def _tune_comparator(model_name, X, y, random_state=RANDOM_STATE, cv=5, n_jobs=4):
    pos_weight = _positive_class_weight(y)
    models = {
        "logistic": LogisticRegression(class_weight="balanced", max_iter=5000, random_state=random_state),
        "svm_linear": SVC(kernel="linear", probability=True, class_weight="balanced", random_state=random_state),
        "svm_rbf": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=random_state),
        "rf": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=random_state),
        "extratrees": ExtraTreesClassifier(n_estimators=300, class_weight="balanced", random_state=random_state),
        "xgboost": XGBClassifier(eval_metric="logloss", scale_pos_weight=pos_weight,
                                  random_state=random_state, n_jobs=1),
    }
    grids = {
        "logistic": {"model__C": [0.001, 0.01, 0.1, 1, 10]},
        "svm_linear": {"model__C": [0.001, 0.01, 0.1, 1, 10]},
        "svm_rbf": {"model__C": [1, 10, 100], "model__gamma": ["scale", 0.01]},
        "rf": {"model__max_depth": [2, 3, 4, 5], "model__min_samples_leaf": [1, 2, 5, 10]},
        "extratrees": {"model__max_depth": [2, 3, 4, 5], "model__min_samples_leaf": [1, 2, 5, 10]},
        "xgboost": {"model__n_estimators": [50, 100, 200], "model__max_depth": [1, 2, 3],
                    "model__learning_rate": [0.01, 0.05, 0.1], "model__subsample": [0.6, 0.8, 1.0]},
    }
    pipe = Pipeline([
        ("imputer", KNNImputer(n_neighbors=5)),
        ("scaler", StandardScaler()),
        ("model", models[model_name]),
    ])
    safe_cv = min(cv, int(y.value_counts().min()))
    splitter = StratifiedKFold(n_splits=safe_cv, shuffle=True, random_state=random_state)
    gs = GridSearchCV(pipe, grids[model_name], scoring="roc_auc", cv=splitter, n_jobs=n_jobs, refit=True)
    gs.fit(X, y)
    return gs.best_estimator_, gs.best_params_, gs.best_score_


def nested_oof_auc(X, y, workflow, stability_args, model_names=None, random_state=RANDOM_STATE):
    if model_names is None:
        model_names = ["signature_logistic"]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    oof = pd.DataFrame(np.nan, index=X.index, columns=model_names)
    fold_rows = []
    global_selector_features = None
    if workflow != "ridge":
        if stability_args["selector_input"] == "clusters":
            global_selector_features = cluster_representative_features(
                X, threshold=stability_args["cluster_threshold"]
            )
        else:
            global_selector_features = list(X.columns)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_val = X.iloc[val_idx]

        if workflow == "ridge":
            model = _ridge_pipeline(random_state=random_state, cs=stability_args["cs"])
            model.fit(X_tr, y_tr)
            oof.loc[X_val.index, "ridge"] = model.predict_proba(X_val)[:, 1]
            fold_rows.append({"fold": fold, "workflow": workflow, "n_features": X.shape[1]})
            continue

        coefs, imputer, scaler, _selector = fit_penalized_selector(
            X_tr[global_selector_features], y_tr, workflow, random_state=random_state,
            l1_ratios=stability_args["l1_ratios"], cs=stability_args["cs"]
        )
        nonzero = coefs[coefs != 0]
        if len(nonzero) == 0:
            continue
        features = nonzero.abs().sort_values(ascending=False).head(
            stability_args["target_feature_count"]
        ).index.tolist()
        if not features:
            continue

        for model_name in model_names:
            if model_name == "signature_logistic":
                estimator, best_params, best_score = _tune_comparator("logistic", X_tr[features], y_tr,
                                                                       random_state=random_state)
            else:
                estimator, best_params, best_score = _tune_comparator(model_name, X_tr[features], y_tr,
                                                                       random_state=random_state)
            oof.loc[X_val.index, model_name] = estimator.predict_proba(X_val[features])[:, 1]
            fold_rows.append({
                "fold": fold,
                "workflow": workflow,
                "model": model_name,
                "n_features": len(features),
                "best_inner_auc": best_score,
                "best_params": json.dumps(best_params),
                "oof_selection_mode": "single_fold_penalized_selector_top_k",
            })

    rows = []
    for model_name in oof.columns:
        valid = oof[model_name].notna()
        if valid.sum() == 0:
            continue
        y_valid = y[valid]
        proba = oof.loc[valid, model_name]
        auc = roc_auc_score(y_valid, proba)
        ci = bootstrap_ci(y_valid, proba, roc_auc_score, n_bootstrap=1000, seed=random_state)
        fpr, tpr, thresholds = roc_curve(y_valid, proba)
        rows.append({
            "workflow": workflow,
            "model": model_name,
            "oof_auc": auc,
            "auc_ci_lower": ci["ci_lower"],
            "auc_ci_upper": ci["ci_upper"],
            "youden_threshold": thresholds[np.argmax(tpr - fpr)],
            "n_oof": int(valid.sum()),
        })
    return pd.DataFrame(rows), pd.DataFrame(fold_rows), oof


def load_dev_lock_clean(args):
    df_all_seq = pd.read_csv(args.features_path, index_col=0)
    df_filtered = preselect_sequences(df_all_seq, allowed_sequences=args.sequences)
    df_filtered = filter_feature_families(df_filtered, exclude_families=args.exclude_feature_families)

    if args.feature_set == "both":
        df_clinical_feats = load_clinical_features(args.clinical_data_path)
        df_filtered = df_filtered.join(df_clinical_feats, how="left")

    X_dev, y_dev, X_lock, y_lock, df_clinical_dev, df_clinical_lock = partition_with_constraints(
        df_filtered,
        args.outcomes_path,
        args.icc_path,
        clinical_data_path=args.clinical_data_path,
        label_col=args.label_col,
        cutoff=args.cutoff,
        test_size=args.test_size,
        output_dir=args.output_dir,
        clinical_output_dir=args.clinical_output_dir,
    )

    X_dev_clean = global_cleanup(X_dev, id_col="PatientID", cv_threshold=args.cv_threshold)
    clinical_cols = [c for c in X_dev_clean.columns if c.startswith(CLINICAL_FEATURE_PREFIX)]
    radiomics_cols = [c for c in X_dev_clean.columns if c not in clinical_cols]
    X_dev_icc = interobserver_stability(
        X_dev_clean[radiomics_cols], args.path_r1, args.path_r2,
        id_col="PatientID", threshold=args.icc_threshold,
    )
    if clinical_cols:
        X_dev_icc = pd.concat([X_dev_icc, X_dev_clean[clinical_cols]], axis=1)

    y_dev = y_dev.loc[X_dev_icc.index]
    X_lock = X_lock[X_dev_icc.columns].replace([np.inf, -np.inf], np.nan)
    return X_dev_icc, y_dev, X_lock, y_lock


def run_dev_stage(args):
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    X_dev, y_dev, X_lock, y_lock = load_dev_lock_clean(args)
    pd.Series(y_dev, name="y_dev").to_csv(Path(args.output_dir) / "y_dev.csv")
    pd.Series(y_lock, name="y_lock").to_csv(Path(args.output_dir) / "y_lock.csv")
    X_dev.to_csv(Path(args.output_dir) / "X_dev_clean_icc.csv")
    X_lock.to_csv(Path(args.output_dir) / "X_lock_clean_icc.csv")

    stability_args = {
        "n_resamples": args.stability_n_resamples,
        "resample_frac": args.stability_resample_frac,
        "stability_threshold": args.stability_threshold,
        "target_feature_count": args.target_feature_count,
        "cluster_threshold": args.cluster_threshold,
        "l1_ratios": args.l1_ratios,
        "selector_input": args.selector_input,
        "cs": np.logspace(-3, 1, args.selector_c_grid_size),
    }

    summaries = []
    for workflow in args.workflows:
        if workflow == "ridge":
            summary, folds, oof = nested_oof_auc(X_dev, y_dev, "ridge", stability_args,
                                                 model_names=["ridge"], random_state=RANDOM_STATE)
        else:
            stability_df = run_cluster_stability_selection(
                X_dev, y_dev, workflow,
                n_resamples=args.full_stability_n_resamples,
                resample_frac=args.stability_resample_frac,
                stability_threshold=args.stability_threshold,
                cluster_threshold=args.cluster_threshold,
                random_state=RANDOM_STATE,
                l1_ratios=args.l1_ratios,
                selector_input=args.selector_input,
                cs=stability_args["cs"],
            )
            stability_df.to_csv(Path(args.output_dir) / f"stability_{workflow}_clusters.csv")
            features, signature = choose_signature(
                stability_df,
                stability_threshold=args.stability_threshold,
                target_feature_count=args.target_feature_count,
            )
            signature.to_csv(Path(args.output_dir) / f"signature_{workflow}_clusters.csv")
            pd.Series(features, name="feature").to_csv(
                Path(args.output_dir) / f"signature_{workflow}_features.csv", index=False
            )
            summary, folds, oof = nested_oof_auc(
                X_dev, y_dev, workflow, stability_args,
                model_names=args.oof_models,
                random_state=RANDOM_STATE,
            )
        summary.to_csv(Path(args.output_dir) / f"nested_oof_{workflow}.csv", index=False)
        folds.to_csv(Path(args.output_dir) / f"nested_oof_{workflow}_folds.csv", index=False)
        oof.to_csv(Path(args.output_dir) / f"nested_oof_{workflow}_predictions.csv")
        summaries.append(summary)

    all_summary = pd.concat(summaries, ignore_index=True)
    all_summary.to_csv(Path(args.output_dir) / "nested_oof_summary.csv", index=False)
    print(all_summary.sort_values("oof_auc", ascending=False).to_string(index=False))


def run_lock_stage(args):
    out_dir = Path(args.output_dir)
    in_dir = Path(args.input_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X_dev = pd.read_csv(in_dir / "X_dev_clean_icc.csv", index_col=0)
    X_lock = pd.read_csv(in_dir / "X_lock_clean_icc.csv", index_col=0)
    y_dev = pd.read_csv(in_dir / "y_dev.csv", index_col=0).squeeze()
    y_lock = pd.read_csv(in_dir / "y_lock.csv", index_col=0).squeeze()
    features = pd.read_csv(in_dir / f"signature_{args.primary_workflow}_features.csv")["feature"].tolist()

    rows = []
    models = ["logistic"] + args.comparator_models
    for model_name in models:
        estimator, params, inner_auc = _tune_comparator(model_name, X_dev[features], y_dev,
                                                        random_state=RANDOM_STATE, n_jobs=args.n_jobs)
        proba = estimator.predict_proba(X_lock[features])[:, 1]
        auc = roc_auc_score(y_lock, proba)
        auc_ci = bootstrap_ci(y_lock, proba, roc_auc_score, n_bootstrap=args.bootstrap, seed=RANDOM_STATE)
        sens = compute_sensitivity(y_lock, proba, threshold=args.threshold)
        spec = compute_specificity(y_lock, proba, threshold=args.threshold)
        rows.append({
            "Model": model_name,
            "Primary_Workflow": args.primary_workflow,
            "N_Features": len(features),
            "AUC": auc,
            "AUC 95% CI": f"({auc_ci['ci_lower']:.3f}, {auc_ci['ci_upper']:.3f})",
            "Sensitivity": sens,
            "Specificity": spec,
            "Threshold": args.threshold,
            "Inner_CV_AUC": inner_auc,
            "Best_Params": json.dumps(params),
        })
        with open(out_dir / f"lock_model_{model_name}.pkl", "wb") as f:
            pickle.dump(estimator, f)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "lock_model_comparison.csv", index=False)
    print(df.to_string(index=False))


def parse_arguments():
    parser = argparse.ArgumentParser(description="Radiomics immunophenotype v2 pipeline")
    parser.add_argument("--stage", choices=["dev", "lock"], default="dev")
    parser.add_argument("--sequences", nargs="+", default=["ap", "pvp", "hbp", "dwi_b1", "adc"])
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
    parser.add_argument("--cutoff", type=float, default=3)
    parser.add_argument("--test-size", type=int, default=30)
    parser.add_argument("--feature-set", choices=["radiomics", "both"], default="radiomics")
    parser.add_argument("--exclude-feature-families", nargs="*", default=["wavelet"])
    parser.add_argument("--cv-threshold", type=float, default=0.1)
    parser.add_argument("--icc-threshold", type=float, default=0.85)
    parser.add_argument("--cluster-threshold", type=float, default=0.85)
    parser.add_argument("--stability-threshold", type=float, default=0.70)
    parser.add_argument("--stability-resample-frac", type=float, default=0.75)
    parser.add_argument("--stability-n-resamples", type=int, default=50,
                        help="Nested-CV stability resamples per outer fold.")
    parser.add_argument("--full-stability-n-resamples", type=int, default=200,
                        help="Full-development stability resamples used for final signature tables.")
    parser.add_argument("--target-feature-count", type=int, default=15)
    parser.add_argument("--l1-ratios", nargs="+", type=float, default=DEFAULT_L1_RATIOS)
    parser.add_argument("--selector-input", choices=["clusters", "all"], default="clusters",
                        help="'clusters' uses one unsupervised representative per correlation cluster; "
                             "'all' lets the sparse selector see every ICC-surviving feature.")
    parser.add_argument("--selector-c-grid-size", type=int, default=7,
                        help="Number of log-spaced C values from 1e-3 to 1e1 for selector CV.")
    parser.add_argument("--workflows", nargs="+", choices=["ridge", "lasso", "elasticnet"],
                        default=["ridge", "lasso", "elasticnet"])
    parser.add_argument("--oof-models", nargs="+", default=["signature_logistic"],
                        help="OOF models for selected signatures; keep small for method selection.")
    parser.add_argument("--comparator-models", nargs="+",
                        default=["svm_linear", "svm_rbf", "rf", "extratrees", "xgboost"])
    parser.add_argument("--primary-workflow", choices=["lasso", "elasticnet"], default="elasticnet")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--threshold", type=float, default=TARGET_THRESHOLD)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--clinical-output-dir", type=str, default=None)
    parser.add_argument("--input-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    if args.icc_path is None:
        args.icc_path = args.path_r1

    cutoff_val = int(args.cutoff) if args.cutoff == int(args.cutoff) else args.cutoff
    neg_max = int(args.cutoff) - 1
    cutoff_label = f"tils_{cutoff_val}_vs_0-{neg_max}"
    feature_set_label = "clinical+radiomics" if args.feature_set == "both" else "radiomics"
    base = (
        Path.home() / "Library" / "CloudStorage" / "OneDrive-TheMountSinaiHospital"
        / "Regeneron retrospective radiomics" / "Results" / "main_pipeline"
        / "immunophenotype" / cutoff_label / feature_set_label / "v2_stability"
    )
    combo = "_".join(args.sequences)
    if args.output_dir is None:
        args.output_dir = str(base / "combinations" / combo / ("dev" if args.stage == "dev" else "lock"))
    if args.input_dir is None:
        args.input_dir = str(base / "combinations" / combo / "dev")
    if args.clinical_output_dir is None:
        args.clinical_output_dir = str(base / "summary_results")
    return args


if __name__ == "__main__":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    parsed = parse_arguments()
    print(f"V2 stage={parsed.stage} output={parsed.output_dir}")
    if parsed.stage == "dev":
        run_dev_stage(parsed)
    else:
        run_lock_stage(parsed)
