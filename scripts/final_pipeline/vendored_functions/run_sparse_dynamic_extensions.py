"""Development-only DWI-b2/dynamic extension with ridge, lasso, and elastic net.

Lock outcomes are never loaded. Technical ICC is fit on each outer-training set.
All remaining preprocessing and feature selection is fit inside inner training folds.
Sparse models are eligible only when 10-20 coefficients are nonzero in every
inner fold and after refitting on the applicable outer-training set.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(name, "1")

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import run_compact_ridge_sequence_search as core


SEQUENCE_SETS = {
    "dwi_b2": ("dwi_b2",),
    "dwi_b2+pre": ("dwi_b2", "pre"),
    "dwi_b2+ap": ("dwi_b2", "ap"),
    "dwi_b2+pvp": ("dwi_b2", "pvp"),
    "dwi_b2+tp": ("dwi_b2", "tp"),
    "dwi_b2+pvp+tp": ("dwi_b2", "pvp", "tp"),
    "dwi_b2+pre+pvp+tp": ("dwi_b2", "pre", "pvp", "tp"),
    "dwi_b2+ap+pvp+tp": ("dwi_b2", "ap", "pvp", "tp"),
    "dwi_b2+dynamic_all": ("dwi_b2", "pre", "ap", "pvp", "tp"),
}
FAMILIES = ("ridge", "lasso", "elastic_net")
SPARSE_C_GRID = tuple(float(x) for x in np.geomspace(0.05, 6.0, 17))
EN_RATIOS = (0.25, 0.50, 0.75)
SUPPORT_MIN = 10
SUPPORT_MAX = 20
COEF_EPS = 1e-7


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, default=core.write_json.__globals__["json"].JSONEncoder().default) + "\n")


class AllClusterRepresentatives(core.WithinSequenceClusterTopK):
    """Return one outcome-ranked representative from every strict correlation cluster."""

    def __init__(self, threshold: float = 0.85):
        self.threshold = threshold

    def fit(self, X, y):
        self.k = 1
        super().fit(X, y)
        ordered = sorted(
            self.all_cluster_records_,
            key=lambda item: (-item["score"], item["representative"]),
        )
        self.selected_records_ = ordered
        self.selected_features_ = [item["representative"] for item in ordered]
        return self


def sparse_pipeline(memory=None) -> Pipeline:
    return Pipeline(
        [
            ("qc", core.TrainingQC()),
            ("impute", core.MedianImputerFrame()),
            ("select", AllClusterRepresentatives()),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=1.0,
                    l1_ratio=1.0,
                    solver="saga",
                    max_iter=100000,
                    tol=1e-4,
                    random_state=42,
                ),
            ),
        ],
        memory=memory,
    )


def support(estimator: Pipeline) -> np.ndarray:
    coefficient = np.asarray(estimator.named_steps["model"].coef_).ravel()
    return np.flatnonzero(np.abs(coefficient) > COEF_EPS)


def constrained_auc(estimator: Pipeline, X, y) -> float:
    n = len(support(estimator))
    if n < SUPPORT_MIN or n > SUPPORT_MAX:
        return np.nan
    return float(roc_auc_score(y, estimator.predict_proba(X)[:, 1]))


def support_count(estimator: Pipeline, X, y) -> float:
    return float(len(support(estimator)))


def sparse_grid(family: str) -> dict:
    ratios = (1.0,) if family == "lasso" else EN_RATIOS
    return {"model__C": SPARSE_C_GRID, "model__l1_ratio": ratios}


def refit_sparse_candidate(search: GridSearchCV, X, y):
    frame = pd.DataFrame(search.cv_results_)
    support_columns = [c for c in frame if c.startswith("split") and c.endswith("_test_support")]
    eligible = (
        frame["mean_test_support"].between(SUPPORT_MIN, SUPPORT_MAX)
        & frame[support_columns].min(axis=1).ge(5)
        & frame[support_columns].max(axis=1).le(30)
    )
    frame = frame[eligible & np.isfinite(frame["mean_test_auc"])].sort_values(
        ["mean_test_auc", "param_model__C"], ascending=[False, True]
    )
    if frame.empty:
        raise RuntimeError("No sparse parameter candidate met the inner mean-support constraint")
    for _, row in frame.iterrows():
        params = {
            "model__C": float(row["param_model__C"]),
            "model__l1_ratio": float(row["param_model__l1_ratio"]),
        }
        # Support can shift when an inner-valid candidate is refit on the
        # larger outer-training set. Search locally in C using training data
        # only, prioritizing the value closest to the inner-CV choice.
        center = params["model__C"]
        local = np.unique(np.r_[center, np.geomspace(max(center / 6, 0.002), min(center * 6, 50), 81)])
        local = sorted(local, key=lambda value: abs(np.log(value / center)))
        for candidate_c in local:
            adjusted = dict(params)
            adjusted["model__C"] = float(candidate_c)
            estimator = clone(search.estimator).set_params(**adjusted)
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                estimator.fit(X, y)
            n = len(support(estimator))
            if SUPPORT_MIN <= n <= SUPPORT_MAX:
                adjusted["inner_selected_C"] = center
                adjusted["inner_mean_support"] = float(row["mean_test_support"])
                adjusted["inner_min_support"] = float(row[support_columns].min())
                adjusted["inner_max_support"] = float(row[support_columns].max())
                return estimator, adjusted, float(row["mean_test_auc"]), n
    raise RuntimeError("Inner-valid sparse candidates did not retain 10-20 features after refit")


def sparse_audit(estimator: Pipeline, icc_filter: core.TechnicalICCFilter) -> dict:
    qc = estimator.named_steps["qc"]
    selector = estimator.named_steps["select"]
    model = estimator.named_steps["model"]
    active = support(estimator)
    records = [selector.selected_records_[i] for i in active]
    coefficients = np.asarray(model.coef_).ravel()[active]
    return {
        "n_before_icc": icc_filter.n_before_icc_,
        "n_after_icc": len(icc_filter.columns_),
        "icc_training_n": len(icc_filter.icc_ids_),
        "n_qc": len(qc.columns_),
        "n_clusters": selector.n_clusters_,
        "n_model_features": len(active),
        "selected_features": [item["representative"] for item in records],
        "selected_feature_scores": [item["score"] for item in records],
        "selected_cluster_sizes": [item["cluster_size"] for item in records],
        "selected_sequences": [item["sequence"] for item in records],
        "sequence_counts": dict(Counter(item["sequence"] for item in records)),
        "coefficients": list(coefficients),
        "intercept": list(np.asarray(model.intercept_).ravel()),
        "qc_training_ids": qc.training_ids_,
        "icc_training_ids": icc_filter.icc_ids_,
    }


def fit_one(family, X_train, y_train, inner_splits, readers, memory, jobs):
    icc_filter = core.TechnicalICCFilter(*readers).fit(X_train)
    transformed = icc_filter.transform(X_train)
    if family == "ridge":
        feasible_k = core.feasible_k_grid(transformed, y_train, inner_splits)
        search = GridSearchCV(
            core.make_pipeline(memory),
            {"select__k": feasible_k, "model__C": core.C_GRID},
            scoring="roc_auc",
            cv=inner_splits,
            n_jobs=jobs,
            error_score="raise",
            refit=True,
        )
        with joblib.parallel_backend("threading", n_jobs=jobs):
            search.fit(transformed, y_train)
        estimator = search.best_estimator_
        params = {key: float(value) for key, value in search.best_params_.items()}
        score = float(search.best_score_)
        audit = core.model_audit(estimator, icc_filter)
    else:
        search = GridSearchCV(
            sparse_pipeline(memory),
            sparse_grid(family),
            scoring={"auc": "roc_auc", "support": support_count},
            cv=inner_splits,
            n_jobs=jobs,
            error_score="raise",
            refit=False,
        )
        with warnings.catch_warnings(), joblib.parallel_backend("threading", n_jobs=jobs):
            warnings.filterwarnings("ignore", message="One or more of the test scores are non-finite")
            warnings.simplefilter("error", ConvergenceWarning)
            search.fit(transformed, y_train)
        estimator, params, score, _ = refit_sparse_candidate(search, transformed, y_train)
        audit = sparse_audit(estimator, icc_filter)
    return icc_filter, estimator, params, score, audit, pd.DataFrame(search.cv_results_)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-parent", required=True)
    parser.add_argument("--root", default=str(core.ROOT))
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()

    X, y, dev_ids, lock_ids, split_path, feature_path, readers = core.load_development(Path(args.root))
    run = Path(args.output_parent) / datetime.now().strftime("sparse_dynamic_extensions_%Y%m%d_%H%M%S")
    run.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, run / Path(__file__).name)
    shutil.copy2(Path(__file__).with_name("run_compact_ridge_sequence_search.py"), run / "run_compact_ridge_sequence_search.py")
    memory = joblib.Memory(Path("/private/tmp") / run.name, verbose=0)
    manifest = {
        "started": datetime.now().isoformat(),
        "status": "running",
        "endpoint": "TILs_resection >= 2",
        "n_development": len(dev_ids),
        "n_positive": int(y.sum()),
        "n_negative": int((1-y).sum()),
        "n_lock_excluded": len(lock_ids),
        "lock_evaluated": False,
        "sequence_sets": SEQUENCE_SETS,
        "families": FAMILIES,
        "sparse_support": [SUPPORT_MIN, SUPPORT_MAX],
        "sparse_C_grid": SPARSE_C_GRID,
        "elastic_net_l1_ratios": EN_RATIOS,
        "input_hashes": {str(p): core.digest(Path(p)) for p in [split_path, feature_path, *readers]},
    }
    core.write_json(run / "manifest.json", manifest)
    print(f"RUN_DIRECTORY={run}", flush=True)

    outer_splits = list(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y))
    fold_rows, oof_rows, selected_rows, summary_rows = [], [], [], []
    total = len(SEQUENCE_SETS) * len(FAMILIES)
    task = 0
    for set_index, (set_name, sequences) in enumerate(SEQUENCE_SETS.items(), 1):
        columns = sorted(c for c in X if any(c.startswith(s + "_original_") for s in sequences))
        X_set = X[columns]
        for family in FAMILIES:
            task += 1
            label = f"{set_name}__{family}"
            result_dir = run / label
            result_dir.mkdir()
            predictions = np.full(len(y), np.nan)
            print(f"[{task}/{total}] {set_name} / {family}: {len(columns)} inputs", flush=True)
            for fold, (train_index, validation_index) in enumerate(outer_splits, 1):
                X_train, X_validation = X_set.iloc[train_index], X_set.iloc[validation_index]
                y_train, y_validation = y.iloc[train_index], y.iloc[validation_index]
                inner = list(StratifiedKFold(3, shuffle=True, random_state=420+fold).split(X_train, y_train))
                icc, estimator, params, inner_auc, audit, search_frame = fit_one(
                    family, X_train, y_train, inner, readers, memory, args.jobs
                )
                probability = estimator.predict_proba(icc.transform(X_validation))[:, 1]
                predictions[validation_index] = probability
                metrics = core.classification_metrics(y_validation, probability)
                fold_rows.append({
                    "sequence_set": set_name, "family": family, "outer_fold": fold,
                    "inner_best_auc": inner_auc, "n_train": len(train_index),
                    "n_validation": len(validation_index), "n_model_features": audit["n_model_features"],
                    "C": params.get("model__C"), "l1_ratio": params.get("model__l1_ratio", 0.0),
                    "selected_k": params.get("select__k", audit["n_model_features"]), **metrics,
                })
                for feature, sequence, score, cluster_size, coefficient in zip(
                    audit["selected_features"], audit["selected_sequences"],
                    audit["selected_feature_scores"], audit["selected_cluster_sizes"], audit["coefficients"]
                ):
                    selected_rows.append({
                        "fit": "outer_fold", "sequence_set": set_name, "family": family,
                        "outer_fold": fold, "feature": feature, "sequence": sequence,
                        "univariate_auc_distance": score, "cluster_size": cluster_size,
                        "coefficient": coefficient,
                    })
                oof_rows.extend({
                    "sequence_set": set_name, "family": family, "outer_fold": fold,
                    "PatientID": patient, "y": int(outcome), "prediction": float(prediction),
                } for patient, outcome, prediction in zip(X_validation.index, y_validation, probability))
                search_frame.to_csv(result_dir / f"fold_{fold}_inner_search.csv", index=False)
                core.write_json(result_dir / f"fold_{fold}_model.json", {
                    "params": params, "inner_best_auc": inner_auc,
                    "outer_metrics": metrics, "audit": audit,
                })
                print(f"  fold {fold}: inner={inner_auc:.3f}, n={audit['n_model_features']}, "
                      f"C={params.get('model__C'):.4g}, outer_auc={metrics['auc']:.3f}", flush=True)

            pooled = core.classification_metrics(y, predictions)
            low, high = core.bootstrap_auc(y, predictions, args.bootstrap, 20262000 + task)
            relevant = pd.DataFrame(fold_rows)
            relevant = relevant[(relevant.sequence_set == set_name) & (relevant.family == family)]
            summary_rows.append({
                "sequence_set": set_name, "sequences": "+".join(sequences), "family": family,
                "n_input_features": len(columns), **pooled, "auc_ci_low": low, "auc_ci_high": high,
                "mean_fold_auc": float(relevant.auc.mean()), "sd_fold_auc": float(relevant.auc.std(ddof=1)),
                "mean_selected_features": float(relevant.n_model_features.mean()),
            })

            final_inner = list(StratifiedKFold(3, shuffle=True, random_state=999).split(X_set, y))
            icc, estimator, params, inner_auc, audit, search_frame = fit_one(
                family, X_set, y, final_inner, readers, memory, args.jobs
            )
            joblib.dump({
                "icc_filter": icc, "model_pipeline": estimator, "input_sequences": sequences,
                "family": family, "lock_evaluated": False,
            }, result_dir / "final_development_model.joblib")
            search_frame.to_csv(result_dir / "final_inner_search.csv", index=False)
            core.write_json(result_dir / "final_development_model.json", {
                "params": params, "inner_best_auc": inner_auc, "audit": audit, "lock_evaluated": False,
            })
            for rank, (feature, sequence, score, cluster_size, coefficient) in enumerate(zip(
                audit["selected_features"], audit["selected_sequences"], audit["selected_feature_scores"],
                audit["selected_cluster_sizes"], audit["coefficients"]
            ), 1):
                selected_rows.append({
                    "fit": "final_development", "sequence_set": set_name, "family": family,
                    "outer_fold": 0, "rank": rank, "feature": feature, "sequence": sequence,
                    "univariate_auc_distance": score, "cluster_size": cluster_size,
                    "coefficient": coefficient,
                })

    summary = pd.DataFrame(summary_rows).sort_values("auc", ascending=False)
    pd.DataFrame(fold_rows).to_csv(run / "fold_metrics.csv", index=False)
    pd.DataFrame(oof_rows).to_csv(run / "oof_predictions.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(run / "selected_features_all_fits.csv", index=False)
    pd.DataFrame(selected_rows).query("fit == 'final_development'").to_csv(
        run / "final_development_signatures.csv", index=False
    )
    summary.to_csv(run / "summary.csv", index=False)
    manifest.update({"status": "complete", "finished": datetime.now().isoformat(), "run_directory": str(run)})
    core.write_json(run / "manifest.json", manifest)
    (run / "REPORT.md").write_text(
        "# DWI b2 and dynamic-phase sparse extension\n\n"
        "Development-only nested CV for TILs ≥2; lock outcomes were not evaluated.\n\n"
        + summary.round(3).to_markdown(index=False) + "\n"
    )
    print("\nFINAL SUMMARY", flush=True)
    print(summary.to_string(index=False), flush=True)
    print(f"COMPLETED={run}", flush=True)


if __name__ == "__main__":
    main()
