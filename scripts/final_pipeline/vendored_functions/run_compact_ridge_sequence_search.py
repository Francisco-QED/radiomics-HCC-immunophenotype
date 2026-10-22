"""Development-only targeted sequence search with compact ridge signatures.

The script never evaluates lock outcomes. Every outer-fold model refits QC, ICC,
imputation, within-sequence clustering, supervised representative selection,
top-k selection, scaling, and ridge logistic regression using training data only.
Each fitted signature contains exactly 10, 15, or 20 named features.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import warnings
from collections import Counter
from datetime import datetime
from functools import lru_cache
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_name, "1")

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import rankdata
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(
    "/Users/Francisco/Library/CloudStorage/OneDrive-TheMountSinaiHospital/"
    "Regeneron retrospective radiomics"
)
ALL_SEQUENCES = ("pre", "ap", "pvp", "tp", "hbp", "t2", "dwi_b1", "dwi_b2", "adc")
SEQUENCE_SETS = {
    "hbp": ("hbp",),
    "dwi_b1": ("dwi_b1",),
    "dwi_b2": ("dwi_b2",),
    "adc": ("adc",),
    "dwi_b1+dwi_b2": ("dwi_b1", "dwi_b2"),
    "dwi_b1+adc": ("dwi_b1", "adc"),
    "dwi_b2+adc": ("dwi_b2", "adc"),
    "diffusion_all": ("dwi_b1", "dwi_b2", "adc"),
    "hbp+adc": ("hbp", "adc"),
    "hbp+dwi_b1": ("hbp", "dwi_b1"),
    "hbp+dwi_b2": ("hbp", "dwi_b2"),
    "hbp+dwi_b1+dwi_b2": ("hbp", "dwi_b1", "dwi_b2"),
    "hbp+diffusion_all": ("hbp", "dwi_b1", "dwi_b2", "adc"),
    "hbp+t2": ("hbp", "t2"),
    "dynamic": ("pre", "ap", "pvp", "tp"),
    "all9": ALL_SEQUENCES,
}
K_GRID = (10, 15, 20)
C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(
            value,
            indent=2,
            default=lambda item: item.item() if isinstance(item, np.generic) else str(item),
        )
        + "\n"
    )


def feature_sequence(name: str) -> str:
    for sequence in sorted(ALL_SEQUENCES, key=len, reverse=True):
        if name.startswith(sequence + "_"):
            return sequence
    raise ValueError(f"Cannot infer sequence for feature: {name}")


def auc_distance_scores(array: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Absolute univariate AUC distance from 0.5, with average ranks for ties."""
    y = np.asarray(y, dtype=int)
    n1 = int((y == 1).sum())
    n0 = int((y == 0).sum())
    if not n1 or not n0:
        raise ValueError("Both outcome classes are required")
    ranks = rankdata(np.asarray(array), axis=0)
    auc = (ranks[y == 1].sum(axis=0) - n1 * (n1 + 1) / 2) / (n1 * n0)
    return np.abs(auc - 0.5)


@lru_cache(maxsize=4)
def read_reader(path: str) -> pd.DataFrame:
    return pd.read_csv(
        path,
        index_col=0,
        usecols=lambda col: col == "PatientID" or "_original_" in col,
    )


class TechnicalICCFilter(BaseEstimator, TransformerMixin):
    """Estimate reader reliability using only the current outer-training IDs."""

    def __init__(
        self,
        reader1: str,
        reader2: str,
        icc_threshold: float = 0.85,
        min_readers: int = 4,
    ):
        self.reader1 = reader1
        self.reader2 = reader2
        self.icc_threshold = icc_threshold
        self.min_readers = min_readers

    def fit(self, X: pd.DataFrame, y=None):
        reader1 = read_reader(self.reader1)
        reader2 = read_reader(self.reader2)
        ids = sorted(set(X.index) & set(reader1.index) & set(reader2.index))
        if len(ids) < self.min_readers:
            raise ValueError(f"Only {len(ids)} double-reader training patients")
        columns = [col for col in X.columns if col in reader1.columns and col in reader2.columns]
        a = reader1.loc[ids, columns].to_numpy(float)
        b = reader2.loc[ids, columns].to_numpy(float)
        valid = np.isfinite(a).all(axis=0) & np.isfinite(b).all(axis=0)
        a = a[:, valid]
        b = b[:, valid]
        columns = np.asarray(columns)[valid]
        n = len(ids)
        grand = (a.mean(axis=0) + b.mean(axis=0)) / 2
        subject_mean = (a + b) / 2
        ss_between = 2 * ((subject_mean - grand) ** 2).sum(axis=0)
        ss_reader = n * ((a.mean(axis=0) - grand) ** 2 + (b.mean(axis=0) - grand) ** 2)
        ss_total = ((a - grand) ** 2).sum(axis=0) + ((b - grand) ** 2).sum(axis=0)
        ms_between = ss_between / (n - 1)
        ms_error = (ss_total - ss_between - ss_reader) / (n - 1)
        icc = np.divide(
            ms_between - ms_error,
            ms_between + ms_error,
            out=np.zeros_like(ms_between),
            where=(ms_between + ms_error) > 0,
        )
        self.columns_ = list(columns[icc >= self.icc_threshold])
        if len(self.columns_) < min(K_GRID):
            raise ValueError(f"QC retained only {len(self.columns_)} features")
        self.icc_ = dict(zip(columns, icc))
        self.training_ids_ = list(X.index)
        self.icc_ids_ = ids
        self.n_before_icc_ = len(columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.loc[:, self.columns_].replace([np.inf, -np.inf], np.nan)


class TrainingQC(BaseEstimator, TransformerMixin):
    """Fit distribution-based filters on the current inner-training data."""

    def __init__(self, min_observed: float = 0.80, cv_threshold: float = 0.10):
        self.min_observed = min_observed
        self.cv_threshold = cv_threshold

    def fit(self, X: pd.DataFrame, y=None):
        data = X.replace([np.inf, -np.inf], np.nan)
        std = data.std()
        mean = data.mean().abs()
        keep = (
            (data.notna().mean() >= self.min_observed)
            & (std > 1e-8)
            & ((std / mean.replace(0, np.nan) > self.cv_threshold) | mean.eq(0))
        )
        self.columns_ = list(data.columns[keep])
        if len(self.columns_) < min(K_GRID):
            raise ValueError(f"Distribution QC retained only {len(self.columns_)} features")
        self.training_ids_ = list(X.index)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.loc[:, self.columns_].replace([np.inf, -np.inf], np.nan)


class MedianImputerFrame(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y=None):
        self.columns_ = list(X.columns)
        self.medians_ = X.median(axis=0)
        if self.medians_.isna().any():
            raise ValueError("At least one retained feature has no finite training median")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.loc[:, self.columns_].fillna(self.medians_)


class WithinSequenceClusterTopK(BaseEstimator, TransformerMixin):
    """Complete-linkage clusters within sequence; outcome-ranked representative and top k."""

    def __init__(self, k: int = 10, threshold: float = 0.85):
        self.k = k
        self.threshold = threshold

    def fit(self, X: pd.DataFrame, y):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("WithinSequenceClusterTopK requires a pandas DataFrame")
        y_array = np.asarray(y, dtype=int)
        records = []
        cluster_number = 0
        minima = []
        for sequence in ALL_SEQUENCES:
            columns = sorted(col for col in X.columns if feature_sequence(col) == sequence)
            if not columns:
                continue
            values = X[columns].to_numpy(float)
            scores = auc_distance_scores(values, y_array)
            if len(columns) == 1:
                labels = np.array([1])
                corr = np.ones((1, 1))
            else:
                corr = pd.DataFrame(values).corr(method="spearman").fillna(0).to_numpy()
                corr = np.clip((corr + corr.T) / 2, -1, 1)
                np.fill_diagonal(corr, 1)
                distance = np.clip(1 - np.abs(corr), 0, 1)
                np.fill_diagonal(distance, 0)
                labels = fcluster(
                    linkage(squareform(distance, checks=True), method="complete"),
                    t=1 - self.threshold,
                    criterion="distance",
                )
            for local_label in np.unique(labels):
                cluster_number += 1
                indices = np.flatnonzero(labels == local_label)
                best_score = scores[indices].max()
                tied = [i for i in indices if np.isclose(scores[i], best_score, rtol=0, atol=1e-15)]
                representative_index = min(tied, key=lambda i: columns[i])
                if len(indices) > 1:
                    local = np.abs(corr[np.ix_(indices, indices)])
                    minima.append(float(local[np.triu_indices_from(local, k=1)].min()))
                else:
                    minima.append(1.0)
                records.append(
                    {
                        "cluster": cluster_number,
                        "sequence": sequence,
                        "cluster_size": int(len(indices)),
                        "representative": columns[representative_index],
                        "score": float(scores[representative_index]),
                        "members": [columns[i] for i in indices],
                    }
                )
        ordered = sorted(records, key=lambda item: (-item["score"], item["representative"]))
        if len(ordered) < int(self.k):
            raise ValueError(f"Only {len(ordered)} clusters available for k={self.k}")
        self.selected_records_ = ordered[: int(self.k)]
        self.selected_features_ = [item["representative"] for item in self.selected_records_]
        self.all_cluster_records_ = records
        self.n_clusters_ = len(records)
        self.min_cluster_abs_rho_ = float(min(minima)) if minima else 1.0
        if self.min_cluster_abs_rho_ < self.threshold - 1e-10:
            raise AssertionError("Complete-linkage cluster violated pairwise threshold")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.loc[:, self.selected_features_]


class SafeRidgeClassifier(ClassifierMixin, BaseEstimator):
    _estimator_type = "classifier"

    def __init__(self, C: float = 1.0):
        self.C = C

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        if not np.any(np.std(np.asarray(X), axis=0) > 1e-12):
            self.model_ = DummyClassifier(strategy="prior").fit(X, y)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                self.model_ = LogisticRegression(
                    l1_ratio=0,
                    C=self.C,
                    solver="lbfgs",
                    max_iter=20000,
                    tol=1e-5,
                    random_state=42,
                ).fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def predict(self, X):
        return self.model_.predict(X)

    def decision_function(self, X):
        if hasattr(self.model_, "decision_function"):
            return self.model_.decision_function(X)
        probability = np.clip(self.predict_proba(X)[:, 1], 1e-12, 1 - 1e-12)
        return np.log(probability / (1 - probability))


def make_pipeline(memory):
    return Pipeline(
        [
            ("qc", TrainingQC()),
            ("impute", MedianImputerFrame()),
            ("select", WithinSequenceClusterTopK()),
            ("scale", StandardScaler()),
            ("model", SafeRidgeClassifier()),
        ],
        memory=memory,
    )


def feasible_k_grid(X: pd.DataFrame, y: pd.Series, splits) -> tuple[int, ...]:
    """Return feature counts supported by every inner-training split."""
    minimum_clusters = np.inf
    for train_index, _ in splits:
        X_train = X.iloc[train_index]
        y_train = y.iloc[train_index]
        qc = TrainingQC().fit(X_train)
        imputed = MedianImputerFrame().fit_transform(qc.transform(X_train))
        selector = WithinSequenceClusterTopK(k=1).fit(imputed, y_train)
        minimum_clusters = min(minimum_clusters, selector.n_clusters_)
    feasible = tuple(k for k in K_GRID if k <= minimum_clusters)
    if not feasible:
        raise ValueError(f"Fewer than {min(K_GRID)} clusters in at least one inner-training split")
    return feasible


def model_audit(pipeline: Pipeline, icc_filter: TechnicalICCFilter) -> dict:
    qc = pipeline.named_steps["qc"]
    selector = pipeline.named_steps["select"]
    estimator = pipeline.named_steps["model"].model_
    coefficients = np.asarray(getattr(estimator, "coef_", np.zeros((1, len(selector.selected_features_))))).ravel()
    return {
        "n_qc": len(qc.columns_),
        "n_before_icc": icc_filter.n_before_icc_,
        "n_after_icc": len(icc_filter.columns_),
        "icc_training_n": len(icc_filter.icc_ids_),
        "n_clusters": selector.n_clusters_,
        "n_model_features": len(selector.selected_features_),
        "min_cluster_abs_rho": selector.min_cluster_abs_rho_,
        "selected_features": selector.selected_features_,
        "selected_feature_scores": [item["score"] for item in selector.selected_records_],
        "selected_cluster_sizes": [item["cluster_size"] for item in selector.selected_records_],
        "selected_sequences": [item["sequence"] for item in selector.selected_records_],
        "sequence_counts": dict(Counter(item["sequence"] for item in selector.selected_records_)),
        "coefficients": list(coefficients),
        "intercept": getattr(estimator, "intercept_", None),
        "qc_training_ids": qc.training_ids_,
        "icc_training_ids": icc_filter.icc_ids_,
    }


def classification_metrics(y, probability):
    return {
        "auc": float(roc_auc_score(y, probability)),
        "average_precision": float(average_precision_score(y, probability)),
        "brier": float(brier_score_loss(y, probability)),
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
        "prevalence": float(np.mean(y)),
    }


def bootstrap_auc(y, probability, n_bootstrap=2000, seed=20261014):
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=int)
    probability = np.asarray(probability, dtype=float)
    negative = np.flatnonzero(y == 0)
    positive = np.flatnonzero(y == 1)
    values = []
    for _ in range(n_bootstrap):
        sample = np.r_[
            rng.choice(negative, size=len(negative), replace=True),
            rng.choice(positive, size=len(positive), replace=True),
        ]
        values.append(roc_auc_score(y[sample], probability[sample]))
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def load_development(root: Path):
    old = root / "Results/main_pipeline_claude/main_pipeline/final/cohort_v3_141"
    split_path = old / "split/split.csv"
    feature_path = root / "Results/Radiomics_sheets/SA1_2_pre/R-SA1_2_I-sitkBSpline_S-1mm_W-perseq.csv"
    readers = [
        str(
            root
            / f"Results/Radiomics_sheets/inter-reader/Reader{i}/"
            f"R-Reader{i}_I-sitkBSpline_S-1mm_W-perseq.csv"
        )
        for i in (1, 2)
    ]
    assignment = pd.read_csv(split_path, usecols=["PatientID", "set"])
    if assignment.PatientID.duplicated().any():
        raise ValueError("Duplicate split IDs")
    dev_ids = assignment.loc[assignment["set"] == "dev", "PatientID"].tolist()
    lock_ids = set(assignment.loc[assignment["set"] == "lock", "PatientID"])
    if len(dev_ids) != 111 or len(lock_ids) != 30 or set(dev_ids) & lock_ids:
        raise ValueError("Unexpected development/lock assignment")
    positions = np.flatnonzero(assignment["set"].to_numpy() == "dev")
    keep_rows = set(positions)
    labels = (
        pd.read_csv(
            split_path,
            skiprows=lambda row: row > 0 and row - 1 not in keep_rows,
        )
        .set_index("PatientID")["TILs_resection"]
        .loc[dev_ids]
    )
    feature_ids = pd.read_csv(feature_path, usecols=["PatientID"])["PatientID"]
    dev_rows = set(np.flatnonzero(feature_ids.isin(dev_ids)))
    features = pd.read_csv(
        feature_path,
        index_col=0,
        skiprows=lambda row: row > 0 and row - 1 not in dev_rows,
        usecols=lambda col: col == "PatientID" or "_original_" in col,
    ).loc[dev_ids]
    features = features.reindex(sorted(features.columns), axis=1).astype(float)
    if features.shape != (111, 963):
        raise ValueError(f"Unexpected development feature matrix: {features.shape}")
    if labels.isna().any():
        raise ValueError("Missing development outcome")
    return features, (labels >= 2).astype(int), dev_ids, lock_ids, split_path, feature_path, readers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-parent", required=True)
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()

    root = Path(args.root)
    X, y, dev_ids, lock_ids, split_path, feature_path, readers = load_development(root)
    run = Path(args.output_parent) / datetime.now().strftime("compact_ridge_sequences_%Y%m%d_%H%M%S")
    run.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, run / "run_compact_ridge_sequence_search.py")
    cache = Path("/private/tmp") / run.name
    memory = joblib.Memory(cache, verbose=0)
    manifest = {
        "started": datetime.now().isoformat(),
        "status": "running",
        "endpoint": "TILs_resection >= 2",
        "n_development": len(dev_ids),
        "n_positive": int(y.sum()),
        "n_negative": int((1 - y).sum()),
        "n_lock_excluded": len(lock_ids),
        "lock_evaluated": False,
        "feature_source": str(feature_path),
        "sequence_sets": SEQUENCE_SETS,
        "feature_count_grid": K_GRID,
        "ridge_C_grid": C_GRID,
        "outer_folds": 5,
        "inner_folds": 3,
        "seed": 42,
        "qc": (
            "ICC(3,1)>=0.85 estimated on each outer-training paired-reader cohort; "
            ">=80% observed, SD>1e-8, and CV>0.10 refit in every inner-training fold; "
            "missing values median-imputed from that training fold"
        ),
        "clustering": "within-sequence complete linkage on 1-|Spearman rho|; all pairs |rho|>=0.85",
        "representative": "highest absolute univariate AUC distance within cluster, training-only",
        "ranking": "global absolute univariate AUC distance among cluster representatives, training-only",
        "model": "L2 logistic regression; no class weights",
        "versions": {
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
        },
        "input_hashes": {
            str(path): digest(Path(path)) for path in [split_path, feature_path, *readers]
        },
        "script_hash": digest(Path(__file__)),
    }
    write_json(run / "manifest.json", manifest)
    pd.Series(dev_ids, name="PatientID").to_csv(run / "dev_patient_ids.csv", index=False)
    print(f"RUN_DIRECTORY={run}", flush=True)
    print(f"Development: {X.shape}; positives={int(y.sum())}; lock excluded={len(lock_ids)}", flush=True)

    outer_splits = list(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y))
    oof_rows = []
    fold_rows = []
    selected_rows = []
    split_rows = []
    final_rows = []
    summary_rows = []

    baseline_prediction = np.full(len(y), np.nan)
    for fold, (train_index, validation_index) in enumerate(outer_splits, 1):
        baseline_prediction[validation_index] = float(y.iloc[train_index].mean())
        split_rows.extend(
            {
                "outer_fold": fold,
                "role": "outer_train",
                "PatientID": patient,
            }
            for patient in X.index[train_index]
        )
        split_rows.extend(
            {
                "outer_fold": fold,
                "role": "outer_validation",
                "PatientID": patient,
            }
            for patient in X.index[validation_index]
        )

    for set_index, (set_name, sequences) in enumerate(SEQUENCE_SETS.items(), 1):
        columns = sorted(
            col for col in X.columns if any(col.startswith(sequence + "_original_") for sequence in sequences)
        )
        X_set = X[columns]
        predictions = np.full(len(y), np.nan)
        set_dir = run / set_name
        set_dir.mkdir()
        print(f"[{set_index}/{len(SEQUENCE_SETS)}] {set_name}: {sequences}, {len(columns)} features", flush=True)
        for fold, (train_index, validation_index) in enumerate(outer_splits, 1):
            X_train = X_set.iloc[train_index]
            y_train = y.iloc[train_index]
            X_validation = X_set.iloc[validation_index]
            y_validation = y.iloc[validation_index]
            icc_filter = TechnicalICCFilter(*readers).fit(X_train)
            X_train = icc_filter.transform(X_train)
            X_validation = icc_filter.transform(X_validation)
            inner_splits = list(
                StratifiedKFold(3, shuffle=True, random_state=420 + fold).split(X_train, y_train)
            )
            feasible_k = feasible_k_grid(X_train, y_train, inner_splits)
            pipeline = make_pipeline(memory)
            search = GridSearchCV(
                pipeline,
                {"select__k": feasible_k, "model__C": C_GRID},
                scoring="roc_auc",
                cv=inner_splits,
                n_jobs=args.jobs,
                error_score="raise",
                refit=True,
                return_train_score=False,
            )
            with joblib.parallel_backend("threading", n_jobs=args.jobs):
                search.fit(X_train, y_train)
            probability = search.predict_proba(X_validation)[:, 1]
            predictions[validation_index] = probability
            audit = model_audit(search.best_estimator_, icc_filter)
            fold_metric = classification_metrics(y_validation, probability)
            fold_rows.append(
                {
                    "sequence_set": set_name,
                    "sequences": "+".join(sequences),
                    "outer_fold": fold,
                    "n_train": len(train_index),
                    "n_validation": len(validation_index),
                    "n_train_positive": int(y_train.sum()),
                    "n_validation_positive": int(y_validation.sum()),
                    "inner_best_auc": float(search.best_score_),
                    "selected_k": int(search.best_params_["select__k"]),
                    "ridge_C": float(search.best_params_["model__C"]),
                    "n_qc": audit["n_qc"],
                    "n_clusters": audit["n_clusters"],
                    "icc_training_n": audit["icc_training_n"],
                    **fold_metric,
                }
            )
            for feature, score, cluster_size, sequence, coefficient in zip(
                audit["selected_features"],
                audit["selected_feature_scores"],
                audit["selected_cluster_sizes"],
                audit["selected_sequences"],
                audit["coefficients"],
            ):
                selected_rows.append(
                    {
                        "fit": "outer_fold",
                        "sequence_set": set_name,
                        "outer_fold": fold,
                        "feature": feature,
                        "sequence": sequence,
                        "univariate_auc_distance": score,
                        "cluster_size": cluster_size,
                        "standardized_ridge_coefficient": coefficient,
                    }
                )
            pd.DataFrame(search.cv_results_).to_csv(
                set_dir / f"fold_{fold}_inner_search.csv", index=False
            )
            write_json(
                set_dir / f"fold_{fold}_model.json",
                {
                    "best_params": search.best_params_,
                    "inner_best_auc": search.best_score_,
                    "outer_metrics": fold_metric,
                    "audit": audit,
                },
            )
            oof_rows.extend(
                {
                    "sequence_set": set_name,
                    "outer_fold": fold,
                    "PatientID": patient,
                    "y": int(outcome),
                    "prediction": float(prediction),
                }
                for patient, outcome, prediction in zip(
                    X_validation.index, y_validation, probability
                )
            )
            print(
                f"  fold {fold}: inner={search.best_score_:.3f}, k={search.best_params_['select__k']}, "
                f"C={search.best_params_['model__C']}, outer_auc={fold_metric['auc']:.3f}",
                flush=True,
            )
        if not np.isfinite(predictions).all():
            raise RuntimeError(f"Incomplete OOF predictions for {set_name}")
        pooled = classification_metrics(y, predictions)
        low, high = bootstrap_auc(y, predictions, args.bootstrap, 20261014 + set_index)
        fold_frame = pd.DataFrame(fold_rows)
        relevant = fold_frame[fold_frame.sequence_set == set_name]
        summary_rows.append(
            {
                "sequence_set": set_name,
                "sequences": "+".join(sequences),
                "n_input_features": len(columns),
                **pooled,
                "auc_ci_low": low,
                "auc_ci_high": high,
                "mean_fold_auc": float(relevant.auc.mean()),
                "sd_fold_auc": float(relevant.auc.std(ddof=1)),
                "mean_fold_average_precision": float(relevant.average_precision.mean()),
                "mean_fold_brier": float(relevant.brier.mean()),
            }
        )

        final_icc_filter = TechnicalICCFilter(*readers).fit(X_set)
        X_final = final_icc_filter.transform(X_set)
        final_inner = list(StratifiedKFold(3, shuffle=True, random_state=999).split(X_final, y))
        final_feasible_k = feasible_k_grid(X_final, y, final_inner)
        final_search = GridSearchCV(
            make_pipeline(memory),
            {"select__k": final_feasible_k, "model__C": C_GRID},
            scoring="roc_auc",
            cv=final_inner,
            n_jobs=args.jobs,
            error_score="raise",
            refit=True,
            return_train_score=False,
        )
        with joblib.parallel_backend("threading", n_jobs=args.jobs):
            final_search.fit(X_final, y)
        final_audit = model_audit(final_search.best_estimator_, final_icc_filter)
        joblib.dump(
            {
                "icc_filter": final_icc_filter,
                "model_pipeline": final_search.best_estimator_,
                "input_sequences": sequences,
                "lock_evaluated": False,
            },
            set_dir / "final_development_model.joblib",
        )
        pd.DataFrame(final_search.cv_results_).to_csv(set_dir / "final_inner_search.csv", index=False)
        write_json(
            set_dir / "final_development_model.json",
            {
                "best_params": final_search.best_params_,
                "inner_best_auc": final_search.best_score_,
                "audit": final_audit,
                "lock_evaluated": False,
            },
        )
        for rank, (feature, score, cluster_size, sequence, coefficient) in enumerate(
            zip(
                final_audit["selected_features"],
                final_audit["selected_feature_scores"],
                final_audit["selected_cluster_sizes"],
                final_audit["selected_sequences"],
                final_audit["coefficients"],
            ),
            1,
        ):
            selected_rows.append(
                {
                    "fit": "final_development",
                    "sequence_set": set_name,
                    "outer_fold": 0,
                    "feature": feature,
                    "sequence": sequence,
                    "univariate_auc_distance": score,
                    "cluster_size": cluster_size,
                    "standardized_ridge_coefficient": coefficient,
                }
            )
            final_rows.append(
                {
                    "sequence_set": set_name,
                    "rank": rank,
                    "feature": feature,
                    "sequence": sequence,
                    "univariate_auc_distance": score,
                    "cluster_size": cluster_size,
                    "standardized_ridge_coefficient": coefficient,
                    "selected_k": final_search.best_params_["select__k"],
                    "ridge_C": final_search.best_params_["model__C"],
                    "inner_best_auc": final_search.best_score_,
                }
            )

    baseline = classification_metrics(y, baseline_prediction)
    base_low, base_high = bootstrap_auc(y, baseline_prediction, args.bootstrap, 20262014)
    summary_rows.append(
        {
            "sequence_set": "training_mean_baseline",
            "sequences": "",
            "n_input_features": 0,
            **baseline,
            "auc_ci_low": base_low,
            "auc_ci_high": base_high,
            "mean_fold_auc": 0.5,
            "sd_fold_auc": 0.0,
            "mean_fold_average_precision": float(y.mean()),
            "mean_fold_brier": baseline["brier"],
        }
    )
    oof_rows.extend(
        {
            "sequence_set": "training_mean_baseline",
            "outer_fold": fold,
            "PatientID": patient,
            "y": int(outcome),
            "prediction": float(prediction),
        }
        for fold, (_, validation_index) in enumerate(outer_splits, 1)
        for patient, outcome, prediction in zip(
            X.index[validation_index], y.iloc[validation_index], baseline_prediction[validation_index]
        )
    )

    summary = pd.DataFrame(summary_rows).sort_values("auc", ascending=False)
    summary.to_csv(run / "summary.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(run / "fold_metrics.csv", index=False)
    pd.DataFrame(oof_rows).to_csv(run / "oof_predictions.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(run / "selected_features_all_fits.csv", index=False)
    pd.DataFrame(final_rows).to_csv(run / "final_development_signatures.csv", index=False)
    pd.DataFrame(split_rows).drop_duplicates().to_csv(run / "cv_assignments.csv", index=False)
    manifest.update(
        {
            "completed": datetime.now().isoformat(),
            "status": "complete",
            "best_descriptive_sequence_set": summary.iloc[0].sequence_set,
            "best_descriptive_oof_auc": summary.iloc[0].auc,
        }
    )
    write_json(run / "manifest.json", manifest)
    report = [
        "# Compact ridge targeted sequence search",
        "",
        "Primary endpoint: TILs ≥2. Development-only nested CV; lock outcomes were not evaluated.",
        "Each fitted model contains exactly 10, 15, or 20 named original radiomics features.",
        "",
        "## Results",
        "",
        summary.to_markdown(index=False, floatfmt=".3f"),
        "",
        "The highest OOF AUC is descriptive because sequence sets were compared on the same outer folds.",
        "All QC, ICC, clustering, outcome-based representative selection, top-k ranking, scaling, and ridge tuning were refitted within training data.",
        "",
        "## Provenance",
        "",
        f"- Corrected feature sheet: `{feature_path}`",
        f"- SHA-256: `{manifest['input_hashes'][str(feature_path)]}`",
        "- Original features only; no wavelet or LoG features.",
        "- Existing 110-development/30-lock assignment retained.",
    ]
    (run / "REPORT.md").write_text("\n".join(report) + "\n")
    print("\nFINAL SUMMARY", flush=True)
    print(summary.to_string(index=False), flush=True)
    print(f"COMPLETED={run}", flush=True)


if __name__ == "__main__":
    main()
