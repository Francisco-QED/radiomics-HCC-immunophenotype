"""Ablation ladder: walk from the 'gpt' compact-ridge pipeline toward ours, ONE choice per step.

Spine = main_pipeline_gpt/run_compact_ridge_sequence_search.py (imported read-only, never modified):
same cohort/sheet/split, same outer folds (5, seed 42), same inner folds (3, seed 420+fold), ICC fit on
outer-training paired-reader patients, TrainingQC, ridge logistic, C grid, k grid {10,15,20}, AUC-scored
inner CV. Only the steps below differ. Steps are CUMULATIVE.

  S0_theirs      median impute | within-sequence complete-linkage clusters | top-k by AUC           (their method)
  S1_knn         + KNN imputation (k=5)
  S2_crossseq    + clusters formed across ALL selected sequences (not within each sequence)
  S3_aucgate     + hard univariate gate: representative must have AUC >= 0.65 or <= 0.35
  S4_stability   + nested stability selection INSIDE each training set: B subsamples (75%), a cluster is kept
                   only if it enters the top-k in >= 70% of them (cluster level, no sign rule)

An empty selection (gate/stability leaves nothing) falls back to the training prevalence, so the fold still
produces predictions and is counted (n_empty_folds). FDR is not used anywhere (it was a no-op).
Development patients only; lock outcomes are never read.
"""
import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

sys.dont_write_bytecode = True          # never drop __pycache__ into the read-only gpt folder
for _n in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_n, "1")

import numpy as np
import pandas as pd
import sklearn
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import rankdata
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_CORE_DIR = ("/Users/Francisco/Library/CloudStorage/OneDrive-TheMountSinaiHospital/Regeneron retrospective "
                    "radiomics/Results/main_pipeline_gpt")

SEQUENCE_SETS = {
    "all9": ("pre", "ap", "pvp", "tp", "hbp", "t2", "dwi_b1", "dwi_b2", "adc"),
    "tp": ("tp",),
    "dwi_b2+pvp+tp": ("dwi_b2", "pvp", "tp"),
}
STEPS = {  # cumulative
    "S0_theirs":    dict(impute="median", scope="within", gate=None, stab=0),
    "S1_knn":       dict(impute="knn",    scope="within", gate=None, stab=0),
    "S2_crossseq":  dict(impute="knn",    scope="across", gate=None, stab=0),
    "S3_aucgate":   dict(impute="knn",    scope="across", gate=0.65, stab=0),
    "S4_stability": dict(impute="knn",    scope="across", gate=0.65, stab=50),
}


def load_core(core_dir):
    sys.path.insert(0, str(core_dir))
    import run_compact_ridge_sequence_search as core  # noqa: E402
    return core


class KNNFrame(BaseEstimator, TransformerMixin):
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self, X, y=None):
        self.columns_ = list(X.columns)
        self.imputer_ = KNNImputer(n_neighbors=self.n_neighbors).fit(X)
        return self

    def transform(self, X):
        out = self.imputer_.transform(X.loc[:, self.columns_])
        return pd.DataFrame(out, index=X.index, columns=self.columns_)


def make_selector_class(core):
    class LadderSelector(BaseEstimator, TransformerMixin):
        """Cluster -> representative (highest univariate AUC distance) -> optional gate -> top-k
        -> optional nested stability filter. Complete linkage at |rho| >= threshold."""

        def __init__(self, k=10, threshold=0.85, scope="within", gate=None, stab=0, stab_thr=0.70,
                     stab_frac=0.75, seed=42, forced=()):
            self.forced = forced      # columns always kept (not clustered/ranked), e.g. clinical variables
            self.k = k
            self.threshold = threshold
            self.scope = scope
            self.gate = gate
            self.stab = stab
            self.stab_thr = stab_thr
            self.stab_frac = stab_frac
            self.seed = seed

        # -- helpers -------------------------------------------------------------------------------
        def _groups(self, columns):
            if self.scope == "across":
                return [sorted(columns)]
            groups = {}
            for c in columns:
                groups.setdefault("clin" if c.startswith("clin_") else core.feature_sequence(c), []).append(c)
            return [sorted(v) for _, v in sorted(groups.items())]

        def _cluster(self, values):
            if values.shape[1] == 1:
                return np.array([1])
            ranks = rankdata(values, axis=0)
            corr = np.corrcoef(ranks, rowvar=False)
            corr = np.nan_to_num(corr, nan=0.0)
            corr = np.clip((corr + corr.T) / 2, -1, 1)
            np.fill_diagonal(corr, 1.0)
            dist = np.clip(1 - np.abs(corr), 0, 1)
            np.fill_diagonal(dist, 0.0)
            return fcluster(linkage(squareform(dist, checks=False), method="complete"),
                            t=1 - self.threshold, criterion="distance")

        def _pick(self, scores, members, order_names):
            """one representative per cluster (max score, ties -> name), gate, top-k. returns list of cluster ids."""
            reps = []
            for cid, idx in members.items():
                best = scores[idx].max()
                tied = [i for i in idx if np.isclose(scores[i], best, rtol=0, atol=1e-15)]
                rep = min(tied, key=lambda i: order_names[i])
                reps.append((cid, rep, float(scores[rep])))
            if self.gate is not None:
                floor = self.gate - 0.5
                reps = [r for r in reps if r[2] >= floor - 1e-12]
            reps.sort(key=lambda r: (-r[2], order_names[r[1]]))
            return reps[: int(self.k)]

        # -- sklearn API ---------------------------------------------------------------------------
        def fit(self, X, y):
            forced = [c for c in self.forced if c in X.columns]
            Xr = X.drop(columns=forced)
            cols = list(Xr.columns)
            yv = np.asarray(y, dtype=int)
            scores = core.auc_distance_scores(Xr.to_numpy(float), yv)
            members, cid = {}, 0
            for group in self._groups(cols):
                gidx = [cols.index(c) for c in group]
                labels = self._cluster(Xr.iloc[:, gidx].to_numpy(float))
                for lab in np.unique(labels):
                    cid += 1
                    members[cid] = [gidx[i] for i in np.flatnonzero(labels == lab)]
            self.n_clusters_ = len(members)
            full = self._pick(scores, members, cols)
            self.cluster_of_feature_ = {cols[i]: c for c, idx in members.items() for i in idx}

            if self.stab and self.stab > 0:
                rng = np.random.default_rng(self.seed)
                hits = {c: 0 for c in members}
                pos, neg = np.flatnonzero(yv == 1), np.flatnonzero(yv == 0)
                values = Xr.to_numpy(float)
                for _ in range(int(self.stab)):
                    take = np.r_[rng.choice(pos, size=max(2, int(round(len(pos) * self.stab_frac))), replace=False),
                                 rng.choice(neg, size=max(2, int(round(len(neg) * self.stab_frac))), replace=False)]
                    s = core.auc_distance_scores(values[take], yv[take])
                    for c, _, _ in self._pick(s, members, cols):
                        hits[c] += 1
                self.cluster_frequency_ = {c: hits[c] / self.stab for c in members}
                keep = [r for r in full if self.cluster_frequency_[r[0]] >= self.stab_thr - 1e-12]
            else:
                self.cluster_frequency_ = {}
                keep = full
            self.selected_features_ = [cols[r[1]] for r in keep] + forced
            self.selected_clusters_ = [r[0] for r in keep]
            self.selected_scores_ = [r[2] for r in keep]
            self.empty_ = len(self.selected_features_) == 0
            return self

        def transform(self, X):
            if self.empty_:        # prevalence fallback: a constant column makes SafeRidge use the prior
                return pd.DataFrame({"__empty__": np.zeros(len(X))}, index=X.index)
            return X.loc[:, self.selected_features_]

    return LadderSelector


def build_pipeline(core, Selector, cfg, forced=()):
    imputer = KNNFrame(5) if cfg["impute"] == "knn" else core.MedianImputerFrame()
    return Pipeline([
        ("qc", core.TrainingQC()),
        ("impute", imputer),
        ("select", Selector(scope=cfg["scope"], gate=cfg["gate"], stab=cfg["stab"], forced=tuple(forced))),
        ("scale", StandardScaler()),
        ("model", core.SafeRidgeClassifier()),
    ])


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-parent", required=True)
    ap.add_argument("--core-dir", default=DEFAULT_CORE_DIR)
    ap.add_argument("--sets", nargs="+", default=list(SEQUENCE_SETS))
    ap.add_argument("--steps", nargs="+", default=list(STEPS))
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=1,
                    help="repeated outer CV; repeat r uses outer seed 42+r and inner seeds 420+fold+1000*r")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--label", default="ablation_ladder_theirs_to_ours")
    a = ap.parse_args()
    warnings.filterwarnings("ignore")

    core = load_core(a.core_dir)
    Selector = make_selector_class(core)
    root = Path(core.ROOT)
    X, y, dev_ids, lock_ids, split_path, feature_path, readers = core.load_development(root)

    run = Path(a.output_parent) / f"{a.label}_{datetime.now():%Y%m%d_%H%M%S}"
    run.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, run / "ablation_ladder.py")
    manifest = {
        "started": datetime.now().isoformat(), "status": "running", "endpoint": "TILs_resection >= 2",
        "n_development": len(dev_ids), "n_positive": int(y.sum()), "lock_evaluated": False,
        "n_lock_excluded": len(lock_ids), "steps": {s: STEPS[s] for s in a.steps}, "sets": {s: SEQUENCE_SETS[s] for s in a.sets},
        "outer_folds": 5, "outer_seed": "42+repeat", "inner_folds": 3, "inner_seed": "420+fold+1000*repeat", "repeats": a.repeats,
        "k_grid": list(core.K_GRID), "C_grid": list(core.C_GRID), "stability": {"B": 50, "threshold": 0.70, "frac": 0.75},
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__, "numpy": np.__version__,
                     "pandas": pd.__version__}, "core_script": str(Path(core.__file__)),
        "sha256": {"features": sha256(feature_path), "split": sha256(split_path), "core_script": sha256(core.__file__)},
    }
    (run / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print("RUN_DIRECTORY=" + str(run), flush=True)

    summary, folds, oof, selected = [], [], [], []
    for set_name in a.sets:
        seqs = SEQUENCE_SETS[set_name]
        cols = sorted(c for c in X.columns if any(c.startswith(s + "_original_") for s in seqs))
        Xs = X[cols]
        for step in a.steps:
            cfg = STEPS[step]
            for rep in range(a.repeats):
                # repeat 0 == the original single run (outer seed 42, inner seeds 420+fold)
                outer = list(StratifiedKFold(5, shuffle=True, random_state=42 + rep).split(X, y))
                t0 = time.time()
                pred = np.full(len(y), np.nan)
                print(f"[{set_name} | {step} | repeat {rep}] {len(cols)} input features", flush=True)
                fold_rows = []
                for fold, (tr, va) in enumerate(outer, 1):
                    Xtr, ytr, Xva, yva = Xs.iloc[tr], y.iloc[tr], Xs.iloc[va], y.iloc[va]
                    icc = core.TechnicalICCFilter(*readers).fit(Xtr)
                    Xtr, Xva = icc.transform(Xtr), icc.transform(Xva)
                    inner = list(StratifiedKFold(3, shuffle=True, random_state=420 + fold + 1000 * rep).split(Xtr, ytr))
                    search = GridSearchCV(build_pipeline(core, Selector, cfg),
                                          {"select__k": tuple(core.K_GRID), "model__C": core.C_GRID},
                                          scoring="roc_auc", cv=inner, n_jobs=a.jobs, error_score="raise", refit=True)
                    search.fit(Xtr, ytr)
                    p = search.predict_proba(Xva)[:, 1]
                    pred[va] = p
                    sel = search.best_estimator_.named_steps["select"]
                    auc = core.classification_metrics(yva, p)["auc"] if yva.nunique() == 2 else np.nan
                    row = dict(sequence_set=set_name, step=step, repeat=rep, outer_fold=fold,
                               inner_best_auc=float(search.best_score_), k=int(search.best_params_["select__k"]),
                               C=float(search.best_params_["model__C"]),
                               n_selected=0 if sel.empty_ else len(sel.selected_features_), was_empty=bool(sel.empty_),
                               n_clusters=int(sel.n_clusters_), outer_auc=auc)
                    folds.append(row)
                    fold_rows.append(row)
                    for f, c, sc in zip(sel.selected_features_, sel.selected_clusters_, sel.selected_scores_):
                        selected.append(dict(sequence_set=set_name, step=step, repeat=rep, outer_fold=fold, feature=f,
                                             cluster=c, univariate_auc_distance=sc,
                                             cluster_frequency=sel.cluster_frequency_.get(c, np.nan)))
                    oof.extend(dict(sequence_set=set_name, step=step, repeat=rep, outer_fold=fold, PatientID=i,
                                    y=int(o), prediction=float(q)) for i, o, q in zip(Xva.index, yva, p))
                m = core.classification_metrics(y, pred)
                lo, hi = core.bootstrap_auc(y, pred, a.bootstrap, 20261014)
                ff = pd.DataFrame(fold_rows)
                summary.append(dict(sequence_set=set_name, step=step, repeat=rep, n_input_features=len(cols), **m,
                                    auc_ci_low=lo, auc_ci_high=hi, mean_fold_auc=float(ff.outer_auc.mean()),
                                    sd_fold_auc=float(ff.outer_auc.std(ddof=1)), n_empty_folds=int(ff.was_empty.sum()),
                                    mean_n_selected=float(ff.n_selected.mean()), seconds=round(time.time() - t0)))
                print(f"   -> pooled AUC {m['auc']:.4f} (CI {lo:.3f}-{hi:.3f}), empty folds {int(ff.was_empty.sum())}/5, "
                      f"{summary[-1]['seconds']}s", flush=True)
                pd.DataFrame(summary).to_csv(run / "summary.csv", index=False)
                pd.DataFrame(folds).to_csv(run / "fold_metrics.csv", index=False)
                pd.DataFrame(oof).to_csv(run / "oof_predictions.csv", index=False)
                pd.DataFrame(selected).to_csv(run / "selected_features.csv", index=False)

    manifest.update(status="completed", finished=datetime.now().isoformat())
    (run / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print("COMPLETED=" + str(run), flush=True)


if __name__ == "__main__":
    main()
