"""Time-to-recurrence development pipeline that mirrors the TILs/TLS radiomics procedure, with survival components.

Per outer-training set (everything outcome-dependent is fit on training patients only):
  reader-ICC(3,1) >= 0.85 (estimated once on all dev double-reader patients: outcome-independent)  ->  QC (>=80% observed, CV > 0.10)
  -> median imputation -> within-sequence complete-linkage clusters (|rho| >= 0.85) -> ONE representative per cluster, ranked by the
  univariate concordance distance |C-index - 0.5| with (event, time)  -> standardization ->
  ridge Cox (top-k clusters, k in {10,15,20}, alpha grid)  |  lasso / elastic-net Cox (Coxnet) tuned on inner C-index with a 10-20 nonzero-coefficient constraint.
Validation: repeated outer 5-fold CV stratified on event x cohort (repeat r: outer seed 42+r), inner 3-fold (seed 420+fold+1000*r) stratified on event;
metric = pooled out-of-fold Harrell C-index. Dev patients only; lock never read.
Optional clinical variables (--clinical): none | forced (always included, not ranked) | compete (rank together with the radiomics clusters inside each fold).
"""
import argparse, json, sys, time, warnings
from datetime import datetime
from pathlib import Path
sys.dont_write_bytecode = True
import numpy as np, pandas as pd, joblib, sklearn
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import rankdata
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

ROOT = Path("/Users/Francisco/Library/CloudStorage/OneDrive-TheMountSinaiHospital/Regeneron retrospective radiomics")
K_GRID = (10, 15, 20)
RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)
EN_RATIOS = (0.25, 0.5, 0.75)
SUPPORT_MIN, SUPPORT_MAX, EPS = 10, 20, 1e-7


def uni_cindex_distance(X, event, time):
    """|Harrell C - 0.5| of each column vs (event, time); all comparable pairs, vectorized."""
    M = (time[:, None] < time[None, :]) & event[:, None]
    I, J = np.nonzero(M)
    if len(I) == 0:
        return np.zeros(X.shape[1])
    d = X[I, :] - X[J, :]
    c = ((d > 0).sum(0) + 0.5 * (d == 0).sum(0)) / len(I)
    return np.abs(c - 0.5)


def make_selector(core):
    class SurvClusterSelector(BaseEstimator, TransformerMixin):
        def __init__(self, k=None, threshold=0.85, forced=()):
            self.k = k; self.threshold = threshold; self.forced = forced

        def _labels(self, values):
            if values.shape[1] == 1:
                return np.array([1])
            corr = np.nan_to_num(np.corrcoef(rankdata(values, axis=0), rowvar=False), nan=0.0)
            corr = np.clip((corr + corr.T) / 2, -1, 1); np.fill_diagonal(corr, 1.0)
            dist = np.clip(1 - np.abs(corr), 0, 1); np.fill_diagonal(dist, 0.0)
            return fcluster(linkage(squareform(dist, checks=False), method="complete"), t=1 - self.threshold, criterion="distance")

        def fit(self, X, y):
            forced = [c for c in self.forced if c in X.columns]
            Xr = X.drop(columns=forced); cols = list(Xr.columns)
            event = np.asarray(y["event"], bool); time = np.asarray(y["time"], float)
            scores = uni_cindex_distance(Xr.to_numpy(float), event, time)
            groups = {}
            for i, c in enumerate(cols):
                groups.setdefault("clin" if c.startswith("clin_") else core.feature_sequence(c), []).append(i)
            reps = []
            for _, idx in sorted(groups.items()):
                lab = self._labels(Xr.iloc[:, idx].to_numpy(float))
                for l in np.unique(lab):
                    members = [idx[j] for j in np.flatnonzero(lab == l)]
                    best = max(scores[m] for m in members)
                    tied = [m for m in members if np.isclose(scores[m], best, rtol=0, atol=1e-15)]
                    rep = min(tied, key=lambda m: cols[m]); reps.append((float(scores[rep]), cols[rep]))
            reps.sort(key=lambda r: (-r[0], r[1]))
            self.n_clusters_ = len(reps)
            if self.k is not None and len(reps) < int(self.k):
                raise ValueError(f"Only {len(reps)} clusters available for k={self.k}")
            chosen = reps if self.k is None else reps[: int(self.k)]
            self.selected_features_ = [n for _, n in chosen] + forced
            return self

        def transform(self, X):
            return X.loc[:, self.selected_features_]
    return SurvClusterSelector


def prep_steps(core, Sel, k=None, forced=()):
    return [("qc", core.TrainingQC()), ("impute", core.MedianImputerFrame()), ("select", Sel(k=k, forced=tuple(forced))), ("scale", StandardScaler())]


def n_support(est):
    return float(np.sum(np.abs(np.asarray(est.named_steps["model"].coef_)) > EPS))


def fit_ridge(core, Sel, Xtr, ytr, inner, forced, jobs):
    pipe = Pipeline(prep_steps(core, Sel, 10, forced) + [("model", CoxPHSurvivalAnalysis(alpha=1.0, ties="breslow"))])
    gs = GridSearchCV(pipe, {"select__k": K_GRID, "model__alpha": RIDGE_ALPHAS}, cv=inner, n_jobs=jobs, error_score=np.nan, refit=True)
    gs.fit(Xtr, ytr)
    return gs.best_estimator_, dict(gs.best_params_), float(gs.best_score_)


def fit_sparse(core, Sel, family, Xtr, ytr, inner, forced, jobs):
    ratios = (1.0,) if family == "lasso" else EN_RATIOS
    pre = Pipeline(prep_steps(core, Sel, None, forced)).fit(Xtr, ytr); Xt = pre.transform(Xtr)
    grid = []
    for r in ratios:
        amax = float(CoxnetSurvivalAnalysis(l1_ratio=r, n_alphas=2, alpha_min_ratio=0.9, fit_baseline_model=False).fit(Xt, ytr).alphas_[0])
        grid.append({"model__l1_ratio": [r], "model__alphas": [[float(a)] for a in np.geomspace(amax, amax * 0.01, 14)]})
    base = Pipeline(prep_steps(core, Sel, None, forced) + [("model", CoxnetSurvivalAnalysis(l1_ratio=1.0, alphas=[0.1], fit_baseline_model=False, max_iter=100000))])
    gs = GridSearchCV(base, grid, scoring={"c": lambda e, X, y: float(e.score(X, y)), "support": lambda e, X, y: n_support(e)}, cv=inner, n_jobs=jobs, error_score=np.nan, refit=False)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        gs.fit(Xtr, ytr)
    f = pd.DataFrame(gs.cv_results_); scols = [c for c in f if c.startswith("split") and c.endswith("_test_support")]
    ok = f.mean_test_support.between(SUPPORT_MIN, SUPPORT_MAX) & f[scols].min(axis=1).ge(5) & f[scols].max(axis=1).le(30) & np.isfinite(f.mean_test_c)
    cand = f[ok].copy()
    if cand.empty:
        raise RuntimeError("No sparse Cox candidate met the 10-20 support constraint")
    cand["alpha0"] = cand.param_model__alphas.map(lambda a: a[0]); cand = cand.sort_values(["mean_test_c", "alpha0"], ascending=[False, False])
    for _, row in cand.iterrows():
        center = float(row.alpha0); local = sorted(np.unique(np.r_[center, np.geomspace(center / 8, center * 8, 101)]), key=lambda a: abs(np.log(a / center)))
        for a in local:
            est = clone(base).set_params(model__l1_ratio=float(row.param_model__l1_ratio), model__alphas=[float(a)])
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", ConvergenceWarning); est.fit(Xtr, ytr)
            except Exception:
                continue
            if SUPPORT_MIN <= n_support(est) <= SUPPORT_MAX:
                return est, {"alpha": float(a), "l1_ratio": float(row.param_model__l1_ratio)}, float(row.mean_test_c)
    raise RuntimeError("Inner-valid Cox candidates did not retain 10-20 features after refit")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpt-dir", required=True)
    ap.add_argument("--sheet", choices=["pre", "post"], default="pre")
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--families", nargs="+", default=["ridge", "lasso", "elastic_net"])
    ap.add_argument("--clinical", choices=["none", "forced", "compete"], default="none")
    ap.add_argument("--clinical-matrix", default=None, help="CSV (PatientID index) of clinical variables; columns are prefixed clin_ internally")
    ap.add_argument("--clinical-cols", nargs="*", default=None)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--split", default=str(ROOT / "Results/main_pipeline_claude/main_pipeline_recurrence/split/recurrence_split.csv"))
    ap.add_argument("--output-parent", required=True)
    ap.add_argument("--label", default="recurrence")
    a = ap.parse_args(); warnings.filterwarnings("ignore")
    sys.path.insert(0, a.gpt_dir)
    import run_compact_ridge_sequence_search as core
    Sel = make_selector(core)
    sp = pd.read_csv(a.split); dev = sp[sp["set"] == "dev"].reset_index(drop=True); ids = dev.PatientID.tolist()
    ytr_all = Surv.from_arrays(event=dev.event.astype(bool).values, time=dev.time_days.astype(float).values, name_event="event", name_time="time")
    sheet = ROOT / ("Results/Radiomics_sheets/SA1_2_pre/R-SA1_2_I-sitkBSpline_S-1mm_W-perseq.csv" if a.sheet == "pre" else "Results/Radiomics_sheets/SA1_2_post/R-SA1_2-post_I-sitkBSpline_S-1mm_W-perseq.csv")
    X = pd.read_csv(sheet, index_col=0, usecols=lambda c: c == "PatientID" or "_original_" in c).loc[ids]
    X = X.reindex(sorted(X.columns), axis=1).astype(float)
    readers = [str(ROOT / f"Results/Radiomics_sheets/inter-reader/Reader{i}/R-Reader{i}_I-sitkBSpline_S-1mm_W-perseq.csv") for i in (1, 2)]
    forced = ()
    if a.clinical != "none":
        cm = pd.read_csv(a.clinical_matrix, index_col=0).loc[ids]
        cols = a.clinical_cols or [c for c in cm.columns if c not in ("cohort",)]
        clin = cm[cols].add_prefix("clin_"); forced = tuple(clin.columns) if a.clinical == "forced" else ()
    strata_all = dev.event.astype(int).astype(str) + "_" + dev.cohort
    run = Path(a.output_parent) / f"{a.label}_{a.sheet}_{a.clinical}_{'-'.join(s.replace('+', '_') for s in a.sets)}_{datetime.now():%Y%m%d_%H%M%S}"; run.mkdir(parents=True, exist_ok=False)
    (run / "manifest.json").write_text(json.dumps(dict(started=datetime.now().isoformat(), endpoint="time to recurrence (Harrell C, pooled OOF)", n_dev=len(ids), events=int(dev.event.sum()), lock_evaluated=False,
        sets=a.sets, families=a.families, clinical=a.clinical, clinical_cols=list(clin.columns) if a.clinical != "none" else [], repeats=a.repeats, sheet=str(sheet),
        strata=strata_all.value_counts().to_dict(), icc="global (all dev double-reader patients)", sklearn=sklearn.__version__), indent=2))
    print("RUN_DIRECTORY=" + str(run), "| dev", len(ids), "events", int(dev.event.sum()), "| clinical:", a.clinical, flush=True)
    RealICC = core.TechnicalICCFilter; rows = []; oof_rows = []
    for sname in a.sets:
        cols_r = sorted(c for c in X.columns if any(c.startswith(s + "_original_") for s in sname.split("+"))) if sname != "clinical_only" else []
        Xr = X[cols_r]
        if cols_r:
            fixed = list(RealICC(*readers).fit(Xr).columns_)
        else:
            fixed = []
        print(f"[{sname}] global ICC keeps {len(fixed)} of {len(cols_r)} radiomic features", flush=True)
        Xs = pd.concat([Xr[fixed], clin], axis=1) if a.clinical != "none" else Xr[fixed]
        if Xs.shape[1] == 0:
            continue
        for fam in a.families:
            for rep in range(a.repeats):
                t0 = time.time(); risk = np.full(len(ids), np.nan); fold_c = []
                try:
                    for fold, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42 + rep).split(Xs, strata_all), 1):
                        inner = list(StratifiedKFold(3, shuffle=True, random_state=420 + fold + 1000 * rep).split(Xs.iloc[tr], dev.event.iloc[tr]))
                        if fam == "ridge":
                            est, params, score = fit_ridge(core, Sel, Xs.iloc[tr], ytr_all[tr], inner, forced, a.jobs)
                        else:
                            est, params, score = fit_sparse(core, Sel, fam, Xs.iloc[tr], ytr_all[tr], inner, forced, a.jobs)
                        risk[va] = est.predict(Xs.iloc[va])
                        fold_c.append(concordance_index_censored(ytr_all[va]["event"], ytr_all[va]["time"], risk[va])[0])
                except (ValueError, RuntimeError, ArithmeticError, np.linalg.LinAlgError) as e:   # ArithmeticError covers FloatingPointError and Coxnet's "weights too large"
                    rows.append(dict(sequence_set=sname, family=fam, repeat=rep, c_index=np.nan, failed_reason=str(e)[:100]))
                    pd.DataFrame(rows).to_csv(run / "per_repeat.csv", index=False)
                    print(f"[{sname} | {fam}] repeat {rep}: FAILED ({str(e)[:80]})", flush=True); continue
                c = concordance_index_censored(ytr_all["event"], ytr_all["time"], risk)[0]
                rows.append(dict(sequence_set=sname, family=fam, repeat=rep, c_index=float(c), mean_fold_c=float(np.mean(fold_c))))
                oof_rows.append(pd.DataFrame(dict(sequence_set=sname, family=fam, repeat=rep, PatientID=ids, risk=risk))); pd.concat(oof_rows).to_csv(run / "oof_risk.csv", index=False)
                pd.DataFrame(rows).to_csv(run / "per_repeat.csv", index=False)
                print(f"[{sname} | {fam}] repeat {rep}: C-index {c:.3f} ({time.time()-t0:.0f}s)", flush=True)
    D = pd.DataFrame(rows)
    D.groupby(["sequence_set", "family"]).c_index.agg(mean="mean", sd="std", mn="min", mx="max", n_ok=lambda x: int(x.notna().sum()), n="size").reset_index().round(4).to_csv(run / "summary.csv", index=False)
    print("COMPLETED=" + str(run), flush=True)


if __name__ == "__main__":
    main()
