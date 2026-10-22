"""Clinical-only time-to-recurrence baseline on the dev set, same outer folds as the radiomics runs.

Outer: 5-fold stratified on event x cohort, repeat r seed 42+r (identical to repeated_survival_families.py). Inner: 3-fold on event, seed 420+fold+1000*r.
Per fold: median imputation + z-scoring fit on the training part; Cox ridge (alpha grid), Cox lasso / elastic net (Coxnet, alpha anchored at alpha_max,
tuned by inner C-index; no 10-20 support constraint: ~20 inputs), and two fixed small Cox models (size+AFP, size+AFP+macrovascular invasion).
Metrics per repeat: pooled out-of-fold Harrell C and time-dependent AUC at 1/2/3 years (IPCW censoring from the dev set).
Univariate C-index per variable on all dev (descriptive, uncorrected, direction shown).
Dev patients only; lock never read.
"""
import argparse, json, sys, warnings
from datetime import datetime
from pathlib import Path
sys.dont_write_bytecode = True
import numpy as np, pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc
from sksurv.util import Surv

ROOT = Path("/Users/Francisco/Library/CloudStorage/OneDrive-TheMountSinaiHospital/Regeneron retrospective radiomics")
RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
EN_RATIOS = (0.25, 0.5, 0.75, 1.0)
FIXED = {"size_afp": ["tumor_size_pvp_cm", "log_afp"], "size_afp_mvi": ["tumor_size_pvp_cm", "log_afp", "macrovascular_invasion"]}


def base_steps():
    return [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]


def fit_model(family, X, y, inner, cols):
    if family in FIXED:
        keep = [cols.index(c) for c in FIXED[family]]
        est = Pipeline(base_steps() + [("model", CoxPHSurvivalAnalysis(alpha=1.0, ties="breslow"))])
        est.fit(X[:, keep], y); return est, keep
    if family == "ridge":
        gs = GridSearchCV(Pipeline(base_steps() + [("model", CoxPHSurvivalAnalysis(ties="breslow"))]), {"model__alpha": RIDGE_ALPHAS}, cv=inner, error_score=np.nan)
        gs.fit(X, y); return gs.best_estimator_, None
    ratios = (1.0,) if family == "lasso" else EN_RATIOS[:3]
    pre = Pipeline(base_steps()).fit(X); Z = pre.transform(X); grid = []
    for r in ratios:
        amax = float(CoxnetSurvivalAnalysis(l1_ratio=r, n_alphas=2, alpha_min_ratio=0.9, fit_baseline_model=False).fit(Z, y).alphas_[0])
        grid.append({"model__l1_ratio": [r], "model__alphas": [[float(a)] for a in np.geomspace(amax, amax * 0.02, 15)]})
    base = Pipeline(base_steps() + [("model", CoxnetSurvivalAnalysis(l1_ratio=1.0, alphas=[0.1], fit_baseline_model=False, max_iter=100000))])
    gs = GridSearchCV(base, grid, cv=inner, error_score=np.nan, refit=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore"); gs.fit(X, y)
    return gs.best_estimator_, None


def td_auc(y_all, risk, times=(365, 730, 1095)):
    try:
        return cumulative_dynamic_auc(y_all, y_all, risk, np.array(times, float))[0]
    except Exception:
        return np.full(len(times), np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default=str(ROOT / "Results/main_pipeline_claude/main_pipeline_recurrence/clinical/clinical_matrix.csv"))
    ap.add_argument("--tiers", default=str(ROOT / "Results/main_pipeline_claude/main_pipeline_recurrence/clinical/tiers.json"))
    ap.add_argument("--tier", default="pre")
    ap.add_argument("--families", nargs="+", default=["ridge", "lasso", "elastic_net", "size_afp", "size_afp_mvi"])
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--split", default=str(ROOT / "Results/main_pipeline_claude/main_pipeline_recurrence/split/recurrence_split.csv"))
    ap.add_argument("--output-parent", default=str(ROOT / "Results/main_pipeline_claude/main_pipeline_recurrence/clinical"))
    a = ap.parse_args(); warnings.filterwarnings("ignore")
    sp = pd.read_csv(a.split); dev = sp[sp["set"] == "dev"].reset_index(drop=True); ids = dev.PatientID.tolist()
    cols = json.load(open(a.tiers))[a.tier]
    X = pd.read_csv(a.matrix, index_col=0).loc[ids, cols]; Xn = X.to_numpy(float)
    y = Surv.from_arrays(event=dev.event.astype(bool).values, time=dev.time_days.astype(float).values, name_event="event", name_time="time")
    strata = dev.event.astype(int).astype(str) + "_" + dev.cohort
    run = Path(a.output_parent) / f"clinsurv_{a.tier}_{datetime.now():%Y%m%d_%H%M%S}"; run.mkdir(parents=True, exist_ok=False)
    (run / "manifest.json").write_text(json.dumps(dict(started=datetime.now().isoformat(), tier=a.tier, variables=cols, n_dev=len(ids), events=int(dev.event.sum()), lock_evaluated=False, repeats=a.repeats), indent=2))
    uni = []
    for k in cols:
        v = X[k].fillna(X[k].median()).values
        if np.nanstd(v) > 0:
            uni.append(dict(variable=k, c_index=concordance_index_censored(y["event"], y["time"], v)[0], missing=int(X[k].isna().sum())))
    U = pd.DataFrame(uni); U["abs_dev"] = (U.c_index - 0.5).abs(); U.sort_values("abs_dev", ascending=False).drop(columns="abs_dev").round(3).to_csv(run / "univariate_c_index.csv", index=False)
    print("univariate C-index (C>0.5: higher value = earlier recurrence):"); print(U.sort_values("abs_dev", ascending=False).drop(columns="abs_dev").head(10).round(3).to_string(index=False), flush=True)
    rows = []; oof_rows = []
    for fam in a.families:
        for rep in range(a.repeats):
            risk = np.full(len(ids), np.nan)
            try:
                for fold, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42 + rep).split(Xn, strata), 1):
                    inner = list(StratifiedKFold(3, shuffle=True, random_state=420 + fold + 1000 * rep).split(Xn[tr], dev.event.values[tr]))
                    est, keep = fit_model(fam, Xn[tr], y[tr], inner, cols)
                    risk[va] = est.predict(Xn[va][:, keep] if keep is not None else Xn[va])
            except (ValueError, RuntimeError, ArithmeticError, np.linalg.LinAlgError) as e:
                rows.append(dict(family=fam, repeat=rep, c_index=np.nan, failed_reason=str(e)[:100])); continue
            c = concordance_index_censored(y["event"], y["time"], risk)[0]; t = td_auc(y, risk)
            rows.append(dict(family=fam, repeat=rep, c_index=float(c), auc_1y=t[0], auc_2y=t[1], auc_3y=t[2]))
            oof_rows.append(pd.DataFrame(dict(family=fam, repeat=rep, PatientID=ids, risk=risk))); pd.concat(oof_rows).to_csv(run / "oof_risk.csv", index=False)
            pd.DataFrame(rows).to_csv(run / "per_repeat.csv", index=False)
        print(f"[{fam}] done", flush=True)
    D = pd.DataFrame(rows)
    S = D.groupby("family").agg(c_mean=("c_index", "mean"), c_sd=("c_index", "std"), c_max=("c_index", "max"), auc1=("auc_1y", "mean"), auc2=("auc_2y", "mean"), auc3=("auc_3y", "mean"),
                                n_ok=("c_index", "count"), n=("c_index", "size")).round(3)
    S.to_csv(run / "summary.csv"); print(S.to_string()); print("COMPLETED=" + str(run), flush=True)


if __name__ == "__main__":
    main()
