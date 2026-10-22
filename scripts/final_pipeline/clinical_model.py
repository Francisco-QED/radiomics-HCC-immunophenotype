"""Clinical + qualitative-MRI model, built to mimic the radiomics models as closely as possible.

Same cohort (140 scored patients), same dev/lock split (only the 110 dev patients are used; lock outcomes never read),
same endpoint (TILs >= 2), same outer folds (5-fold stratified, seed 42+repeat), same inner folds (3-fold, 420+fold+1000*repeat),
same model families (ridge / lasso / elastic net via LogisticRegression, same C grids) and the same pooled-OOF-AUC metric with bootstrap CI.
Differences forced by the data type: no ICC/clustering/top-k (only ~20 variables); median imputation + z-scoring inside every fold;
the radiomics 10-20 nonzero-coefficient constraint is not applied (it is meaningless for ~20 inputs).
Source: mastersheet.xlsx tabs SA1 and SA2-post (T7). Pathology / resection / recurrence fields are NOT used (known only at resection).

Variable policy (stated in advance): binary variables are kept only if the minority level is >= 10% of dev patients
(drops enhancement, morphology, restricted diffusion, macrovascular invasion); necrosis is dropped (almost always 'none');
race/ethnicity/anatomical location are not modelled (sparse free-text levels). Treatment is a 3-level categorical
(none / IO / IO+SBRT, reference none) and is kept regardless of prevalence. NOTE: treatment == cohort (SA1 untreated, SA2 treated).
"""
import argparse, json, os, sys, warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path("/Users/Francisco/Library/CloudStorage/OneDrive-TheMountSinaiHospital/Regeneron retrospective radiomics")
MASTER = "/Volumes/T7 Shield/Restrepo/HCC_Retrospective_radiomics/databases/mastersheet.xlsx"
C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0)                       # ridge grid used by the radiomics models
SPARSE_C = tuple(float(x) for x in np.geomspace(0.05, 6.0, 17))  # lasso / elastic-net grid used by the radiomics models
EN_RATIOS = (0.25, 0.50, 0.75)
FIXED_C = 0.01                                               # fixed, pre-stated ridge strength (no tuning)
POS = dict(age=2, gender=3, bmi=4, etiology=7, cirrhosis=8, afp=9, treatment=10, capsule=27, margin=28, mosaic=29,
           internal_arteries=30, fat=33, t2=34, tumor_size=38, hbp=39,
           morphology=26, macrovascular=31, necrosis=32, restricted_diffusion=35, enhancement=36, lirads=37)


def load_tab(tab):
    raw = pd.read_excel(MASTER, sheet_name=tab, header=None)
    body = raw.iloc[2:]
    body = body[body.iloc[:, 0].astype(str).str.match(r"^P\d{3}$")]
    body.index = body.iloc[:, 0].astype(str)
    return body


def yn(s):
    t = s.astype(str).str.strip().str.lower()
    return pd.Series(np.where(t.str.startswith("y"), 1.0, np.where(t.str.startswith("n") & ~t.eq("nan"), 0.0, np.nan)), index=s.index)


def encode(sa1, sa2):
    both = pd.concat([sa1.assign(_cohort="SA1"), sa2.assign(_cohort="SA2")])
    g = lambda k: both.iloc[:, POS[k]]
    X = pd.DataFrame(index=both.index)
    X["Age"] = pd.to_numeric(g("age"), errors="coerce")
    X["Gender_M"] = g("gender").astype(str).str.strip().str.upper().eq("M").astype(float)
    X["BMI"] = pd.to_numeric(g("bmi"), errors="coerce")
    X["Cirrhosis_Y"] = yn(g("cirrhosis"))
    X["AFP_log"] = np.log1p(pd.to_numeric(g("afp"), errors="coerce"))
    et = g("etiology").astype(str).str.lower()
    grp = np.where(et.str.contains("hbv|hcv"), "Viral", np.where(et.str.contains("mash|nash|masld|nafld|steatosis"), "MASLD", "Other"))
    X["Etiology_Viral"] = (grp == "Viral").astype(float)
    X["Etiology_MASLD"] = (grp == "MASLD").astype(float)
    tr = g("treatment").astype(str).str.strip().str.upper()
    X["Treatment_IO"] = tr.eq("IO").astype(float)
    X["Treatment_IO+SBRT"] = tr.eq("IO+SBRT").astype(float)
    X["Capsule_Y"] = yn(g("capsule"))
    X["Mosaic_Y"] = yn(g("mosaic"))
    X["InternalArteries_Y"] = yn(g("internal_arteries"))
    X["FatContent_Y"] = yn(g("fat"))
    t2 = g("t2").astype(str).str.strip().str.lower()
    X["T2_Hyper"] = np.where(t2.isin(["iso", "hypo"]), 0.0, np.where(t2.str.contains("mild|hyper|marked"), 1.0, np.nan))
    mg = g("margin").astype(str).str.strip().str.lower()
    X["Margin_IrregularOrInfiltrative"] = np.where(mg.str.startswith("well"), 0.0, np.where(mg.str.startswith(("irreg", "infil", "inflit")), 1.0, np.nan))
    X["TumorSize_PVP_log"] = np.log(pd.to_numeric(g("tumor_size"), errors="coerce"))
    hb = g("hbp").astype(str).str.strip().str.lower()
    X["HBP_NotHypo"] = np.where(hb.eq("hypo"), 0.0, np.where(hb.isin(["iso", "hyper", "mixed"]), 1.0, np.nan))
    # candidates that are evaluated by the prevalence rule (or dropped by decision) so the evidence is in variables_table.csv
    mo = g("morphology").astype(str).str.strip().str.lower()
    X["Morphology_Infiltrative"] = np.where(mo.str.startswith("infil"), 1.0, np.where(mo.eq("nodular"), 0.0, np.nan))
    X["MacrovascularInvasion_Y"] = yn(g("macrovascular"))
    nec = g("necrosis").astype(str).str.strip().str.lower(); nv = pd.to_numeric(nec, errors="coerce")
    X["Necrosis_Any"] = np.where(nec.eq("yes") | (nv > 0), 1.0, np.where(nec.eq("no") | (nv == 0), 0.0, np.nan))
    rd = g("restricted_diffusion").astype(str).str.strip().str.lower()
    X["RestrictedDiffusion_No"] = np.where(rd.eq("yes"), 0.0, np.where(rd.eq("no"), 1.0, np.nan))
    en = g("enhancement").astype(str).str.strip().str.lower()
    X["Enhancement_NotHyper"] = np.where(en.eq("hyper"), 0.0, np.where(en.isin(["hypo", "iso"]), 1.0, np.nan))
    lr = pd.to_numeric(g("lirads"), errors="coerce")
    X["LIRADS_not5"] = np.where(lr == 5, 0.0, np.where(lr < 5, 1.0, np.nan))
    return X, both["_cohort"]


def ridge(C):
    return LogisticRegression(l1_ratio=0, C=C, solver="lbfgs", max_iter=20000, tol=1e-5, random_state=42)


def sparse(C=1.0, r=1.0):
    return LogisticRegression(C=C, l1_ratio=r, solver="saga", max_iter=100000, tol=1e-4, random_state=42)


def pipe(model):
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", model)])


def metrics(y, p):
    return dict(auc=float(roc_auc_score(y, p)), average_precision=float(average_precision_score(y, p)), brier=float(brier_score_loss(y, p)))


def boot_ci(y, p, n=2000, seed=20261016):
    rng = np.random.default_rng(seed); y = np.asarray(y); p = np.asarray(p)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0); v = []
    for _ in range(n):
        i = np.r_[rng.choice(neg, len(neg)), rng.choice(pos, len(pos))]; v.append(roc_auc_score(y[i], p[i]))
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def fit_predict_nested(X, y, family, outer, rep, jobs):
    pred = np.full(len(y), np.nan)
    for fold, (tr, va) in enumerate(outer, 1):
        inner = list(StratifiedKFold(3, shuffle=True, random_state=420 + fold + 1000 * rep).split(X.iloc[tr], y.iloc[tr]))
        if family == "ridge":
            gs = GridSearchCV(pipe(ridge(1.0)), {"model__C": C_GRID}, scoring="roc_auc", cv=inner, n_jobs=jobs, refit=True)
        else:
            ratios = (1.0,) if family == "lasso" else EN_RATIOS
            gs = GridSearchCV(pipe(sparse()), {"model__C": SPARSE_C, "model__l1_ratio": ratios}, scoring="roc_auc", cv=inner, n_jobs=jobs, refit=True)
        gs.fit(X.iloc[tr], y.iloc[tr]); pred[va] = gs.predict_proba(X.iloc[va])[:, 1]
    return pred


def fit_predict_fixed(X, y, outer, C=FIXED_C, yfit=None):
    pred = np.full(len(y), np.nan); yfit = y if yfit is None else yfit
    for tr, va in outer:
        m = pipe(ridge(C)).fit(X.iloc[tr], yfit.iloc[tr]); pred[va] = m.predict_proba(X.iloc[va])[:, 1]
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-parent", default=str(ROOT / "Results/main_pipeline_clinical"))
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--perms", type=int, default=1000)
    ap.add_argument("--jobs", type=int, default=3)
    a = ap.parse_args(); warnings.filterwarnings("ignore")
    run = Path(a.output_parent) / f"clinical_qualitative_models_{datetime.now():%Y%m%d_%H%M%S}"
    run.mkdir(parents=True, exist_ok=False)

    split = pd.read_csv(ROOT / "Results/main_pipeline_claude/main_pipeline/final/cohort_v2_sa2pre/split/split.csv", usecols=["PatientID", "set", "TILs_resection"])
    dev = split[split["set"] == "dev"]                              # lock rows are never touched
    dev_ids = dev.PatientID.tolist(); y = (dev.set_index("PatientID").TILs_resection.loc[dev_ids] >= 2).astype(int)
    assert len(dev_ids) == 110 and int(y.sum()) == 89, (len(dev_ids), int(y.sum()))
    Xall, cohort = encode(load_tab("SA1"), load_tab("SA2-post"))
    X = Xall.loc[dev_ids]; coh = cohort.loc[dev_ids]

    # ---- variable policy (prevalence rule on dev) -------------------------------------------------------------------
    binary = [c for c in X.columns if set(X[c].dropna().unique()) <= {0.0, 1.0}]
    always_keep = {"Treatment_IO", "Treatment_IO+SBRT", "Etiology_Viral", "Etiology_MASLD"}   # levels of categorical variables
    forced_drop = {"Necrosis_Any": "dropped by decision: necrosis is 'none' for most patients"}
    rows = []
    for c in X.columns:
        is_bin = c in binary
        prev = float(X[c].mean()) if is_bin else np.nan
        minor = min(prev, 1 - prev) if is_bin else np.nan
        keep = ((not is_bin) or (c in always_keep) or (minor >= 0.10)) and c not in forced_drop
        reason = forced_drop.get(c, "kept" if keep else "dropped by rule: minority level < 10% of dev")
        auc = np.nan
        ok = X[c].notna()
        if ok.sum() > 10 and X.loc[ok, c].nunique() > 1: auc = roc_auc_score(y[ok], X.loc[ok, c])
        rows.append(dict(variable=c, type="binary" if is_bin else "continuous", n_missing=int(X[c].isna().sum()), prevalence_or_mean=round(prev if is_bin else float(X[c].mean()), 3),
                         minority_fraction=round(minor, 3) if is_bin else np.nan, kept=bool(keep), reason=reason, univariate_auc_dev=round(float(auc), 3) if np.isfinite(auc) else np.nan))
    vt = pd.DataFrame(rows); vt.to_csv(run / "variables_table.csv", index=False)
    kept = vt[vt.kept].variable.tolist(); dropped = vt[~vt.kept].variable.tolist()
    clinical = [c for c in kept if c in ("Age", "Gender_M", "BMI", "Cirrhosis_Y", "AFP_log", "Etiology_Viral", "Etiology_MASLD", "Treatment_IO", "Treatment_IO+SBRT")]
    SETS = {"clinical": clinical, "clinical+qualitative_MRI": kept}
    print("kept:", kept, "\ndropped:", vt[~vt.kept][["variable", "minority_fraction", "reason"]].values.tolist(), flush=True)
    X[kept].assign(cohort=coh, TILs_ge2=y).to_csv(run / "model_matrix_dev.csv")

    # ---- E1: nested CV (mimics the radiomics models), repeated ------------------------------------------------------
    allpred, summ = {}, []
    for sname, cols in SETS.items():
        Xs = X[cols]
        for fam in ("ridge", "lasso", "elastic_net"):
            for rep in range(a.repeats):
                outer = list(StratifiedKFold(5, shuffle=True, random_state=42 + rep).split(Xs, y))
                p = fit_predict_nested(Xs, y, fam, outer, rep, a.jobs); allpred[(sname, fam, rep)] = p
                summ.append(dict(variable_set=sname, family=fam, repeat=rep, n_variables=len(cols), **metrics(y, p)))
            print(f"[nested] {sname} / {fam}: mean AUC {np.mean([r['auc'] for r in summ if r['variable_set']==sname and r['family']==fam]):.3f}", flush=True)
    S = pd.DataFrame(summ); S.to_csv(run / "nested_repeated_per_repeat.csv", index=False)
    agg = S.groupby(["variable_set", "family"]).auc.agg(mean_auc="mean", sd_auc="std", min_auc="min", max_auc="max").reset_index()
    ci = []
    for (sname, fam), _ in agg.groupby(["variable_set", "family"]):
        lo, hi = boot_ci(y, allpred[(sname, fam, 0)]); ci.append((sname, fam, lo, hi, S[(S.variable_set == sname) & (S.family == fam) & (S.repeat == 0)].auc.iloc[0]))
    agg = agg.merge(pd.DataFrame(ci, columns=["variable_set", "family", "repeat0_ci_low", "repeat0_ci_high", "repeat0_auc"]), on=["variable_set", "family"])
    agg.round(4).to_csv(run / "nested_repeated_summary.csv", index=False)

    # ---- E2: fixed ridge (no tuning), repeated --------------------------------------------------------------------
    fx, fxpred = [], {}
    for sname, cols in SETS.items():
        for rep in range(a.repeats):
            outer = list(StratifiedKFold(5, shuffle=True, random_state=42 + rep).split(X[cols], y))
            p = fit_predict_fixed(X[cols], y, outer); fxpred[(sname, rep)] = p
            fx.append(dict(variable_set=sname, repeat=rep, **metrics(y, p),
                           auc_SA1=roc_auc_score(y[coh == "SA1"], p[(coh == "SA1").values]), auc_SA2=roc_auc_score(y[coh == "SA2"], p[(coh == "SA2").values])))
    F = pd.DataFrame(fx); F.to_csv(run / "fixed_ridge_per_repeat.csv", index=False)
    Fa = F.groupby("variable_set")[["auc", "auc_SA1", "auc_SA2"]].agg(["mean", "std"]).round(4); Fa.to_csv(run / "fixed_ridge_summary.csv")

    # ---- E3: fold-blocked permutation null for fixed ridge (seed-42 folds) -------------------------------------------------
    outer0 = {s: list(StratifiedKFold(5, shuffle=True, random_state=42).split(X[c], y)) for s, c in SETS.items()}
    rng = np.random.default_rng(20261016); perm_rows = []
    for sname, cols in SETS.items():
        obs = roc_auc_score(y, fxpred[(sname, 0)]); null = []
        for _ in range(a.perms):
            yp = y.copy()
            for _, va in outer0[sname]:                       # shuffle labels within each validation block; folds unchanged
                yp.iloc[va] = rng.permutation(y.iloc[va].values)
            null.append(roc_auc_score(yp, fit_predict_fixed(X[cols], yp, outer0[sname], yfit=yp)))
        null = np.array(null)
        perm_rows.append(dict(variable_set=sname, observed_auc=obs, null_mean=null.mean(), null_sd=null.std(ddof=1), null_95th=np.percentile(null, 95), null_max=null.max(),
                              n_perm=a.perms, p_one_sided=(1 + (null >= obs).sum()) / (1 + a.perms)))
        print(f"[perm] {sname}: obs {obs:.3f}  null {null.mean():.3f} +- {null.std(ddof=1):.3f}  p={perm_rows[-1]['p_one_sided']:.4f}", flush=True)
    pd.DataFrame(perm_rows).round(4).to_csv(run / "permutation_null_fixed_ridge.csv", index=False)

    # ---- E4: coefficients of the fixed ridge refit on all dev patients -----------------------------------------------------
    co = []
    for sname, cols in SETS.items():
        m = pipe(ridge(FIXED_C)).fit(X[cols], y); b = m.named_steps["model"].coef_.ravel()
        co += [dict(variable_set=sname, variable=c, std_coefficient=round(float(v), 4)) for c, v in zip(cols, b)]
    pd.DataFrame(co).to_csv(run / "ridge_coefficients_full_dev.csv", index=False)

    (run / "manifest.json").write_text(json.dumps(dict(finished=datetime.now().isoformat(), endpoint="TILs>=2", n_dev=110, lock_evaluated=False, repeats=a.repeats, perms=a.perms,
        kept=kept, dropped=dropped, sets=SETS, sklearn=sklearn.__version__, mastersheet=MASTER, fixed_C=FIXED_C), indent=2))
    print("COMPLETED=" + str(run), flush=True)


if __name__ == "__main__":
    main()
