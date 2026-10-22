"""Clinico-radiologic-only model for TILs >= 2 on the current pool (111 dev / 30 lock), same protocol and output format as the radiomics ladder.

Variables: the pre-operative clinico-radiologic columns in clinical/tils_cols.json (binary variables with minority < 10% in dev dropped; cohort = treatment categorical).
Models: ridge / lasso / elastic-net logistic regression (inner 3-fold AUC tuning with the radiomics C grids; no 10-20 support constraint: only ~16 inputs).
Repeated CV: 10 x 5-fold, strata = outcome x cohort, outer seed 42+repeat (identical folds to the radiomics ladder). OOF predictions saved.
Then one-shot: refit on all dev (inner seed 999), apply to lock, sensitivity/specificity at the dev-OOF Youden threshold, bootstrap/Wilson CIs, SHAP values.
Run with /usr/local/bin/python3 (scikit-learn >= 1.8).
"""
import argparse, json, sys, warnings
from datetime import datetime
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import clinical_model as cm
import lock_report_tils as lr

ROOT = lr.ROOT


def final_fit(X, y, family):
    inner = list(StratifiedKFold(3, shuffle=True, random_state=999).split(X, y))
    if family == "ridge":
        gs = GridSearchCV(cm.pipe(cm.ridge(1.0)), {"model__C": cm.C_GRID}, scoring="roc_auc", cv=inner, refit=True)
    else:
        gs = GridSearchCV(cm.pipe(cm.sparse()), {"model__C": cm.SPARSE_C, "model__l1_ratio": (1.0,) if family == "lasso" else cm.EN_RATIOS}, scoring="roc_auc", cv=inner, refit=True)
    gs.fit(X, y); return gs.best_estimator_, gs.best_params_


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default=str(ROOT / "Results/main_pipeline_claude/main_pipeline_recurrence/clinical/clinical_matrix.csv"))
    ap.add_argument("--col-key", default="tils_pre", help="key in the cols json (tils_pre | tils_pre_grade)")
    ap.add_argument("--cols", default=str(ROOT / "Results/main_pipeline_claude/main_pipeline_recurrence/clinical/tils_cols.json"))
    ap.add_argument("--out", default=str(ROOT / "Results/main_pipeline_claude/presentation_20261020/tils_ge2"))
    ap.add_argument("--endpoint", choices=["tils_ge2", "tils_ge3", "tls_ge2"], default="tils_ge2")
    ap.add_argument("--oof-dir", default=str(ROOT / "Results/main_pipeline_claude/main_pipeline_tils/clinical_only_v4_20261021"))
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--jobs", type=int, default=2)
    a = ap.parse_args(); warnings.filterwarnings("ignore")
    sp = pd.read_csv(ROOT / "Results/main_pipeline_claude/main_pipeline/final/cohort_v3_141/split/split.csv").set_index("PatientID")
    dev = sp[sp["set"] == "dev"].index.tolist(); lock = sorted(sp[sp["set"] == "lock"].index)
    M = pd.read_csv(a.matrix, index_col=0); cols = json.load(open(a.cols))[a.col_key]
    X, XL = M.loc[dev, cols], M.loc[lock, cols]
    y = (sp.loc[dev, "TILs_resection"] >= 2).astype(int); yL = (sp.loc[lock, "TILs_resection"] >= 2).astype(int)
    if a.endpoint == "tils_ge3":
        y = (sp.loc[dev, "TILs_resection"] >= 3).astype(int); yL = (sp.loc[lock, "TILs_resection"] >= 3).astype(int)
    if a.endpoint == "tls_ge2":
        tls = pd.read_excel(ROOT / "databases/immunophenotype_outcomes_v4.xlsx").set_index("PatientID")["TLS_resection"]; y = (tls.loc[dev] >= 2).astype(int); yL = (tls.loc[lock] >= 2).astype(int)
    strata = y.astype(str) + "_" + M.loc[dev, "cohort_sa2"].map({0.0: "SA1", 1.0: "SA2"})
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); od = Path(a.oof_dir); od.mkdir(parents=True, exist_ok=True)
    print(f"dev {X.shape} pos {int(y.sum())} | lock {XL.shape} pos {int(yL.sum())} | variables {len(cols)}", flush=True)
    rows, oof_rows = [], []
    for fam in ("ridge", "lasso", "elastic_net"):
        per = []
        for rep in range(a.repeats):
            outer = list(StratifiedKFold(5, shuffle=True, random_state=42 + rep).split(X, strata))
            pred = cm.fit_predict_nested(X, y, fam, outer, rep, a.jobs); per.append(roc_auc_score(y, pred))
            oof_rows.append(pd.DataFrame(dict(sequence_set="clinicoradiologic", family=fam, repeat=rep, PatientID=y.index, y=y.values, prob=pred)))
            pd.concat(oof_rows).to_csv(od / "oof_predictions.csv", index=False)
            print(f"[clinical | {fam}] repeat {rep}: AUC {per[-1]:.3f}", flush=True)
        o = pd.concat([d for d in oof_rows if d.family.iloc[0] == fam]); avg = o.groupby("PatientID").agg(y=("y", "first"), prob=("prob", "mean")).loc[dev]
        thr = lr.youden(avg.y, avg.prob); lo, hi = lr.boot_auc(avg.y, avg.prob); stp, stn, spt, spn = lr.sens_spec(avg.y, avg.prob, thr)
        est, params = final_fit(X, y, fam); p = est.predict_proba(XL)[:, 1]; llo, lhi = lr.boot_auc(yL, p); ltp, ltn, lsp, lsn = lr.sens_spec(yL, p, thr)
        model = est.named_steps["model"]; Z = est.named_steps["scale"].transform(est.named_steps["impute"].transform(X)); coef = np.asarray(model.coef_).ravel()
        keep = np.flatnonzero(np.abs(coef) > 1e-7); sv = coef[None, :] * (Z - Z.mean(0, keepdims=True))
        mdir = out / "models" / "clinical" / f"clinicoradiologic__{fam}"; mdir.mkdir(parents=True, exist_ok=True)
        np.savez(mdir / "shap_values.npz", sv=sv[:, keep], Z=Z[:, keep], names=np.array(cols)[keep], ids=np.array(X.index.astype(str)))
        pd.DataFrame({"feature": np.array(cols)[keep], "standardized_coefficient": coef[keep], "mean_abs_shap": np.abs(sv[:, keep]).mean(0)}).sort_values("mean_abs_shap", ascending=False).to_csv(mdir / "shap_coefficients.csv", index=False)
        pd.DataFrame({"PatientID": lock, "y_lock": yL.values, "prob": p}).to_csv(mdir / "lock_predictions.csv", index=False)
        rows.append(dict(sequence_set="clinicoradiologic", n_sequences=0, family=fam, n_repeats_ok=len(per), oof_auc_repeat_mean=float(np.mean(per)), oof_auc_repeat_sd=float(np.std(per, ddof=1)),
                         oof_auc_avgprob=roc_auc_score(avg.y, avg.prob), oof_auc_ci_lo=lo, oof_auc_ci_hi=hi, threshold_youden_dev_oof=thr,
                         oof_sens=stp / stn, oof_sens_ci_lo=lr.wilson(stp, stn)[0], oof_sens_ci_hi=lr.wilson(stp, stn)[1], oof_spec=spt / spn, oof_spec_ci_lo=lr.wilson(spt, spn)[0], oof_spec_ci_hi=lr.wilson(spt, spn)[1],
                         lock_auc=roc_auc_score(yL, p), lock_auc_ci_lo=llo, lock_auc_ci_hi=lhi, lock_sens=ltp / ltn, lock_sens_ci_lo=lr.wilson(ltp, ltn)[0], lock_sens_ci_hi=lr.wilson(ltp, ltn)[1],
                         lock_spec=lsp / lsn, lock_spec_ci_lo=lr.wilson(lsp, lsn)[0], lock_spec_ci_hi=lr.wilson(lsp, lsn)[1], lock_ap=average_precision_score(yL, p), lock_brier=brier_score_loss(yL, p),
                         n_features=len(keep), n_clinical_in_model=len(keep), params=json.dumps({k: float(v) for k, v in params.items()})))
        pd.DataFrame(rows).to_csv(out / f"{a.endpoint}_model_table_clinical.csv", index=False)
        print(f"[clinical | {fam}] OOF {np.mean(per):.3f} | lock AUC {rows[-1]['lock_auc']:.3f} ({llo:.2f}-{lhi:.2f}) sens {rows[-1]['lock_sens']:.2f} spec {rows[-1]['lock_spec']:.2f}", flush=True)
    (od / "manifest.json").write_text(json.dumps(dict(finished=datetime.now().isoformat(), variables=cols, n_dev=len(X), lock_labels_read=True), indent=2)); print("COMPLETED", flush=True)


if __name__ == "__main__":
    main()
