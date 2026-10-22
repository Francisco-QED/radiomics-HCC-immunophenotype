"""Presentation tables for time-to-recurrence models (110 dev / 30 lock): OOF performance, lock performance, SHAP values.

Categories (--tag): radiomics | radiomics_clinical (clinico-radiologic variables forced in) | clinical (clinico-radiologic only).
Per model (14 sequence combos x ridge / lasso / elastic-net Cox, or the clinical-only families):
  OOF (dev only): oof_risk.csv from the repeated 10 x 5-fold CV (strata = event x cohort). Reported: mean +- SD of the per-repeat pooled Harrell C; C-index of the repeat-averaged
    out-of-fold risk with a patient bootstrap 95% CI; time-dependent AUC at 1 / 2 / 3 years (IPCW censoring, patient bootstrap CI); sensitivity / specificity (Wilson CI) for
    2-year recurrence at the Youden threshold of the averaged OOF risk (patients with unknown 2-year status excluded).
  Lock (one shot): refit on ALL dev patients (inner 3-fold seed 999), apply without refitting to the 30 lock patients: C-index, 1/2/3-year AUC (bootstrap CIs),
    sensitivity / specificity at the DEV-derived threshold (Wilson CIs).
  SHAP: exact linear SHAP values for the Cox linear predictor (coef x centered standardized feature), saved as shap_values.npz; make_shap_plots.py draws them.
NOTE: the recurrence lock set was already opened exploratorily by another session on 2026-10-20; these lock numbers are therefore descriptive, not an independent test.
Run with /usr/bin/python3 (scikit-survival).
"""
import argparse, glob, json, sys, time, warnings
from datetime import datetime
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc
from sksurv.util import Surv
import repeated_survival_families as rsf

ROOT = rsf.ROOT
TIMES = np.array([365.0, 730.0, 1095.0])


def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def td_auc(y_train, y_test, risk):
    try:
        return np.asarray(cumulative_dynamic_auc(y_train, y_test, risk, TIMES)[0], float)
    except Exception:
        return np.full(len(TIMES), np.nan)


def cidx(y, risk):
    return float(concordance_index_censored(y["event"], y["time"], risk)[0])


def boot_metrics(y, risk, y_ref, n=1000, seed=20261020):
    """Patient bootstrap of C-index and 1/2/3-year AUC. IPCW censoring distribution from y_ref (dev)."""
    rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        i = rng.integers(0, len(risk), len(risk))
        if y["event"][i].sum() < 3: continue
        try: out.append(np.r_[cidx(y[i], risk[i]), td_auc(y_ref, y[i], risk[i])])
        except Exception: continue
    O = np.array(out); return np.nanpercentile(O, 2.5, axis=0), np.nanpercentile(O, 97.5, axis=0)


def youden(status, risk):
    fpr, tpr, thr = roc_curve(status, risk); return float(thr[int(np.argmax(tpr - fpr))])


def sens_spec(status, risk, thr):
    s = np.asarray(status); pred = np.asarray(risk) >= thr
    tp, fn, tn, fp = int((pred & (s == 1)).sum()), int((~pred & (s == 1)).sum()), int((~pred & (s == 0)).sum()), int((pred & (s == 0)).sum())
    return tp, tp + fn, tn, tn + fp


def shap_values_cox(est, Xdev, outdir):
    model = est.named_steps["model"]; pre = est[:-2].transform(Xdev); names = [c.replace("_original_", ":") for c in pre.columns]; Z = est.named_steps["scale"].transform(pre)
    coef = np.asarray(model.coef_).ravel(); keep = np.flatnonzero(np.abs(coef) > 1e-7); sv = coef[None, :] * (Z - Z.mean(0, keepdims=True))
    np.savez(outdir / "shap_values.npz", sv=sv[:, keep], Z=Z[:, keep], names=np.array(names)[keep], ids=np.array(Xdev.index.astype(str)))
    pd.DataFrame({"feature": np.array(names)[keep], "coefficient": coef[keep], "mean_abs_shap": np.abs(sv[:, keep]).mean(0)}).sort_values("mean_abs_shap", ascending=False).to_csv(outdir / "shap_coefficients.csv", index=False)
    return len(keep), int(sum(str(n).startswith("clin_") for n in np.array(names)[keep]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpt-dir", required=True)
    ap.add_argument("--tag", required=True, choices=["radiomics", "radiomics_clinical", "clinical"])
    ap.add_argument("--oof-glob", required=True, help="glob of oof_risk.csv files")
    ap.add_argument("--clinical-cols", nargs="*", default=None)
    ap.add_argument("--clinical-matrix", default=str(ROOT / "Results/main_pipeline_claude/main_pipeline_recurrence/clinical/clinical_matrix.csv"))
    ap.add_argument("--out", default=str(ROOT / "Results/main_pipeline_claude/presentation_20261020/recurrence"))
    ap.add_argument("--sets", nargs="*", default=None)
    ap.add_argument("--families", nargs="+", default=None)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--sheet", choices=["pre", "post"], default="post")
    a = ap.parse_args(); warnings.filterwarnings("ignore")
    sys.path.insert(0, a.gpt_dir)
    import run_compact_ridge_sequence_search as core
    Sel = rsf.make_selector(core)
    sp = pd.read_csv(ROOT / "Results/main_pipeline_claude/main_pipeline_recurrence/split/recurrence_split.csv").set_index("PatientID")
    dev = sp[sp["set"] == "dev"]; lock = sp[sp["set"] == "lock"]; dev_ids, lock_ids = dev.index.tolist(), lock.index.tolist()
    mk = lambda d: Surv.from_arrays(event=d.event.astype(bool).values, time=d.time_days.astype(float).values, name_event="event", name_time="time")
    yd, yl = mk(dev), mk(lock)
    r2 = pd.read_excel(ROOT / "databases/recurrence_outcomes_v3.xlsx").set_index("PatientID")["recur_2yr"]
    st_d, st_l = r2.reindex(dev_ids).values, r2.reindex(lock_ids).values
    import run_guard
    oof = pd.concat([pd.read_csv(f) for f in run_guard.usable_files(a.oof_glob)])
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); (out / "README.txt").write_text(__doc__)
    strata_d = dev.event.astype(int).astype(str) + "_" + dev.cohort
    M = pd.read_csv(a.clinical_matrix, index_col=0)
    clin = M[a.clinical_cols].add_prefix("clin_") if a.clinical_cols else None
    sheet = ROOT / ("Results/Radiomics_sheets/SA1_2_post/R-SA1_2-post_I-sitkBSpline_S-1mm_W-perseq.csv" if a.sheet == "post" else "Results/Radiomics_sheets/SA1_2_pre/R-SA1_2_I-sitkBSpline_S-1mm_W-perseq.csv")
    Xall = pd.read_csv(sheet, index_col=0, usecols=lambda c: c == "PatientID" or "_original_" in c); Xall = Xall.reindex(sorted(Xall.columns), axis=1).astype(float)
    readers = [str(ROOT / f"Results/Radiomics_sheets/inter-reader/Reader{i}/R-Reader{i}_I-sitkBSpline_S-1mm_W-perseq.csv") for i in (1, 2)]
    RealICC = core.TechnicalICCFilter; rows = []
    combos = a.sets or sorted(oof.sequence_set.unique()) if "sequence_set" in oof else ["clinical"]
    fams = a.families or sorted(oof.family.unique())
    for sname in combos:
        if a.tag == "clinical":
            Xs_dev, Xs_lock, forced = M.loc[dev_ids, a.clinical_cols].astype(float), M.loc[lock_ids, a.clinical_cols].astype(float), ()
        else:
            cols = sorted(c for c in Xall.columns if any(c.startswith(s + "_original_") for s in sname.split("+")))
            fixed = list(RealICC(*readers).fit(Xall.loc[dev_ids, cols]).columns_)
            Xs_dev, Xs_lock = Xall.loc[dev_ids, fixed], Xall.loc[lock_ids, fixed]; forced = ()
            if clin is not None:
                Xs_dev, Xs_lock = pd.concat([Xs_dev, clin.loc[dev_ids]], axis=1), pd.concat([Xs_lock, clin.loc[lock_ids]], axis=1); forced = tuple(clin.columns)
        for fam in fams:
            t0 = time.time(); o = oof[(oof.family == fam) & oof.risk.notna()]
            if "sequence_set" in o: o = o[o.sequence_set == sname]
            if o.empty: continue
            pos = {pid: k for k, pid in enumerate(dev_ids)}
            per = np.array([cidx(yd[[pos[i] for i in d.PatientID]], d.risk.values) for _, d in o.groupby("repeat")])
            avg = o.groupby("PatientID").risk.mean().reindex(dev_ids).values
            c_avg = cidx(yd, avg); a_avg = td_auc(yd, yd, avg); blo, bhi = boot_metrics(yd, avg, yd, n=a.boot)
            known = ~pd.isna(st_d); thr = youden(st_d[known].astype(int), avg[known]); stp, stn, spt, spn = sens_spec(st_d[known].astype(int), avg[known], thr)
            inner = list(StratifiedKFold(3, shuffle=True, random_state=999).split(Xs_dev, yd["event"]))
            try:
                est, params, score = (rsf.fit_ridge(core, Sel, Xs_dev, yd, inner, forced, 1) if fam == "ridge" else rsf.fit_sparse(core, Sel, fam, Xs_dev, yd, inner, forced, 1)) if a.tag != "clinical" else (None, None, None)
                if a.tag == "clinical":
                    import clinical_survival as cs
                    est, _ = cs.fit_model(fam, Xs_dev.to_numpy(float), yd, inner, list(Xs_dev.columns)); XsD, XsL = Xs_dev.to_numpy(float), Xs_lock.to_numpy(float); params = {}
                    risk_l = est.predict(XsL)
                    Z = est.named_steps["scale"].transform(est.named_steps["impute"].transform(XsD)); coef = np.asarray(est.named_steps["model"].coef_).ravel(); keep = np.flatnonzero(np.abs(coef) > 1e-7)
                    sv = coef[None, :] * (Z - Z.mean(0, keepdims=True)); mdir = out / "models" / a.tag / f"clinicoradiologic__{fam}"; mdir.mkdir(parents=True, exist_ok=True)
                    np.savez(mdir / "shap_values.npz", sv=sv[:, keep], Z=Z[:, keep], names=np.array(Xs_dev.columns)[keep], ids=np.array(dev_ids))
                    pd.DataFrame({"feature": np.array(Xs_dev.columns)[keep], "coefficient": coef[keep], "mean_abs_shap": np.abs(sv[:, keep]).mean(0)}).sort_values("mean_abs_shap", ascending=False).to_csv(mdir / "shap_coefficients.csv", index=False)
                    nf, nc = len(keep), len(keep)
                else:
                    risk_l = est.predict(Xs_lock); mdir = out / "models" / a.tag / f"{sname.replace('+', '_')}__{fam}"; mdir.mkdir(parents=True, exist_ok=True); nf, nc = shap_values_cox(est, Xs_dev, mdir)
            except (ValueError, RuntimeError, ArithmeticError) as e:
                print(f"[{sname}|{fam}] final fit failed: {str(e)[:80]}", flush=True); continue
            c_l = cidx(yl, risk_l); a_l = td_auc(yd, yl, risk_l); llo, lhi = boot_metrics(yl, risk_l, yd, n=a.boot)
            kl = ~pd.isna(st_l); ltp, ltn, lsp, lsn = sens_spec(st_l[kl].astype(int), risk_l[kl], thr)
            pd.DataFrame({"PatientID": lock_ids, "event": yl["event"].astype(int), "time_days": yl["time"], "risk": risk_l}).to_csv(mdir / "lock_predictions.csv", index=False)
            r = dict(sequence_set=sname, family=fam, n_repeats_ok=len(per), oof_c_repeat_mean=float(per.mean()), oof_c_repeat_sd=float(per.std(ddof=1)), oof_c_avg=c_avg, oof_c_ci_lo=blo[0], oof_c_ci_hi=bhi[0])
            for k, y_ in enumerate((1, 2, 3)): r.update({f"oof_auc{y_}y": a_avg[k], f"oof_auc{y_}y_ci_lo": blo[k + 1], f"oof_auc{y_}y_ci_hi": bhi[k + 1], f"lock_auc{y_}y": a_l[k], f"lock_auc{y_}y_ci_lo": llo[k + 1], f"lock_auc{y_}y_ci_hi": lhi[k + 1]})
            r.update(dict(threshold_youden_dev_oof=thr, oof_sens=stp / stn, oof_sens_ci_lo=wilson(stp, stn)[0], oof_sens_ci_hi=wilson(stp, stn)[1], oof_spec=spt / spn, oof_spec_ci_lo=wilson(spt, spn)[0], oof_spec_ci_hi=wilson(spt, spn)[1],
                          lock_c=c_l, lock_c_ci_lo=llo[0], lock_c_ci_hi=lhi[0], lock_sens=ltp / ltn, lock_sens_ci_lo=wilson(ltp, ltn)[0], lock_sens_ci_hi=wilson(ltp, ltn)[1],
                          lock_spec=lsp / lsn, lock_spec_ci_lo=wilson(lsp, lsn)[0], lock_spec_ci_hi=wilson(lsp, lsn)[1], n_features=nf, n_clinical_in_model=nc, n_2yr_known_dev=int(stn + spn), n_2yr_known_lock=int(ltn + lsn),
                          params=json.dumps({k: float(v) for k, v in (params or {}).items()}), seconds=round(time.time() - t0)))
            rows.append(r); pd.DataFrame(rows).to_csv(out / f"recurrence_model_table_{a.tag}.csv", index=False)
            print(f"[{sname}|{fam}] OOF C {r['oof_c_repeat_mean']:.3f} | lock C {c_l:.3f} ({llo[0]:.2f}-{lhi[0]:.2f}) | 2y AUC lock {a_l[1]:.2f}", flush=True)
    (out / f"manifest_{a.tag}.json").write_text(json.dumps(dict(finished=datetime.now().isoformat(), n_dev=len(dev_ids), n_lock=len(lock_ids), lock_exploratory_note="lock set opened earlier by another session", n_models=len(rows)), indent=2)); print("COMPLETED=" + str(out), flush=True)


if __name__ == "__main__":
    main()
