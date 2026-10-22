"""Presentation tables for the TILs >= 2 sequence ladder on the CURRENT pool (111 dev / 30 lock): OOF performance, one-shot lock performance, SHAP.

Per model (14 sequence sets x ridge / lasso / elastic net):
  OOF (dev only): read oof_predictions.csv from the repeated-CV ladder (10 repeats of 5-fold, strata = outcome x cohort, global reader-ICC). Reported: mean +- SD of the
    per-repeat pooled AUC, the AUC of the repeat-averaged out-of-fold probability with a stratified patient bootstrap 95% CI (2000 resamples), and the
    Youden threshold of that averaged OOF probability. OOF sensitivity / specificity at that threshold with Wilson 95% CIs.
  Lock (one shot): refit on ALL dev patients (inner 3-fold seed 999), apply WITHOUT refitting to the 30 lock patients; AUC with stratified bootstrap 95% CI;
    sensitivity / specificity at the DEV-derived Youden threshold (never re-tuned on lock) with Wilson 95% CIs; also AP and Brier.
  SHAP: exact linear SHAP values (coef x centered standardized feature) on the final dev-fit model, dev patients, saved as shap_values.npz; plots are drawn by make_shap_plots.py.
All models are reported; none is selected. Positive class = TILs >= 2.
"""
import argparse, glob, json, sys, time, warnings
from datetime import datetime
from pathlib import Path
sys.dont_write_bytecode = True
import numpy as np, pandas as pd, joblib, sklearn
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

ROOT = Path("/Users/Francisco/Library/CloudStorage/OneDrive-TheMountSinaiHospital/Regeneron retrospective radiomics")
CORE4 = ["tp", "pvp", "dwi_b2", "hbp"]


def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def boot_auc(y, p, n=2000, seed=20261020):
    rng = np.random.default_rng(seed); y = np.asarray(y); p = np.asarray(p); pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    v = [roc_auc_score(y[i], p[i]) for i in (np.r_[rng.choice(neg, len(neg)), rng.choice(pos, len(pos))] for _ in range(n))]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def youden(y, p):
    fpr, tpr, thr = roc_curve(y, p); j = tpr - fpr; return float(thr[int(np.argmax(j))])


def sens_spec(y, p, thr):
    y = np.asarray(y); pred = np.asarray(p) >= thr
    tp, fn, tn, fp = int((pred & (y == 1)).sum()), int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum()), int((pred & (y == 0)).sum())
    return tp, tp + fn, tn, tn + fp


def short(f):
    return f.replace("_original_", ":")


def shap_values_linear(est, icc, Xdev, outdir):
    """Exact linear SHAP (independent features): phi = coef * (z - mean(z)) on standardized selected features. Saved for plotting with the shap library."""
    model = est.named_steps["model"]; pre = est[:-2].transform(icc.transform(Xdev))
    names = [short(c) for c in pre.columns]; Z = est.named_steps["scale"].transform(pre)
    coef = np.asarray(getattr(model, "model_", model).coef_).ravel(); keep = np.flatnonzero(np.abs(coef) > 1e-7); sv = coef[None, :] * (Z - Z.mean(0, keepdims=True))
    np.savez(outdir / "shap_values.npz", sv=sv[:, keep], Z=Z[:, keep], names=np.array(names)[keep], ids=np.array(Xdev.index.astype(str)))
    pd.DataFrame({"feature": np.array(names)[keep], "standardized_coefficient": coef[keep], "mean_abs_shap": np.abs(sv[:, keep]).mean(0)}).sort_values("mean_abs_shap", ascending=False).to_csv(outdir / "shap_coefficients.csv", index=False)
    return len(keep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpt-dir", required=True)
    ap.add_argument("--ladder-dir", default=str(ROOT / "Results/main_pipeline_claude/main_pipeline_tils/ladder_post_20261021"))
    ap.add_argument("--out", default=str(ROOT / "Results/main_pipeline_claude/presentation_20261020/tils_ge2"))
    ap.add_argument("--endpoint", choices=["tils_ge2", "tils_ge3", "tls_ge2"], default="tils_ge2")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--sets", nargs="*", default=None)
    ap.add_argument("--oof-glob", default="tilsGE2_post_*/oof_predictions.csv", help="glob (relative to --ladder-dir) of the ladder OOF files")
    ap.add_argument("--clinical-matrix", default=None); ap.add_argument("--clinical-cols", nargs="*", default=None)
    ap.add_argument("--sheet", choices=["pre", "post"], default="post")
    ap.add_argument("--tag", default="radiomics", help="category label used in output file names")
    ap.add_argument("--families", nargs="+", default=["ridge", "lasso", "elastic_net"])
    a = ap.parse_args(); warnings.filterwarnings("ignore")
    sys.path.insert(0, a.gpt_dir)
    import run_compact_ridge_sequence_search as core
    import run_sparse_dynamic_extensions as sp
    X, y, dev_ids, lock_ids, split_path, feature_path, readers = core.load_development(ROOT); lock_ids = sorted(lock_ids)
    sheet = ROOT / ("Results/Radiomics_sheets/SA1_2_post/R-SA1_2-post_I-sitkBSpline_S-1mm_W-perseq.csv" if a.sheet == "post" else "Results/Radiomics_sheets/SA1_2_pre/R-SA1_2_I-sitkBSpline_S-1mm_W-perseq.csv")
    X = pd.read_csv(sheet, index_col=0, usecols=lambda c: c == "PatientID" or "_original_" in c).loc[dev_ids]; X = X.reindex(sorted(X.columns), axis=1).astype(float)      # dev features from the chosen sheet
    XL = pd.read_csv(sheet, index_col=0, usecols=lambda c: c == "PatientID" or "_original_" in c).loc[lock_ids]; XL = XL.reindex(sorted(XL.columns), axis=1).astype(float)
    assert list(XL.columns) == list(X.columns)
    clin_cols = []
    if a.clinical_matrix:
        import clinrad_patch as cp
        cm = pd.read_csv(a.clinical_matrix, index_col=0); cm = cm[a.clinical_cols] if a.clinical_cols else cm
        X = pd.concat([X, cm.loc[dev_ids].add_prefix(cp.CLIN).astype(float)], axis=1); XL = pd.concat([XL, cm.loc[lock_ids].add_prefix(cp.CLIN).astype(float)], axis=1)
        clin_cols = [c for c in X.columns if c.startswith(cp.CLIN)]; cp.apply(core, sp); core.TechnicalICCFilter = cp.icc_keep_clin(core.TechnicalICCFilter)
    lab = pd.read_csv(split_path, usecols=["PatientID", "set", "TILs_resection"]).set_index("PatientID"); assert (lab.loc[lock_ids, "set"] == "lock").all()
    yL = (lab.loc[lock_ids, "TILs_resection"] >= 2).astype(int)
    if a.endpoint == "tils_ge3":     # TILs >= 3 vs 0-2
        y = (lab.loc[dev_ids, "TILs_resection"] >= 3).astype(int); yL = (lab.loc[lock_ids, "TILs_resection"] >= 3).astype(int)
    if a.endpoint == "tls_ge2":      # TLS >= 2 vs 0-1 (outcomes table v4; identical to v3)
        tls = pd.read_excel(ROOT / "databases/immunophenotype_outcomes_v4.xlsx").set_index("PatientID")["TLS_resection"]; y = (tls.loc[dev_ids] >= 2).astype(int); yL = (tls.loc[lock_ids] >= 2).astype(int)
    import run_guard
    oof = pd.concat([pd.read_csv(f) for f in run_guard.usable_files(Path(a.ladder_dir) / a.oof_glob)])
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    print(f"dev {X.shape} pos {int(y.sum())} | lock {XL.shape} pos {int(yL.sum())} neg {int((1 - yL).sum())} | OOF rows {len(oof)}", flush=True)
    (out / "README.txt").write_text(__doc__)
    cache_dir = Path("/private/tmp") / ("lockrep_" + datetime.now().strftime("%H%M%S") + f"_{a.endpoint}_{a.tag}")
    memory = joblib.Memory(cache_dir, verbose=0); rows = []
    import atexit, shutil; atexit.register(shutil.rmtree, str(cache_dir), True)
    import itertools
    sets = a.sets or ["+".join(c) for k in (1, 2, 3) for c in itertools.combinations(CORE4, k)]
    for sname in sets:
        seqs = sname.split("+"); cols = [c for c in X.columns if any(c.startswith(s + "_original_") for s in seqs)] + clin_cols
        for fam in a.families:
            t0 = time.time(); o = oof[(oof.sequence_set == sname) & (oof.family == fam) & oof.prob.notna()]
            if o.empty:
                print(f"[{sname}|{fam}] no OOF predictions, skipped", flush=True); continue
            per = o.groupby("repeat").apply(lambda d: roc_auc_score(d.y, d.prob)); avg = o.groupby("PatientID").agg(y=("y", "first"), prob=("prob", "mean")).loc[dev_ids]
            thr = youden(avg.y, avg.prob); lo, hi = boot_auc(avg.y, avg.prob); stp, stn, spt, spn = sens_spec(avg.y, avg.prob, thr)
            inner = list(StratifiedKFold(3, shuffle=True, random_state=999).split(X[cols], y))
            icc, est, params, score, audit, _ = sp.fit_one(fam, X[cols], y, inner, readers, memory, a.jobs)
            p = est.predict_proba(icc.transform(XL[cols]))[:, 1]; llo, lhi = boot_auc(yL, p); ltp, ltn, lsp, lsn = sens_spec(yL, p, thr)
            mdir = out / "models" / a.tag / f"{sname.replace('+', '_')}__{fam}"; mdir.mkdir(parents=True, exist_ok=True)
            nshap = shap_values_linear(est, icc, X[cols], mdir)
            pd.DataFrame({"PatientID": lock_ids, "y_lock": yL.values, "prob": p}).to_csv(mdir / "lock_predictions.csv", index=False)
            r = dict(sequence_set=sname, n_sequences=len(seqs), family=fam, n_repeats_ok=int(per.shape[0]),
                     oof_auc_repeat_mean=float(per.mean()), oof_auc_repeat_sd=float(per.std()), oof_auc_avgprob=roc_auc_score(avg.y, avg.prob), oof_auc_ci_lo=lo, oof_auc_ci_hi=hi,
                     threshold_youden_dev_oof=thr, oof_sens=stp / stn, oof_sens_ci_lo=wilson(stp, stn)[0], oof_sens_ci_hi=wilson(stp, stn)[1],
                     oof_spec=spt / spn, oof_spec_ci_lo=wilson(spt, spn)[0], oof_spec_ci_hi=wilson(spt, spn)[1],
                     lock_auc=roc_auc_score(yL, p), lock_auc_ci_lo=llo, lock_auc_ci_hi=lhi,
                     lock_sens=ltp / ltn, lock_sens_ci_lo=wilson(ltp, ltn)[0], lock_sens_ci_hi=wilson(ltp, ltn)[1],
                     lock_spec=lsp / lsn, lock_spec_ci_lo=wilson(lsp, lsn)[0], lock_spec_ci_hi=wilson(lsp, lsn)[1],
                     lock_ap=average_precision_score(yL, p), lock_brier=brier_score_loss(yL, p), n_features=len(audit.get("selected_features", [])), n_clinical_in_model=int(sum(str(f).startswith("clin_") for f in audit.get("selected_features", []))), n_shap_features=nshap,
                     params=json.dumps({k: float(v) for k, v in params.items()}), seconds=round(time.time() - t0))
            rows.append(r); pd.DataFrame(rows).to_csv(out / f"{a.endpoint}_model_table_{a.tag}.csv", index=False)
            print(f"[{sname}|{fam}] OOF AUC {r['oof_auc_repeat_mean']:.3f} | lock AUC {r['lock_auc']:.3f} ({llo:.2f}-{lhi:.2f}) sens {r['lock_sens']:.2f} spec {r['lock_spec']:.2f}", flush=True)
    (out / "manifest.json").write_text(json.dumps(dict(finished=datetime.now().isoformat(), n_dev=len(X), n_lock=len(XL), lock_labels_read=True, sklearn=sklearn.__version__, sheet=str(sheet), n_models=len(rows)), indent=2))
    print("COMPLETED=" + str(out), flush=True)


if __name__ == "__main__":
    main()
