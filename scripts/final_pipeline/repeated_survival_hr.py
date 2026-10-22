"""Old recurrence selection (univariate-Cox HR screen) re-run inside the same repeated outer CV as repeated_survival_families.py.

Per outer-training set, mirroring recurrence_pipeline.run_fold_pruning_survival:
  global reader-ICC>=0.85 + QC (>=80% observed, CV>0.10) -> KNN(5) impute -> standardize ->
  univariate Cox per feature (log HR per SD; Breslow Newton iterations, vectorized over features) ->
  keep |log HR| > log(hr_cutoff) -> BH-FDR q < q_cutoff within the screened pool ->
  greedy pool-wide Spearman prune |rho| > 0.85 (lower p kept) ->
  lasso-Cox, alpha chosen by inner 3-fold C-index on a 20-point grid anchored at alpha_max (steps up if the fit diverges), keep <= target_count by |coef|.
Metric: pooled out-of-fold Harrell C-index; repeat r uses outer seed 42+r, strata = event x cohort; dev patients only.
"""
import argparse, json, sys, time, warnings
from datetime import datetime
from pathlib import Path
sys.dont_write_bytecode = True
import numpy as np, pandas as pd
from scipy.stats import norm, rankdata
from sklearn.impute import KNNImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

ROOT = Path("/Users/Francisco/Library/CloudStorage/OneDrive-TheMountSinaiHospital/Regeneron retrospective radiomics")


def univariate_cox(X, event, time, iters=8):
    """Per-column Cox log HR, Wald p (Breslow ties), Newton from 0. X: (n, p) standardized."""
    order = np.argsort(time, kind="stable"); X = X[order]; ev = event[order]; t = time[order]
    start = np.searchsorted(t, t, side="left")                     # first index of each tie group = risk-set start
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        w = np.exp(np.clip(X * beta, -30, 30))
        rc = lambda a: np.cumsum(a[::-1], axis=0)[::-1][start]
        S0, S1, S2 = rc(w), rc(w * X), rc(w * X * X)
        m = S1 / S0
        U = (ev[:, None] * (X - m)).sum(0); V = (ev[:, None] * (S2 / S0 - m * m)).sum(0)
        step = np.where(V > 1e-12, U / np.maximum(V, 1e-12), 0.0)
        beta = beta + np.clip(step, -2, 2)
    se = 1 / np.sqrt(np.maximum(V, 1e-12))
    p = 2 * norm.sf(np.abs(beta / se))
    return beta, p


def bh(p):
    n = len(p); o = np.argsort(p); q = np.empty(n)
    q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1.0)


def spearman_prune(X, p, thr=0.85):
    """Greedy pool-wide prune equal to the old double loop: scan in column order, drop the less significant of any correlated pair."""
    if X.shape[1] < 2:
        return np.arange(X.shape[1])
    R = np.abs(np.corrcoef(rankdata(X, axis=0), rowvar=False)); R = np.nan_to_num(R)
    drop = np.zeros(X.shape[1], bool)
    for i in range(X.shape[1]):
        if drop[i]:
            continue
        hit = np.flatnonzero((R[i] > thr) & ~drop & (np.arange(X.shape[1]) > i))
        for j in hit:
            if drop[i]:
                break
            if p[i] <= p[j]:
                drop[j] = True
            else:
                drop[i] = True
    return np.flatnonzero(~drop)


def fit_fold(Xtr, ytr, hr_cutoff, q_cutoff, target, seed, forced_mode=False, top_n=40):
    imp = KNNImputer(n_neighbors=5).fit(Xtr); sc = StandardScaler().fit(imp.transform(Xtr))
    Z = sc.transform(imp.transform(Xtr)); event = np.asarray(ytr["event"], bool); time = np.asarray(ytr["time"], float)
    beta, p = univariate_cox(Z, event, time)
    if forced_mode:   # TILs-style: no HR/FDR cutoff; rank by Cox p, prune pool-wide, lasso constrained to 10-20 nonzero
        order = np.argsort(p, kind="stable"); kp = order[spearman_prune(Z[:, order], p[order])][:top_n]; Zs = Z[:, kp]
        amax = float(CoxnetSurvivalAnalysis(l1_ratio=1.0, n_alphas=2, alpha_min_ratio=0.9, max_iter=100000, fit_baseline_model=False).fit(Zs, ytr).alphas_.max())
        alphas = np.geomspace(amax, amax * 1e-3, 40)
        while True:
            try:
                path = CoxnetSurvivalAnalysis(alphas=alphas.tolist(), l1_ratio=1.0, max_iter=100000, fit_baseline_model=False).fit(Zs, ytr); break
            except ArithmeticError:
                alphas = alphas[: len(alphas) // 2]
                if len(alphas) < 4:
                    raise RuntimeError("Coxnet path diverged")
        nnz = (np.abs(np.asarray(path.coef_)) > 1e-7).sum(0)
        ok = np.flatnonzero((nnz >= 10) & (nnz <= 20))
        if len(ok) == 0:
            raise RuntimeError("No alpha gives 10-20 nonzero coefficients")
        score = np.zeros(len(ok)); cnt = np.zeros(len(ok))
        for tr, va in StratifiedKFold(3, shuffle=True, random_state=seed).split(Zs, event):
            try:
                m = CoxnetSurvivalAnalysis(alphas=alphas[: ok.max() + 1].tolist(), l1_ratio=1.0, max_iter=100000, fit_baseline_model=False).fit(Zs[tr], ytr[tr])
            except ArithmeticError:
                continue
            for i, j in enumerate(ok):
                c = concordance_index_censored(ytr[va]["event"], ytr[va]["time"], m.predict(Zs[va], alpha=alphas[j]))[0]
                if np.isfinite(c):
                    score[i] += c; cnt[i] += 1
        j = ok[int(np.argmax(np.where(cnt > 0, score / np.maximum(cnt, 1), -np.inf)))]
        coef = np.asarray(path.coef_)[:, j]; nz = np.flatnonzero(np.abs(coef) > 1e-7)
        return imp, sc, kp, nz, path, dict(n_screened=int(len(kp)), n_selected=int(len(nz)), alpha=float(alphas[j]), _coef=coef)
    keep = np.flatnonzero(np.abs(beta) > np.log(hr_cutoff))
    if len(keep) == 0:
        raise RuntimeError(f"No features passed the HR screen (cutoff={hr_cutoff})")
    keep = keep[bh(p[keep]) < q_cutoff]
    if len(keep) == 0:
        raise RuntimeError("No features survived FDR")
    keep = keep[spearman_prune(Z[:, keep], p[keep])]
    Zs = Z[:, keep]
    amax = float(CoxnetSurvivalAnalysis(l1_ratio=1.0, n_alphas=100, max_iter=100000, fit_baseline_model=False).fit(Zs, ytr).alphas_.max())
    alphas = np.sort(np.logspace(np.log10(amax) - 5, np.log10(amax), 20))[::-1]
    score = np.zeros(len(alphas)); cnt = np.zeros(len(alphas))
    for tr, va in StratifiedKFold(3, shuffle=True, random_state=seed).split(Zs, event):
        try:
            m = CoxnetSurvivalAnalysis(alphas=alphas.tolist(), l1_ratio=1.0, max_iter=100000, fit_baseline_model=False).fit(Zs[tr], ytr[tr])
        except ArithmeticError:
            continue
        for i, a in enumerate(alphas):
            try:
                c = concordance_index_censored(ytr[va]["event"], ytr[va]["time"], m.predict(Zs[va], alpha=a))[0]
            except Exception:
                continue
            if np.isfinite(c):
                score[i] += c; cnt[i] += 1
    mean = np.where(cnt > 0, score / np.maximum(cnt, 1), -np.inf); best = alphas[int(np.argmax(mean))]
    for a in sorted(a for a in alphas if a >= best):
        try:
            lasso = CoxnetSurvivalAnalysis(alphas=[float(a)], l1_ratio=1.0, max_iter=100000, fit_baseline_model=False).fit(Zs, ytr); break
        except ArithmeticError:
            continue
    else:
        raise RuntimeError("lasso diverged at every candidate alpha")
    coef = np.asarray(lasso.coef_).ravel(); nz = np.flatnonzero(coef)
    if len(nz) == 0:
        nz = np.argsort(-np.abs(coef))[:target]
    nz = nz[np.argsort(-np.abs(coef[nz]))][:target]
    return imp, sc, keep, nz, lasso, dict(n_screened=int(len(keep)), n_selected=int(len(nz)), alpha=float(a))


def predict(model, Xva):
    imp, sc, keep, nz, lasso, info = model
    Z = sc.transform(imp.transform(Xva))[:, keep]
    if "_coef" in info:
        return Z @ info["_coef"]
    coef = np.zeros(len(keep)); lc = np.asarray(lasso.coef_).ravel(); coef[nz] = lc[nz]
    return Z @ coef


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpt-dir", required=True)
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--hr-cutoff", type=float, default=1.5)
    ap.add_argument("--q-cutoff", type=float, default=0.20)
    ap.add_argument("--target-count", type=int, default=20)
    ap.add_argument("--forced", action="store_true", help="TILs-style: rank by Cox p (no HR/FDR cutoff), top-N after prune, lasso constrained to 10-20 nonzero")
    ap.add_argument("--top-n", type=int, default=40)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--split", default=str(ROOT / "Results/main_pipeline_claude/main_pipeline_recurrence/split/recurrence_split.csv"))
    ap.add_argument("--output-parent", required=True)
    ap.add_argument("--label", default="rechr")
    a = ap.parse_args(); warnings.filterwarnings("ignore")
    sys.path.insert(0, a.gpt_dir)
    import run_compact_ridge_sequence_search as core
    sp = pd.read_csv(a.split); dev = sp[sp["set"] == "dev"].reset_index(drop=True); ids = dev.PatientID.tolist()
    y = Surv.from_arrays(event=dev.event.astype(bool).values, time=dev.time_days.astype(float).values, name_event="event", name_time="time")
    X = pd.read_csv(ROOT / "Results/Radiomics_sheets/SA1_2_pre/R-SA1_2_I-sitkBSpline_S-1mm_W-perseq.csv", index_col=0, usecols=lambda c: c == "PatientID" or "_original_" in c).loc[ids]
    X = X.reindex(sorted(X.columns), axis=1).astype(float)
    readers = [str(ROOT / f"Results/Radiomics_sheets/inter-reader/Reader{i}/R-Reader{i}_I-sitkBSpline_S-1mm_W-perseq.csv") for i in (1, 2)]
    strata = dev.event.astype(int).astype(str) + "_" + dev.cohort
    run = Path(a.output_parent) / f"{a.label}_{'forced' if a.forced else 'hr' + str(a.hr_cutoff)}_{('-'.join(s.replace('+', '_') for s in a.sets) if len(a.sets) <= 2 else f'{len(a.sets)}sets')}_{datetime.now():%Y%m%d_%H%M%S}"; run.mkdir(parents=True, exist_ok=False)
    (run / "manifest.json").write_text(json.dumps(dict(started=datetime.now().isoformat(), n_dev=len(ids), events=int(dev.event.sum()), lock_evaluated=False, sets=a.sets,
        hr_cutoff=a.hr_cutoff, q_cutoff=a.q_cutoff, forced=a.forced, top_n=a.top_n, target_count=a.target_count, repeats=a.repeats, cox_screen="Newton Breslow, Wald p"), indent=2))
    print("RUN_DIRECTORY=" + str(run), flush=True)
    rows = []
    for sname in a.sets:
        cols = [c for c in X.columns if any(c.startswith(s + "_original_") for s in sname.split("+"))]
        keepc = list(core.TechnicalICCFilter(*readers).fit(X[cols]).columns_); Xs = X[keepc]
        # QC once on dev (>=80% observed, CV>0.10): outcome-independent
        obs = Xs.notna().mean() >= 0.8; cv = (Xs.std() / Xs.mean().abs()); Xs = Xs.loc[:, obs & ((cv > 0.10) | cv.isna())]
        print(f"[{sname}] ICC {len(keepc)}/{len(cols)}, after QC {Xs.shape[1]}", flush=True)
        for rep in range(a.repeats):
            t0 = time.time(); risk = np.full(len(ids), np.nan); info = []
            try:
                for fold, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42 + rep).split(Xs, strata), 1):
                    m = fit_fold(Xs.iloc[tr].to_numpy(), y[tr], a.hr_cutoff, a.q_cutoff, a.target_count, 420 + fold + 1000 * rep, a.forced, a.top_n)
                    risk[va] = predict(m, Xs.iloc[va].to_numpy()); info.append(m[-1])
            except (ValueError, RuntimeError, ArithmeticError, np.linalg.LinAlgError) as e:
                rows.append(dict(sequence_set=sname, repeat=rep, c_index=np.nan, failed_reason=str(e)[:100])); pd.DataFrame(rows).to_csv(run / "per_repeat.csv", index=False)
                print(f"[{sname}] repeat {rep}: FAILED ({str(e)[:80]})", flush=True); continue
            c = concordance_index_censored(y["event"], y["time"], risk)[0]
            rows.append(dict(sequence_set=sname, repeat=rep, c_index=float(c), mean_screened=float(np.mean([i["n_screened"] for i in info])), mean_selected=float(np.mean([i["n_selected"] for i in info]))))
            pd.DataFrame(rows).to_csv(run / "per_repeat.csv", index=False)
            print(f"[{sname}] repeat {rep}: C-index {c:.3f} screened {rows[-1]['mean_screened']:.0f} selected {rows[-1]['mean_selected']:.1f} ({time.time()-t0:.0f}s)", flush=True)
    D = pd.DataFrame(rows)
    D.groupby("sequence_set").c_index.agg(mean="mean", sd="std", mn="min", mx="max", n_ok=lambda x: int(x.notna().sum()), n="size").reset_index().round(4).to_csv(run / "summary.csv", index=False)
    print("COMPLETED=" + str(run), flush=True)


if __name__ == "__main__":
    main()
