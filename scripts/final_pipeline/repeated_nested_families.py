"""Repeated outer-CV of the 'gpt' ladder procedure (ridge / lasso / elastic net) for chosen sequence sets.

Uses THEIR fit_one() from run_sparse_dynamic_extensions.py (imported read-only) so the inner procedure is identical:
reader-ICC on the outer-training patients, QC, median imputation, within-sequence complete-linkage clusters, outcome-ranked
representatives; ridge = top-k (10/15/20) + C grid, lasso/EN = all cluster representatives with a 10-20 nonzero-coefficient
constraint, inner scoring AUC (3-fold). Only the OUTER split is repeated (repeat r: outer seed 42+r, inner seeds 420+fold+1000*r);
repeat 0 reproduces the single-split numbers in the ladder. Dev patients only; lock never read. Requires scikit-learn >= 1.8
(the sparse code relies on l1_ratio acting without penalty='elasticnet').
"""
import argparse, json, sys, time, warnings
from datetime import datetime
from pathlib import Path
sys.dont_write_bytecode = True
import numpy as np, pandas as pd, joblib, sklearn
from sklearn.model_selection import StratifiedKFold

GPT = None      # no default: --gpt-dir must point at this session's own snapshot copy, never at another agent's live folder
ROOT = Path("/Users/Francisco/Library/CloudStorage/OneDrive-TheMountSinaiHospital/Regeneron retrospective radiomics")
SETS = {"tp": ("tp",), "pvp+tp": ("pvp", "tp"), "dwi_b2+tp": ("dwi_b2", "tp"), "dwi_b2+pvp+tp": ("dwi_b2", "pvp", "tp")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", choices=["post", "pre"], default="post")
    ap.add_argument("--sets", nargs="+", default=list(SETS), help="sequence sets as e.g. tp or dwi_b2+tp (any 1+ of the 9 sequences)")
    ap.add_argument("--families", nargs="+", default=["ridge", "lasso", "elastic_net"])
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--output-parent", default=str(ROOT / "Results/main_pipeline_best_pipeline"))
    ap.add_argument("--label", default="repeated_nested")
    ap.add_argument("--endpoint", choices=["tils_ge2", "tils_ge3", "tls_ge2", "tls_ge1"], default="tils_ge2",
                    help="tils_ge2: TILs>=2 (default, from the split file); tls_ge2: TLS_resection>=2 (0-1 vs 2) read from --outcomes for the dev patients")
    ap.add_argument("--outcomes", default=str(ROOT / "databases/immunophenotype_outcomes_v4.xlsx"))
    ap.add_argument("--strata", choices=["outcome", "outcome_cohort"], default="outcome",
                    help="what the outer folds are stratified on; outcome_cohort balances SA1/SA2 as well as the outcome")
    ap.add_argument("--icc", choices=["fold", "global"], default="fold",
                    help="fold: reader-ICC re-estimated on each outer-training set (default); global: estimated ONCE on all dev double-reader patients "
                         "(ICC uses reader measurements only, never the outcome, so this is not outcome leakage)")
    ap.add_argument("--clinical-matrix", default=None, help="CSV (PatientID index, numeric) of clinico-radiologic variables to add to every radiomics model (prefix clin_)")
    ap.add_argument("--clinical-cols", nargs="*", default=None)
    ap.add_argument("--gpt-dir", required=True, help="folder holding run_compact_ridge_sequence_search.py and run_sparse_dynamic_extensions.py (a byte-identical local copy is more robust than OneDrive)")
    a = ap.parse_args(); warnings.filterwarnings("ignore")
    assert tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 8), "needs scikit-learn >= 1.8 (use /usr/local/bin/python3)"
    sys.path.insert(0, a.gpt_dir)
    import run_compact_ridge_sequence_search as core
    import run_sparse_dynamic_extensions as sp
    X0, y, dev_ids, lock_ids, split_path, feature_path, readers = core.load_development(ROOT)
    sheet = ROOT / ("Results/Radiomics_sheets/SA1_2_post/R-SA1_2-post_I-sitkBSpline_S-1mm_W-perseq.csv" if a.sheet == "post"
                    else "Results/Radiomics_sheets/SA1_2_pre/R-SA1_2_I-sitkBSpline_S-1mm_W-perseq.csv")
    if a.endpoint == "tils_ge3":
        tils = pd.read_excel(a.outcomes).set_index("PatientID")["TILs_resection"].reindex(dev_ids)
        assert tils.notna().all(), f"missing TILs for dev patients: {tils[tils.isna()].index.tolist()}"
        y = (tils >= 3).astype(int)
    if a.endpoint in ("tls_ge2", "tls_ge1"):
        tls = pd.read_excel(a.outcomes).set_index("PatientID")["TLS_resection"].reindex(dev_ids)
        assert tls.notna().all(), f"missing TLS for dev patients: {tls[tls.isna()].index.tolist()}"
        y = (tls >= (2 if a.endpoint == "tls_ge2" else 1)).astype(int)
    X = pd.read_csv(sheet, index_col=0, usecols=lambda c: c == "PatientID" or "_original_" in c).loc[dev_ids]
    X = X.reindex(sorted(X.columns), axis=1).astype(float); assert X.shape == (len(dev_ids), 963), X.shape
    clin_cols = []
    if a.clinical_matrix:
        import clinrad_patch as cp
        cm = pd.read_csv(a.clinical_matrix, index_col=0).loc[dev_ids]; cm = cm[a.clinical_cols] if a.clinical_cols else cm
        X = pd.concat([X, cm.add_prefix(cp.CLIN).astype(float)], axis=1); clin_cols = [c for c in X.columns if c.startswith(cp.CLIN)]
        assert a.icc == "global", "clinical variables are supported with --icc global only"
    run = Path(a.output_parent) / f"{a.label}_{a.sheet}_{'-'.join(s.replace('+','_') for s in a.sets)}_{datetime.now():%Y%m%d_%H%M%S}"
    run.mkdir(parents=True, exist_ok=False)
    (run / "manifest.json").write_text(json.dumps(dict(started=datetime.now().isoformat(), sheet=str(sheet), endpoint={"tils_ge3": "TILs>=3 (0-2 vs 3)", "tls_ge2": "TLS>=2 (0-1 vs 2)", "tls_ge1": "TLS>=1 (0 vs 1-2)", "tils_ge2": "TILs>=2"}[a.endpoint], n_dev=len(dev_ids), lock_evaluated=False,
        sets=a.sets, families=a.families, repeats=a.repeats, clinical_cols=a.clinical_cols if a.clinical_matrix else [], sklearn=sklearn.__version__, gpt_sparse_script=sp.__file__, gpt_sha256={Path(m.__file__).name: __import__("hashlib").sha256(Path(m.__file__).read_bytes()).hexdigest() for m in (core, sp)}, outer="5-fold stratified seed 42+repeat",
        inner="3-fold seed 420+fold+1000*repeat"), indent=2))
    print("RUN_DIRECTORY=" + str(run), flush=True)
    if clin_cols:
        cp.apply(core, sp)
    RealICC = core.TechnicalICCFilter      # keep the ORIGINAL class: the global-ICC patch below must never feed back into the next set
    memory = joblib.Memory(Path("/private/tmp") / run.name, verbose=0); rows = []; oof_rows = []
    import atexit, shutil; atexit.register(shutil.rmtree, str(Path("/private/tmp") / run.name), True)      # the joblib cache is only needed during the run; never leave it behind (it filled the disk once)
    sa2_ids = set(pd.read_csv(ROOT / "Results/Radiomics_sheets/SA2_pre/R-SA2_I-sitkBSpline_S-1mm_W-perseq.csv", index_col=0, usecols=[0]).index)
    cohort = pd.Series(["SA2" if i in sa2_ids else "SA1" for i in y.index], index=y.index)
    strata = y.astype(str) if a.strata == "outcome" else (y.astype(str) + "_" + cohort)
    print("fold strata:", strata.value_counts().to_dict(), "| ICC:", a.icc, flush=True)
    for sname in a.sets:
        cols = sorted(c for c in X.columns if any(c.startswith(s + "_original_") for s in sname.split("+"))); Xs = X[cols + clin_cols]
        if a.icc == "global":
            fixed = list(RealICC(*readers).fit(Xs).columns_)
            class FixedICC(RealICC):
                def fit(self, Xf, yf=None):
                    self.columns_ = [c for c in fixed if c in Xf.columns] + [c for c in Xf.columns if c in clin_cols]; self.icc_ = {}; self.training_ids_ = list(Xf.index)
                    self.icc_ids_ = []; self.n_before_icc_ = len(self.columns_); return self
            core.TechnicalICCFilter = FixedICC
            print(f"[{sname}] global ICC filter keeps {len(fixed)} of {len(cols)} features", flush=True)
        for fam in a.families:
            for rep in range(a.repeats):
                t0 = time.time(); pred = np.full(len(y), np.nan); sel = []
                try:
                    for fold, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42 + rep).split(Xs, strata), 1):
                        inner = list(StratifiedKFold(3, shuffle=True, random_state=420 + fold + 1000 * rep).split(Xs.iloc[tr], y.iloc[tr]))
                        icc, est, params, score, audit, _ = sp.fit_one(fam, Xs.iloc[tr], y.iloc[tr], inner, readers, memory, a.jobs)
                        pred[va] = est.predict_proba(icc.transform(Xs.iloc[va]))[:, 1]
                        sel.append(len(audit["selected_features"]) if "selected_features" in audit else np.nan)
                except (ValueError, RuntimeError) as e:
                    # pipeline feasibility limits (e.g. fewer feature clusters than the required k, or a sparse model that cannot reach
                    # 10-20 nonzero coefficients): record the repeat as FAILED (NaN) with the reason and continue; failures are counted in the summary
                    rows.append(dict(sequence_set=sname, family=fam, repeat=rep, auc=np.nan, average_precision=np.nan, brier=np.nan, mean_selected_features=np.nan,
                                     failed_reason=str(e)[:100]))
                    pd.DataFrame(rows).to_csv(run / "per_repeat_auc.csv", index=False)
                    print(f"[{a.sheet} | {sname} | {fam}] repeat {rep}: FAILED ({str(e)[:90]})", flush=True)
                    continue
                m = core.classification_metrics(y, pred)
                oof_rows.append(pd.DataFrame(dict(sequence_set=sname, family=fam, repeat=rep, PatientID=y.index, y=y.values, prob=pred))); pd.concat(oof_rows).to_csv(run / "oof_predictions.csv", index=False)
                rows.append(dict(sequence_set=sname, family=fam, repeat=rep, **m, mean_selected_features=float(np.mean(sel))))
                pd.DataFrame(rows).to_csv(run / "per_repeat_auc.csv", index=False)
                print(f"[{a.sheet} | {sname} | {fam}] repeat {rep}: AUC {m['auc']:.3f} ({time.time()-t0:.0f}s)", flush=True)
    D = pd.DataFrame(rows)
    D.groupby(["sequence_set", "family"]).auc.agg(mean_auc="mean", sd_auc="std", min_auc="min", max_auc="max", repeat0_auc="first", n_ok=lambda x: int(x.notna().sum()), n_failed=lambda x: int(x.isna().sum())).reset_index().round(4).to_csv(run / "summary.csv", index=False)
    print("COMPLETED=" + str(run), flush=True)


if __name__ == "__main__":
    main()
