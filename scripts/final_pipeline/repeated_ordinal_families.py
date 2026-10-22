"""Repeated outer-CV of the 'gpt' ORDINAL procedure (regression analogues of lasso / elastic net) for chosen sequence sets.

Imports their fit_one() from run_radiomics_tils_ordinal_shortlist.py (read-only copy): reader-ICC, QC, median imputation, within-sequence
complete-linkage clusters, representatives ranked by |Spearman rho| with the ordinal outcome, Lasso / ElasticNet tuned on inner Spearman rho with
a 10-20 nonzero-coefficient constraint. Only the OUTER design is repeated: repeat r = outer seed 42+r, inner seeds 420+fold+1000*r.
Outer folds are stratified on ordinal level x cohort (SA1/SA2); reader-ICC is estimated once on all dev double-reader patients (outcome-independent).
Endpoints: tls_ordinal (TLS_resection 0/1/2) or tils_ordinal (TILs_resection 0-3). Dev patients only; lock never read. Needs scikit-learn >= 1.8 python.
Primary metric: pooled OOF Spearman rho between observed score and the continuous prediction; secondary MAE, RMSE, R2, weighted kappa.
"""
import argparse, json, sys, time, warnings
from datetime import datetime
from pathlib import Path
sys.dont_write_bytecode = True
import numpy as np, pandas as pd, joblib, sklearn
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path("/Users/Francisco/Library/CloudStorage/OneDrive-TheMountSinaiHospital/Regeneron retrospective radiomics")


def ord_metrics(y, pred, top):
    clipped = np.clip(np.asarray(pred, float), 0, top); rounded = np.rint(clipped).astype(int)
    rho = spearmanr(y, clipped).statistic
    return dict(spearman_rho=float(rho) if np.isfinite(rho) else 0.0, mae=float(mean_absolute_error(y, clipped)), rmse=float(mean_squared_error(y, clipped) ** 0.5),
                r2=float(r2_score(y, clipped)), quadratic_weighted_kappa=float(cohen_kappa_score(y, rounded, weights="quadratic")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpt-dir", required=True)
    ap.add_argument("--sheet", choices=["post", "pre"], default="pre")
    ap.add_argument("--endpoint", choices=["tls_ordinal", "tils_ordinal"], default="tls_ordinal")
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--families", nargs="+", default=["lasso", "elastic_net"])
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--outcomes", default=str(ROOT / "databases/immunophenotype_outcomes_v3.xlsx"))
    ap.add_argument("--output-parent", required=True)
    ap.add_argument("--label", default="ordinal")
    a = ap.parse_args(); warnings.filterwarnings("ignore")
    assert tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 8)
    sys.path.insert(0, a.gpt_dir)
    import run_compact_ridge_sequence_search as core
    import run_radiomics_tils_ordinal_shortlist as od
    X, y0, dev_ids, lock_ids, split_path, feature_path, readers = core.load_development(ROOT)
    col = "TLS_resection" if a.endpoint == "tls_ordinal" else "TILs_resection"
    lab = pd.read_excel(a.outcomes).set_index("PatientID")[col].reindex(dev_ids); assert lab.notna().all()
    y = lab.astype(int); top = int(y.max())
    sheet = ROOT / ("Results/Radiomics_sheets/SA1_2_post/R-SA1_2-post_I-sitkBSpline_S-1mm_W-perseq.csv" if a.sheet == "post" else "Results/Radiomics_sheets/SA1_2_pre/R-SA1_2_I-sitkBSpline_S-1mm_W-perseq.csv")
    X = pd.read_csv(sheet, index_col=0, usecols=lambda c: c == "PatientID" or "_original_" in c).loc[dev_ids]
    X = X.reindex(sorted(X.columns), axis=1).astype(float); assert X.shape == (len(dev_ids), 963)
    sa2 = set(pd.read_csv(ROOT / "Results/Radiomics_sheets/SA2_pre/R-SA2_I-sitkBSpline_S-1mm_W-perseq.csv", index_col=0, usecols=[0]).index)
    cohort = pd.Series(["SA2" if i in sa2 else "SA1" for i in y.index], index=y.index); strata = y.astype(str) + "_" + cohort
    run = Path(a.output_parent) / f"{a.label}_{a.sheet}_{'-'.join(s.replace('+', '_') for s in a.sets)}_{datetime.now():%Y%m%d_%H%M%S}"; run.mkdir(parents=True, exist_ok=False)
    (run / "manifest.json").write_text(json.dumps(dict(started=datetime.now().isoformat(), endpoint=col + " ordinal", levels=y.value_counts().sort_index().to_dict(), n_dev=len(y), lock_evaluated=False,
        sets=a.sets, families=a.families, repeats=a.repeats, strata=strata.value_counts().to_dict(), icc="global (all dev double-reader patients)", sklearn=sklearn.__version__), indent=2, default=str))
    print("RUN_DIRECTORY=" + str(run), "| levels", y.value_counts().sort_index().to_dict(), "| min stratum", int(strata.value_counts().min()), flush=True)
    RealICC = core.TechnicalICCFilter; memory = joblib.Memory(Path("/private/tmp") / run.name, verbose=0); rows = []
    for sname in a.sets:
        cols = sorted(c for c in X.columns if any(c.startswith(s + "_original_") for s in sname.split("+"))); Xs = X[cols]
        fixed = list(RealICC(*readers).fit(Xs).columns_)
        class FixedICC(RealICC):
            def fit(self, Xf, yf=None):
                self.columns_ = [c for c in fixed if c in Xf.columns]; self.icc_ = {}; self.training_ids_ = list(Xf.index); self.icc_ids_ = []; self.n_before_icc_ = len(self.columns_); return self
        core.TechnicalICCFilter = FixedICC
        print(f"[{sname}] global ICC keeps {len(fixed)} of {len(cols)} features", flush=True)
        for fam in a.families:
            for rep in range(a.repeats):
                t0 = time.time(); pred = np.full(len(y), np.nan); nf = []
                try:
                    for fold, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42 + rep).split(Xs, strata), 1):
                        inner = list(StratifiedKFold(3, shuffle=True, random_state=420 + fold + 1000 * rep).split(Xs.iloc[tr], y.iloc[tr]))
                        icc, est, params, score, audit, _ = od.fit_one(fam, Xs.iloc[tr], y.iloc[tr], inner, readers, memory, a.jobs)
                        pred[va] = est.predict(icc.transform(Xs.iloc[va])); nf.append(audit["n_model_features"])
                except (ValueError, RuntimeError) as e:
                    rows.append(dict(sequence_set=sname, family=fam, repeat=rep, spearman_rho=np.nan, failed_reason=str(e)[:100]))
                    pd.DataFrame(rows).to_csv(run / "per_repeat.csv", index=False)
                    print(f"[{a.endpoint} | {sname} | {fam}] repeat {rep}: FAILED ({str(e)[:80]})", flush=True); continue
                m = ord_metrics(y.values, pred, top)
                rows.append(dict(sequence_set=sname, family=fam, repeat=rep, **m, mean_selected_features=float(np.mean(nf))))
                pd.DataFrame(rows).to_csv(run / "per_repeat.csv", index=False)
                print(f"[{a.endpoint} | {sname} | {fam}] repeat {rep}: rho {m['spearman_rho']:.3f} MAE {m['mae']:.3f} ({time.time()-t0:.0f}s)", flush=True)
    D = pd.DataFrame(rows)
    D.groupby(["sequence_set", "family"]).agg(mean_rho=("spearman_rho", "mean"), sd_rho=("spearman_rho", "std"), min_rho=("spearman_rho", "min"), max_rho=("spearman_rho", "max"),
        mean_mae=("mae", "mean"), mean_qwk=("quadratic_weighted_kappa", "mean"), n_ok=("spearman_rho", lambda x: int(x.notna().sum())), n=("spearman_rho", "size")).reset_index().round(4).to_csv(run / "summary.csv", index=False)
    print("COMPLETED=" + str(run), flush=True)


if __name__ == "__main__":
    main()
