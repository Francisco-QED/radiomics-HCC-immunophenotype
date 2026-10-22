"""One-shot lock-set evaluation of the 'old batch' ladder models (single, double and triple sequence sets).

Procedure per model (identical to the ladder's final development fit):
  1. fit on ALL 110 development patients: reader-ICC(3,1) on all dev double-reader patients, QC, median imputation,
     within-sequence complete-linkage clusters, outcome-ranked representatives, ridge / lasso / elastic net tuned by 3-fold
     inner CV (seed 999) with the 10-20 nonzero-coefficient constraint for the sparse families;
  2. apply the fitted model, WITHOUT refitting, to the 30 lock patients and compute AUC / AP / Brier.
Reporting plan fixed BEFORE running: all models are reported, none is selected, nothing is refitted after seeing lock results.
Uses their fit_one() (imported read-only). Pre-treatment SA2 imaging (SA1_2_pre), as in the ladder that produced 0.699.
"""
import argparse, hashlib, itertools, json, sys, time, warnings
from datetime import datetime
from pathlib import Path
sys.dont_write_bytecode = True
import numpy as np, pandas as pd, joblib, sklearn
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path("/Users/Francisco/Library/CloudStorage/OneDrive-TheMountSinaiHospital/Regeneron retrospective radiomics")
CORE4 = ["hbp", "dwi_b2", "pvp", "tp"]


def boot(y, p, n=2000, seed=20261019):
    rng = np.random.default_rng(seed); y = np.asarray(y); p = np.asarray(p)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0); v = []
    for _ in range(n):
        i = np.r_[rng.choice(neg, len(neg)), rng.choice(pos, len(pos))]; v.append(roc_auc_score(y[i], p[i]))
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpt-dir", required=True)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--sizes", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--families", nargs="+", default=["ridge", "lasso", "elastic_net"])
    ap.add_argument("--reference", default=str(ROOT / "Results/main_pipeline_post_replication/ladder45_sa1_sa2post_20261016/pre_vs_post_ladder45_comparison.csv"))
    ap.add_argument("--output-parent", default=str(ROOT / "Results/main_pipeline_lock_evaluation"))
    a = ap.parse_args(); warnings.filterwarnings("ignore")
    sys.path.insert(0, a.gpt_dir)
    import run_compact_ridge_sequence_search as core
    import run_sparse_dynamic_extensions as sp
    X, y, dev_ids, lock_ids, split_path, feature_path, readers = core.load_development(ROOT)          # dev only
    sheet = ROOT / "Results/Radiomics_sheets/SA1_2_pre/R-SA1_2_I-sitkBSpline_S-1mm_W-perseq.csv"
    lock_ids = sorted(lock_ids)
    XL = pd.read_csv(sheet, index_col=0, usecols=lambda c: c == "PatientID" or "_original_" in c).loc[lock_ids]
    XL = XL.reindex(sorted(XL.columns), axis=1).astype(float)
    assert list(XL.columns) == list(X.columns)
    lab = pd.read_csv(split_path, usecols=["PatientID", "set", "TILs_resection"]).set_index("PatientID")        # first read of lock labels
    assert (lab.loc[lock_ids, "set"] == "lock").all()
    yL = (lab.loc[lock_ids, "TILs_resection"] >= 2).astype(int)
    print(f"dev {X.shape} positives {int(y.sum())} | lock {XL.shape} positives {int(yL.sum())} negatives {int((1 - yL).sum())}", flush=True)
    ref = pd.read_csv(a.reference) if Path(a.reference).exists() else None
    run = Path(a.output_parent) / f"lock_eval_ladder_pre_{datetime.now():%Y%m%d_%H%M%S}"; run.mkdir(parents=True, exist_ok=False)
    (run / "README.md").write_text(__doc__ + f"""
Models: all {sum(len(list(itertools.combinations(CORE4, k))) for k in a.sizes)} sequence sets of sizes {a.sizes} from {CORE4} x families {a.families}.
Lock: {len(lock_ids)} patients ({int(yL.sum())} TILs>=2, {int((1 - yL).sum())} TILs<2); AUC standard error about 0.11-0.13 with this few negatives.
Dev OOF reference column = single seed-42 outer split from the original ladder (optimistic; repeated-split means are 0.05-0.13 lower).
""")
    memory = joblib.Memory(Path("/private/tmp") / run.name, verbose=0); rows, preds = [], []
    sets = [c for k in a.sizes for c in itertools.combinations(CORE4, k)]
    for seqs in sets:
        name = "+".join(seqs)
        cols = [c for c in X.columns if any(c.startswith(s + "_original_") for s in seqs)]
        for fam in a.families:
            t0 = time.time()
            final_inner = list(StratifiedKFold(3, shuffle=True, random_state=999).split(X[cols], y))
            icc, est, params, score, audit, _ = sp.fit_one(fam, X[cols], y, final_inner, readers, memory, a.jobs)
            p = est.predict_proba(icc.transform(XL[cols]))[:, 1]
            lo, hi = boot(yL, p)
            dev_oof = np.nan
            if ref is not None:
                key = "+".join(sorted(seqs)); r = ref[(ref.k == key) & (ref.family == fam)]
                dev_oof = float(r.auc_pre.iloc[0]) if len(r) else np.nan
            rows.append(dict(sequence_set=name, n_sequences=len(seqs), family=fam, dev_oof_auc_seed42=dev_oof, lock_auc=float(roc_auc_score(yL, p)), lock_ci_low=lo, lock_ci_high=hi,
                             lock_average_precision=float(average_precision_score(yL, p)), lock_brier=float(brier_score_loss(yL, p)),
                             n_features_in_model=len(audit.get("selected_features", [])), inner_cv_auc=float(score), params=json.dumps({k: float(v) for k, v in params.items()}),
                             seconds=round(time.time() - t0)))
            preds += [dict(sequence_set=name, family=fam, PatientID=i, y_lock=int(o), prob=float(q)) for i, o, q in zip(lock_ids, yL, p)]
            pd.DataFrame(rows).to_csv(run / "lock_results.csv", index=False); pd.DataFrame(preds).to_csv(run / "lock_predictions.csv", index=False)
            print(f"[{name} | {fam}] lock AUC {rows[-1]['lock_auc']:.3f} (95% CI {lo:.2f}-{hi:.2f}) | dev OOF seed-42 {dev_oof:.3f} | {rows[-1]['seconds']}s", flush=True)
    (run / "manifest.json").write_text(json.dumps(dict(finished=datetime.now().isoformat(), n_dev=len(X), n_lock=len(XL), lock_labels_read=True, sklearn=sklearn.__version__,
        sheet=str(sheet), sha256_sheet=hashlib.sha256(sheet.read_bytes()).hexdigest(), n_models=len(rows)), indent=2))
    print("COMPLETED=" + str(run), flush=True)


if __name__ == "__main__":
    main()
