"""Radiomics + clinical/qualitative variables, repeated CV with a FIXED procedure (ridge C=0.01, k=15), nothing tuned.

Variants (same outer folds, same ICC/QC/imputation/clustering/top-k as the 'gpt' compact ridge pipeline, step S0):
  V0_radiomics_only        radiomics clusters only (baseline)
  V1_forced_clinical_core  + 9 pre-specified clinical variables, always included (not ranked, not selected)
  V2_forced_all18          + all 18 clinical/qualitative variables from clinical_model.py, always included
  V3_compete_all18         the 18 variables enter the SAME top-k ranking as the radiomics clusters, ranked by univariate AUC
                           inside each training fold (the 'most powerful' ones are chosen without peeking)
ICC (reader reliability) is applied to radiomics only; clinical variables bypass it. Dev patients only; lock never read.
"""
import argparse, json, sys, time, warnings
from datetime import datetime
from pathlib import Path
sys.dont_write_bytecode = True
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ablation_ladder as al

ROOT = Path("/Users/Francisco/Library/CloudStorage/OneDrive-TheMountSinaiHospital/Regeneron retrospective radiomics")
SETS = {"tp": ("tp",), "pvp+tp": ("pvp", "tp"), "dwi_b2+tp": ("dwi_b2", "tp"), "dwi_b2+pvp+tp": ("dwi_b2", "pvp", "tp")}
VARIANTS = ["V0_radiomics_only", "V1_forced_clinical_core", "V2_forced_all18", "V3_compete_all18"]
K, C = 15, 0.01


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", choices=["post", "pre"], default="post")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--sets", nargs="+", default=list(SETS))
    ap.add_argument("--clinical-run", default=str(ROOT / "Results/main_pipeline_clinical/clinical_qualitative_models_20261016_170517"))
    ap.add_argument("--output-parent", default=str(ROOT / "Results/main_pipeline_clinical"))
    a = ap.parse_args(); warnings.filterwarnings("ignore")
    core = al.load_core(al.DEFAULT_CORE_DIR); Sel = al.make_selector_class(core)
    X0, y, dev_ids, lock_ids, split_path, feature_path, readers = core.load_development(ROOT)
    sheet = ROOT / ("Results/Radiomics_sheets/SA1_2_post/R-SA1_2-post_I-sitkBSpline_S-1mm_W-perseq.csv" if a.sheet == "post"
                    else "Results/Radiomics_sheets/SA1_2_pre/R-SA1_2_I-sitkBSpline_S-1mm_W-perseq.csv")
    X = pd.read_csv(sheet, index_col=0, usecols=lambda c: c == "PatientID" or "_original_" in c).loc[dev_ids]
    X = X.reindex(sorted(X.columns), axis=1).astype(float); assert X.shape == (110, 963), X.shape
    man = json.load(open(Path(a.clinical_run) / "manifest.json"))
    cm = pd.read_csv(Path(a.clinical_run) / "model_matrix_dev.csv", index_col=0).loc[dev_ids]
    assert (cm["TILs_ge2"].values == y.values).all(), "clinical matrix outcome does not match the radiomics outcome"
    core9, all18 = man["sets"]["clinical"], man["sets"]["clinical+qualitative_MRI"]
    clin = cm[all18].add_prefix("clin_"); clin9 = ["clin_" + c for c in core9]; clin18 = ["clin_" + c for c in all18]
    run = Path(a.output_parent) / f"radiomics_plus_clinical_{a.sheet}_{datetime.now():%Y%m%d_%H%M%S}"; run.mkdir(parents=True, exist_ok=False)
    (run / "manifest.json").write_text(json.dumps(dict(started=datetime.now().isoformat(), sheet=str(sheet), endpoint="TILs>=2", n_dev=110, lock_evaluated=False,
        procedure=f"fixed ridge C={C}, k={K}, no tuning; QC/median impute/within-sequence complete-linkage/top-k as gpt S0", variants=VARIANTS, sets=a.sets,
        clinical_core=clin9, clinical_all=clin18, repeats=a.repeats, clinical_source=a.clinical_run), indent=2))
    print("RUN_DIRECTORY=" + str(run), flush=True)
    base_cfg = al.STEPS["S0_theirs"]; rows, sel_rows = [], []
    for sname in a.sets:
        cols = sorted(c for c in X.columns if any(c.startswith(s + "_original_") for s in SETS[sname]))
        Xs = X[cols]
        for rep in range(a.repeats):
            t0 = time.time(); preds = {v: np.full(len(y), np.nan) for v in VARIANTS}; nclin = {v: [] for v in VARIANTS}
            for fold, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42 + rep).split(Xs, y), 1):
                icc = core.TechnicalICCFilter(*readers).fit(Xs.iloc[tr])
                Rtr, Rva, ytr = icc.transform(Xs.iloc[tr]), icc.transform(Xs.iloc[va]), y.iloc[tr]

                def one(v):
                    extra = {"V0_radiomics_only": [], "V1_forced_clinical_core": clin9, "V2_forced_all18": clin18, "V3_compete_all18": clin18}[v]
                    forced = tuple(extra) if v.startswith(("V1", "V2")) else ()
                    Ttr = pd.concat([Rtr, clin.loc[Rtr.index, extra]], axis=1) if extra else Rtr
                    Tva = pd.concat([Rva, clin.loc[Rva.index, extra]], axis=1) if extra else Rva
                    m = al.build_pipeline(core, Sel, base_cfg, forced=forced).set_params(select__k=K, model__C=C).fit(Ttr, ytr)
                    sel = m.named_steps["select"].selected_features_
                    return v, m.predict_proba(Tva)[:, 1], sum(f.startswith("clin_") for f in sel)
                for v, p, nc in Parallel(n_jobs=a.jobs, backend="threading")(delayed(one)(v) for v in VARIANTS):
                    preds[v][va] = p; nclin[v].append(nc)
            for v in VARIANTS:
                rows.append(dict(sequence_set=sname, variant=v, repeat=rep, auc=roc_auc_score(y, preds[v]), mean_clinical_vars_in_model=float(np.mean(nclin[v]))))
            pd.DataFrame(rows).to_csv(run / "per_repeat_auc.csv", index=False)
            print(f"[{sname}] repeat {rep}: " + " ".join(f"{v[:2]}={roc_auc_score(y, preds[v]):.3f}" for v in VARIANTS) + f"  ({time.time() - t0:.0f}s)", flush=True)
    D = pd.DataFrame(rows)
    S = D.groupby(["sequence_set", "variant"]).agg(mean_auc=("auc", "mean"), sd_auc=("auc", "std"), min_auc=("auc", "min"), max_auc=("auc", "max"),
                                                  mean_clinical_vars_in_model=("mean_clinical_vars_in_model", "mean")).reset_index()
    S.round(4).to_csv(run / "summary.csv", index=False)
    W = D.pivot_table(index=["sequence_set", "repeat"], columns="variant", values="auc"); pr = []
    for st in a.sets:
        for v in VARIANTS[1:]:
            d = W.loc[st][v] - W.loc[st][VARIANTS[0]]; se = d.std(ddof=1) / np.sqrt(len(d))
            pr.append(dict(sequence_set=st, variant=v, mean_delta_vs_V0=d.mean(), sd=d.std(ddof=1), pct_repeats_positive=(d > 0).mean(), t=d.mean() / se if se > 0 else np.nan))
    pd.DataFrame(pr).round(4).to_csv(run / "paired_delta_vs_radiomics_only.csv", index=False)
    print("COMPLETED=" + str(run), flush=True)


if __name__ == "__main__":
    main()
