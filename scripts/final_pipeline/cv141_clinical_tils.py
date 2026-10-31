"""Clinico-radiologic-only logistic models (ridge / lasso / elastic net) with repeated 5-fold CV on ALL 141 patients (no held-out set); saves oof_predictions.csv like the ladder.
Same variables, tuning and fold seeds as tils_clinical_only.py. Run with /usr/local/bin/python3 (scikit-learn >= 1.8)."""
import argparse, json, sys, warnings
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import clinical_model as cm
import lock_report_tils as lr
ROOT = lr.ROOT


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--endpoint", choices=["tils_ge2", "tils_ge3", "tls_ge2"], required=True); ap.add_argument("--out", required=True); ap.add_argument("--repeats", type=int, default=10); ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--col-key", default="tils_pre_grade"); ap.add_argument("--matrix", default=str(ROOT / "Results/main_pipeline_claude/main_pipeline_recurrence/clinical/clinical_matrix.csv"))
    ap.add_argument("--cols", default=str(ROOT / "Results/main_pipeline_claude/main_pipeline_recurrence/clinical/tils_cols.json")); a = ap.parse_args(); warnings.filterwarnings("ignore")
    sp = pd.read_csv(ROOT / "Results/main_pipeline_claude/main_pipeline/final/cohort_v3_141/split/split.csv").set_index("PatientID"); ids = sp.index.tolist()
    M = pd.read_csv(a.matrix, index_col=0); cols = json.load(open(a.cols))[a.col_key]; X = M.loc[ids, cols]
    if a.endpoint == "tls_ge2": y = (pd.read_excel(ROOT / "databases/immunophenotype_outcomes_v4.xlsx").set_index("PatientID")["TLS_resection"].loc[ids] >= 2).astype(int)
    else: y = (sp["TILs_resection"] >= (3 if a.endpoint == "tils_ge3" else 2)).astype(int)
    sa2 = set(pd.read_csv(ROOT / "Results/Radiomics_sheets/SA2_pre/R-SA2_I-sitkBSpline_S-1mm_W-perseq.csv", index_col=0, usecols=[0]).index)      # same cohort definition as the radiomics ladders
    strata = y.astype(str) + "_" + pd.Series(["SA2" if i in sa2 else "SA1" for i in ids], index=ids); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); rows = []
    print(f"all patients {X.shape} pos {int(y.sum())}", flush=True)
    for fam in ("ridge", "lasso", "elastic_net"):
        for rep in range(a.repeats):
            outer = list(StratifiedKFold(5, shuffle=True, random_state=42 + rep).split(X, strata)); pred = cm.fit_predict_nested(X, y, fam, outer, rep, a.jobs)
            rows.append(pd.DataFrame(dict(sequence_set="clinicoradiologic", family=fam, repeat=rep, PatientID=y.index, y=y.values, prob=pred))); pd.concat(rows).to_csv(out / "oof_predictions.csv", index=False)
            print(f"[clinical | {fam}] repeat {rep}: AUC {roc_auc_score(y, pred):.3f}", flush=True)
    (out / "summary.csv").write_text("done\n"); print("COMPLETED", flush=True)


if __name__ == "__main__":
    main()
