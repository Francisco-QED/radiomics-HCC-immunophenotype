"""Assemble the presentation tables and ROC figure for TILs >= 2 from the model tables written by lock_report_tils.py / tils_clinical_only.py.

Model per sequence combo = the family (ridge / lasso / elastic net) with the highest dev OOF AUC (mean over repeats); this uses dev data only, never lock.
Tables: (1) radiomics, top 10 of the 14 combos by that OOF AUC, with lock performance; (2) radiomics + clinico-radiologic, all 14 combos;
(3) clinico-radiologic only. Every row carries 95% CIs: OOF AUC by patient bootstrap of the repeat-averaged out-of-fold probability, sensitivity / specificity by Wilson,
lock AUC by stratified bootstrap. Sensitivity / specificity use the Youden threshold derived from the dev OOF predictions (never from lock).
ROC figure: best model of each category (radiomics, radiomics + clinico-radiologic, clinico-radiologic only), dev OOF (repeat-averaged) and lock.
"""
import argparse, glob, sys
from pathlib import Path
sys.dont_write_bytecode = True
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path("/Users/Francisco/Library/CloudStorage/OneDrive-TheMountSinaiHospital/Regeneron retrospective radiomics")
FAM = {"ridge": "Ridge", "lasso": "Lasso", "elastic_net": "Elastic net"}


def ci(v, lo, hi, p=2):
    return f"{v:.{p}f} ({lo:.{p}f}-{hi:.{p}f})"


def best_per_combo(df):
    return df.loc[df.groupby("sequence_set").oof_auc_repeat_mean.idxmax()].sort_values("oof_auc_repeat_mean", ascending=False).reset_index(drop=True)


def present(df, with_clin=False):
    rows = []
    for i, r in df.iterrows():
        degenerate = min(r.oof_sens, r.oof_spec, r.lock_sens, r.lock_spec) <= 0.05          # classifies (almost) everyone one way at the dev-derived threshold
        d = {"Rank": i + 1, "Sequences": r.sequence_set.replace("+", " + "), "Model": FAM.get(r.family, r.family) + (" †" if degenerate else "")}
        if with_clin: d["Clin. vars in model"] = int(r.n_clinical_in_model)
        d.update({"OOF AUC, mean ± SD": f"{r.oof_auc_repeat_mean:.2f} ± {r.oof_auc_repeat_sd:.2f}", "OOF AUC (95% CI)": ci(r.oof_auc_avgprob, r.oof_auc_ci_lo, r.oof_auc_ci_hi),
                  "OOF sensitivity (95% CI)": ci(r.oof_sens, r.oof_sens_ci_lo, r.oof_sens_ci_hi), "OOF specificity (95% CI)": ci(r.oof_spec, r.oof_spec_ci_lo, r.oof_spec_ci_hi),
                  "Lock AUC (95% CI)": ci(r.lock_auc, r.lock_auc_ci_lo, r.lock_auc_ci_hi), "Lock sensitivity (95% CI)": ci(r.lock_sens, r.lock_sens_ci_lo, r.lock_sens_ci_hi),
                  "Lock specificity (95% CI)": ci(r.lock_spec, r.lock_spec_ci_lo, r.lock_spec_ci_hi)})
        rows.append(d)
    return pd.DataFrame(rows)


def oof_avg(pattern, seqset, family, dev_order=None):
    o = pd.concat([pd.read_csv(f) for f in glob.glob(pattern)]); o = o[(o.sequence_set == seqset) & (o.family == family) & o.prob.notna()]
    return o.groupby("PatientID").agg(y=("y", "first"), prob=("prob", "mean"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "Results/main_pipeline_claude/presentation_20261020/tils_ge2"))
    a = ap.parse_args(); d = Path(a.dir)
    rad = pd.read_csv(d / "tils_ge2_model_table_radiomics.csv"); com = pd.read_csv(d / "tils_ge2_model_table_radiomics_clinical.csv"); cli = pd.read_csv(d / "tils_ge2_model_table_clinical.csv")
    rb, cb = best_per_combo(rad), best_per_combo(com); clb = cli.sort_values("oof_auc_repeat_mean", ascending=False).reset_index(drop=True)
    t1, t2, t3 = present(rb.head(10)), present(cb, with_clin=True), present(clb)
    t2.insert(3, "Radiomics-only OOF AUC", [f"{rb.set_index('sequence_set').oof_auc_repeat_mean.get(s, np.nan):.2f}" for s in cb.sequence_set])
    t3["Sequences"] = "clinico-radiologic only"
    with pd.ExcelWriter(d / "tils_ge2_tables.xlsx") as xw:
        t1.to_excel(xw, sheet_name="radiomics_top10", index=False); t2.to_excel(xw, sheet_name="radiomics_plus_clinical_14", index=False); t3.to_excel(xw, sheet_name="clinical_only", index=False)
        present(rad.sort_values(["sequence_set", "family"]).reset_index(drop=True)).to_excel(xw, sheet_name="supp_radiomics_all42", index=False)
        present(com.sort_values(["sequence_set", "family"]).reset_index(drop=True), with_clin=True).to_excel(xw, sheet_name="supp_combined_all42", index=False)
    t1.to_csv(d / "table_radiomics_top10.csv", index=False); t2.to_csv(d / "table_radiomics_plus_clinical_14.csv", index=False); t3.to_csv(d / "table_clinical_only.csv", index=False)
    cats = [("Radiomics", rb.iloc[0], str(ROOT / "Results/main_pipeline_claude/main_pipeline_tils/ladder_post_20261021/tilsGE2_post_*/oof_predictions.csv"), "radiomics", "#1f77b4"),
            ("Radiomics + clinico-radiologic", cb.iloc[0], str(ROOT / "Results/main_pipeline_claude/main_pipeline_tils/ladder_clinrad_post_20261021/tilsGE2cr_post_*/oof_predictions.csv"), "radiomics_clinical", "#d62728"),
            ("Clinico-radiologic", clb.iloc[0], str(ROOT / "Results/main_pipeline_claude/main_pipeline_tils/clinical_only_v4_20261021/oof_predictions.csv"), "clinical", "#2ca02c")]
    fig, ax = plt.subplots(1, 2, figsize=(12, 6.6)); best = []
    for name, r, pat, tag, col in cats:
        label = f"{name}: {r.sequence_set.replace('+', ' + ') if tag != 'clinical' else 'all variables'}, {FAM[r.family].lower()}"
        o = oof_avg(pat, r.sequence_set, r.family); fpr, tpr, _ = roc_curve(o.y, o.prob)
        ax[0].plot(fpr, tpr, color=col, lw=2, label=f"{label}\nAUC {ci(r.oof_auc_avgprob, r.oof_auc_ci_lo, r.oof_auc_ci_hi)}")
        lp = pd.read_csv(d / "models" / tag / f"{r.sequence_set.replace('+', '_') if tag != 'clinical' else 'clinicoradiologic'}__{r.family}" / "lock_predictions.csv"); fpr, tpr, _ = roc_curve(lp.y_lock, lp.prob)
        ax[1].plot(fpr, tpr, color=col, lw=2, label=f"AUC {ci(r.lock_auc, r.lock_auc_ci_lo, r.lock_auc_ci_hi)}")
        best.append(dict(category=name, sequences=r.sequence_set, family=r.family, oof_auc_mean=r.oof_auc_repeat_mean, lock_auc=r.lock_auc, tag=tag))
    for k, t in enumerate(("Development set: out-of-fold (repeat-averaged), n=111", "Lock set (one-shot), n=30")):
        ax[k].plot([0, 1], [0, 1], "--", color="gray", lw=1); ax[k].set_xlabel("1 - specificity"); ax[k].set_ylabel("Sensitivity"); ax[k].set_title(t, fontsize=11); ax[k].legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), fontsize=8.5, frameon=False)
        ax[k].set_aspect("equal"); ax[k].spines[["top", "right"]].set_visible(False)
    plt.suptitle("TILs >= 2: best model of each category (selected on dev OOF AUC)", fontsize=12, y=1.03); plt.tight_layout(); plt.savefig(d / "roc_best_per_category.png", dpi=200, bbox_inches="tight"); plt.close()
    pd.DataFrame(best).to_csv(d / "best_per_category.csv", index=False); print(pd.DataFrame(best).to_string(index=False)); print("tables:", t1.shape, t2.shape, t3.shape)


if __name__ == "__main__":
    main()
