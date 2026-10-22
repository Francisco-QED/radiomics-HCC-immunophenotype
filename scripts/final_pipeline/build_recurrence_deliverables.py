"""Presentation tables and ROC figure for time-to-recurrence models, from the model tables written by lock_report_recurrence.py.

Model per sequence combo = family with the highest dev OOF C-index (mean over repeats): development data only. Tables per category, each as an OOF table and a lock table:
  radiomics (top 10 of 14), radiomics + clinico-radiologic (all 14), clinico-radiologic only. Measures: Harrell C, time-dependent AUC at 1 / 2 / 3 years, and sensitivity /
  specificity for 2-year recurrence at the Youden threshold of the dev OOF risk; all with 95% CIs (patient bootstrap for C and AUC, Wilson for sens / spec).
ROC figure: best model of each category for 2-year recurrence (binary status among patients with known 2-year status), dev OOF (repeat-averaged risk) and lock.
The recurrence lock set was opened earlier by another session: lock numbers are descriptive.
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
    return df.loc[df.groupby("sequence_set").oof_c_repeat_mean.idxmax()].sort_values("oof_c_repeat_mean", ascending=False).reset_index(drop=True)


def tables(df, name_fn):
    oof, lock = [], []
    for i, r in df.iterrows():
        degenerate = min(r.oof_sens, r.oof_spec, r.lock_sens, r.lock_spec) <= 0.05
        base = {"Rank": i + 1, "Sequences": name_fn(r), "Model": FAM.get(r.family, r.family) + (" †" if degenerate else "")}
        oof.append({**base, "C-index, mean ± SD": f"{r.oof_c_repeat_mean:.2f} ± {r.oof_c_repeat_sd:.2f}", "C-index (95% CI)": ci(r.oof_c_avg, r.oof_c_ci_lo, r.oof_c_ci_hi),
                    **{f"AUC {y}y (95% CI)": ci(r[f"oof_auc{y}y"], r[f"oof_auc{y}y_ci_lo"], r[f"oof_auc{y}y_ci_hi"]) for y in (1, 2, 3)},
                    "Sens, 2y (95% CI)": ci(r.oof_sens, r.oof_sens_ci_lo, r.oof_sens_ci_hi), "Spec, 2y (95% CI)": ci(r.oof_spec, r.oof_spec_ci_lo, r.oof_spec_ci_hi)})
        lock.append({**base, "C-index (95% CI)": ci(r.lock_c, r.lock_c_ci_lo, r.lock_c_ci_hi),
                     **{f"AUC {y}y (95% CI)": ci(r[f"lock_auc{y}y"], r[f"lock_auc{y}y_ci_lo"], r[f"lock_auc{y}y_ci_hi"]) for y in (1, 2, 3)},
                     "Sens, 2y (95% CI)": ci(r.lock_sens, r.lock_sens_ci_lo, r.lock_sens_ci_hi), "Spec, 2y (95% CI)": ci(r.lock_spec, r.lock_spec_ci_lo, r.lock_spec_ci_hi)})
    return pd.DataFrame(oof), pd.DataFrame(lock)


def boot_auc(y, p, n=1000, seed=20261020):
    rng = np.random.default_rng(seed); y = np.asarray(y); p = np.asarray(p); v = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(y): v.append(roc_auc_score(y[i], p[i]))
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", default=str(ROOT / "Results/main_pipeline_claude/presentation_20261020/recurrence")); a = ap.parse_args(); d = Path(a.dir)
    rad = pd.read_csv(d / "recurrence_model_table_radiomics.csv"); com = pd.read_csv(d / "recurrence_model_table_radiomics_clinical.csv"); cli = pd.read_csv(d / "recurrence_model_table_clinical.csv")
    rb, cb = best_per_combo(rad), best_per_combo(com); clb = cli.sort_values("oof_c_repeat_mean", ascending=False).reset_index(drop=True)
    name = lambda r: r.sequence_set.replace("+", " + ")
    t = {"radiomics_top10": tables(rb.head(10), name), "radiomics_plus_clinical_14": tables(cb, name), "clinical_only": tables(clb, lambda r: "clinico-radiologic only")}
    with pd.ExcelWriter(d / "recurrence_tables.xlsx") as xw:
        for k, (o, l) in t.items():
            o.to_csv(d / f"table_{k}_OOF.csv", index=False); l.to_csv(d / f"table_{k}_lock.csv", index=False); o.to_excel(xw, sheet_name=f"{k}_OOF"[:31], index=False); l.to_excel(xw, sheet_name=f"{k}_lock"[:31], index=False)
        tables(rad.sort_values(["sequence_set", "family"]).reset_index(drop=True), name)[0].to_excel(xw, sheet_name="supp_radiomics_all42_OOF", index=False)
        tables(com.sort_values(["sequence_set", "family"]).reset_index(drop=True), name)[0].to_excel(xw, sheet_name="supp_combined_all42_OOF", index=False)
    sp = pd.read_csv(ROOT / "Results/main_pipeline_claude/main_pipeline_recurrence/split/recurrence_split.csv").set_index("PatientID"); r2 = pd.read_excel(ROOT / "databases/recurrence_outcomes_v3.xlsx").set_index("PatientID")["recur_2yr"]
    dev_ids = sp.index[sp["set"] == "dev"].tolist()
    cats = [("Radiomics", rb.iloc[0], str(ROOT / "Results/main_pipeline_claude/main_pipeline_recurrence/ladder14_radiomics_post_20261021/recA_post_none_*/oof_risk.csv"), "radiomics", "#1f77b4"),
            ("Radiomics + clinico-radiologic", cb.iloc[0], str(ROOT / "Results/main_pipeline_claude/main_pipeline_recurrence/ladder14_radiomics_clinical_post_20261021/recB_post_forced_*/oof_risk.csv"), "radiomics_clinical", "#d62728"),
            ("Clinico-radiologic", clb.iloc[0], str(ROOT / "Results/main_pipeline_claude/main_pipeline_recurrence/clinical/clinsurv_pre_*/oof_risk.csv"), "clinical", "#2ca02c")]
    fig, ax = plt.subplots(1, 2, figsize=(12, 6.6)); best = []
    for nm, r, pat, tag, col in cats:
        o = pd.concat([pd.read_csv(f) for f in glob.glob(pat)]); o = o[(o.family == r.family) & o.risk.notna()]
        if "sequence_set" in o: o = o[o.sequence_set == r.sequence_set]
        avg = o.groupby("PatientID").risk.mean().reindex(dev_ids); st = r2.reindex(dev_ids); k = st.notna().values
        fpr, tpr, _ = roc_curve(st[k].astype(int), avg[k]); auc = roc_auc_score(st[k].astype(int), avg[k]); lo, hi = boot_auc(st[k].astype(int), avg[k].values)
        label = f"{nm}: {r.sequence_set.replace('+', ' + ') if tag != 'clinical' else 'all variables'}, {FAM[r.family].lower()}"
        ax[0].plot(fpr, tpr, color=col, lw=2, label=f"{label}\nAUC {ci(auc, lo, hi)}")
        folder = f"{r.sequence_set.replace('+', '_')}__{r.family}" if tag != "clinical" else f"clinicoradiologic__{r.family}"
        lp = pd.read_csv(d / "models" / tag / folder / "lock_predictions.csv"); stl = r2.reindex(lp.PatientID).values; kl = ~pd.isna(stl)
        fpr, tpr, _ = roc_curve(stl[kl].astype(int), lp.risk.values[kl]); auc_l = roc_auc_score(stl[kl].astype(int), lp.risk.values[kl]); llo, lhi = boot_auc(stl[kl].astype(int), lp.risk.values[kl])
        ax[1].plot(fpr, tpr, color=col, lw=2, label=f"AUC {ci(auc_l, llo, lhi)}")
        best.append(dict(category=nm, sequences=r.sequence_set, family=r.family, oof_c_mean=r.oof_c_repeat_mean, lock_c=r.lock_c, tag=tag, roc2y_oof_auc=auc, roc2y_lock_auc=auc_l))
    for k_, tt in enumerate((f"Development set: out-of-fold risk (repeat-averaged), n={len(dev_ids)}", "Lock set (exploratory), n=30")):
        ax[k_].plot([0, 1], [0, 1], "--", color="gray", lw=1); ax[k_].set_xlabel("1 - specificity"); ax[k_].set_ylabel("Sensitivity"); ax[k_].set_title(tt, fontsize=11)
        ax[k_].legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), fontsize=8.5, frameon=False); ax[k_].set_aspect("equal"); ax[k_].spines[["top", "right"]].set_visible(False)
    plt.suptitle("2-year recurrence: best model of each category (selected on dev OOF C-index)", fontsize=12, y=1.03); plt.tight_layout(); plt.savefig(d / "roc2y_best_per_category.png", dpi=200, bbox_inches="tight"); plt.close()
    pd.DataFrame(best).to_csv(d / "best_per_category.csv", index=False); print(pd.DataFrame(best).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
