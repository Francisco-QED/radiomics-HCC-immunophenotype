"""Kaplan-Meier curves (median-split risk groups, log-rank p, HR) and time-dependent AUC curves for the best recurrence model of each category.

Like slide 23 of the 08-17-26 deck: one KM figure and one time-dependent-AUC figure per model, for the lock set and for the development set.
Risk groups: high vs low risk = above vs at/below the median risk of the set being plotted (unsupervised; no labels used). Development: repeat-averaged out-of-fold risk.
Lock: risk from the final dev-fit model applied once. Time-dependent AUC(t): censoring-weighted (IPCW, censoring distribution estimated on the development set), grid every 60 days.
The recurrence lock set was opened earlier in this project, so the lock panels are exploratory.
"""
import argparse, glob, json, sys
from pathlib import Path
sys.dont_write_bytecode = True
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from sksurv.metrics import cumulative_dynamic_auc
from sksurv.util import Surv

ROOT = Path("/Users/Francisco/Library/CloudStorage/OneDrive-TheMountSinaiHospital/Regeneron retrospective radiomics")
REC = ROOT / "Results/main_pipeline_claude/main_pipeline_recurrence"
OOF = {"radiomics": REC / "ladder14_radiomics_post_20261021/recA_post_none_*/oof_risk.csv", "radiomics_clinical": REC / "ladder14_radiomics_clinical_post_20261021/recB_post_forced_*/oof_risk.csv",
       "clinical": REC / "clinical/clinsurv_pre_*/oof_risk.csv"}
NAMES = {"radiomics": "Radiomics", "radiomics_clinical": "Radiomics + clinico-radiologic", "clinical": "Clinico-radiologic"}
COL = {"high": "#d62728", "low": "#1f77b4"}


def km_figure(time, event, risk, title, path):
    hi = risk > np.median(risk); T = time / 365.25
    lr = logrank_test(T[hi], T[~hi], event[hi], event[~hi])
    cox = CoxPHFitter().fit(pd.DataFrame({"T": T, "E": event.astype(int), "high": hi.astype(int)}), "T", "E"); s = cox.summary.loc["high"]
    fig, (ax, tab) = plt.subplots(2, 1, figsize=(5.4, 4.1), sharex=True, gridspec_kw=dict(height_ratios=[3.4, 0.75], hspace=0.22))
    for name, m in (("High risk", hi), ("Low risk", ~hi)):
        k = KaplanMeierFitter().fit(T[m], event[m], label=f"{name} (n={int(m.sum())}, events={int(event[m].sum())})")
        k.plot_survival_function(ax=ax, ci_show=True, color=COL["high" if name == "High risk" else "low"], lw=2, show_censors=True, censor_styles={"ms": 4})
    ax.set_xlim(0, max(4.6, 0)); ticks = [t for t in range(0, 6) if t <= T.max()]
    tab.set_ylim(0, 3); tab.set_yticks([]); tab.set_xticks(ticks); tab.spines[["top", "right", "left"]].set_visible(False); tab.tick_params(axis="x", labelsize=8); tab.set_xlabel("Years from surgery", fontsize=9)
    tab.text(-0.07, 2.75, "No. at risk (cumulative events)", transform=tab.get_yaxis_transform(), ha="left", va="center", fontsize=7.5, color="dimgray")
    for yy, (name, m, col) in zip((1.75, 0.75), (("High", hi, COL["high"]), ("Low", ~hi, COL["low"]))):
        tab.text(-0.07, yy, name, transform=tab.get_yaxis_transform(), ha="right", va="center", fontsize=8, color=col, fontweight="bold")
        for t in ticks: tab.text(t, yy, f"{int((T[m] >= t).sum())} ({int(((T[m] <= t) & (event[m] == 1)).sum())})", ha="center", va="center", fontsize=8, color=col)
    ax.set_xlabel(""); ax.tick_params(labelbottom=False); ax.set_ylabel("Recurrence-free probability"); ax.set_ylim(0, 1.02); ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(title, fontsize=10.5); ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    ax.text(0.97, 0.97, f"log-rank p {'< 0.001' if lr.p_value < 0.001 else '= ' + format(lr.p_value, '.3f')}\nHR = {np.exp(s['coef']):.2f} ({np.exp(s['coef lower 95%']):.2f}-{np.exp(s['coef upper 95%']):.2f})", transform=ax.transAxes, ha="right", va="top", fontsize=9)
    plt.savefig(path, dpi=200, bbox_inches="tight"); plt.close()
    return float(lr.p_value), float(np.exp(s["coef"])), float(np.exp(s["coef lower 95%"])), float(np.exp(s["coef upper 95%"]))


def tdauc_figure(y_ref, y_set, risk, title, path):
    ev_t = y_set["time"][y_set["event"]]; tmax = min(ev_t.max() - 1, 1700.0); grid = np.arange(180.0, tmax, 60.0)
    auc, mean_auc = cumulative_dynamic_auc(y_ref, y_set, risk, grid)
    fig, (ax, tab) = plt.subplots(2, 1, figsize=(4.6, 2.5), sharex=True, gridspec_kw=dict(height_ratios=[3, 0.8], hspace=0.12)); ax.plot(grid / 365.25, auc, color="#2ca02c", lw=2); ax.axhline(mean_auc, ls="--", color="#2ca02c", lw=1); ax.axhline(0.5, ls=":", color="gray", lw=1)
    ax.set_ylim(0.2, 1.0); ax.set_ylabel("AUC(t)"); ax.spines[["top", "right"]].set_visible(False); ax.set_title(title, fontsize=9.5); ax.tick_params(labelbottom=False)
    ax.text(0.97, 0.06, f"mean AUC(t) = {mean_auc:.2f}", transform=ax.transAxes, ha="right", fontsize=9)
    T = np.asarray(y_set["time"], float) / 365.25; E = np.asarray(y_set["event"], bool); ticks = [t for t in (0.5, 1, 2, 3, 4, 5) if grid.min() / 365.25 - 0.02 <= t <= grid.max() / 365.25 + 0.05]
    tab.set_ylim(0, 2); tab.set_yticks([]); tab.set_xticks(ticks); tab.set_xticklabels([f"{t:g}" for t in ticks]); tab.spines[["top", "right", "left"]].set_visible(False); tab.tick_params(axis="x", labelsize=8); tab.set_xlabel("Years from surgery", fontsize=9)
    for yy, lab in ((1.5, "Cases"), (0.5, "Controls")): tab.text(-0.02, yy, lab, transform=tab.get_yaxis_transform(), ha="right", va="center", fontsize=7.5, color="dimgray")
    for t in ticks:
        tab.text(t, 1.5, str(int((E & (T <= t)).sum())), ha="center", va="center", fontsize=8); tab.text(t, 0.5, str(int((T > t).sum())), ha="center", va="center", fontsize=8)
    fig.subplots_adjust(left=0.2, right=0.97, top=0.9, bottom=0.2); plt.savefig(path, dpi=200); plt.close()
    return float(mean_auc)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", default=str(ROOT / "Results/main_pipeline_claude/presentation_20261020/recurrence")); ap.add_argument("--spec", default=None, help="json: best rows, model_dirs, oof globs, out dir"); a = ap.parse_args()
    d = Path(a.dir)
    spec = json.load(open(a.spec)) if a.spec else None
    out = Path(spec["out"]) if spec else d / "km"; out.mkdir(parents=True, exist_ok=True)
    mdirs = {k: Path(v) for k, v in spec["model_dirs"].items()} if spec else {t: d for t in OOF}; oofs = {k: v for k, v in spec["oof"].items()} if spec else {k: str(v) for k, v in OOF.items()}
    sp = pd.read_csv(REC / "split/recurrence_split.csv").set_index("PatientID"); dev = sp[sp["set"] == "dev"]; dev_ids = dev.index.tolist()
    yd = Surv.from_arrays(event=dev.event.astype(bool).values, time=dev.time_days.astype(float).values, name_event="event", name_time="time")
    best = pd.DataFrame(spec["best"]) if spec else pd.read_csv(d / "best_per_category.csv"); rows = []
    for _, r in best.iterrows():
        folder = f"{r.sequences.replace('+', '_')}__{r.family}" if r.tag != "clinical" else f"clinicoradiologic__{r.family}"
        label = f"{NAMES[r.tag]}: {r.sequences.replace('+', ' + ') if r.tag != 'clinical' else 'all variables'}, {r.family.replace('_', ' ')}"
        lp = pd.read_csv(mdirs[r.tag] / "models" / r.tag / folder / "lock_predictions.csv"); yl = Surv.from_arrays(event=lp.event.astype(bool).values, time=lp.time_days.astype(float).values, name_event="event", name_time="time")
        import run_guard
        o = pd.concat([pd.read_csv(f) for f in run_guard.usable_files(oofs[r.tag], verbose=False)]); o = o[(o.family == r.family) & o.risk.notna()]
        if "sequence_set" in o: o = o[o.sequence_set == r.sequences]
        avg = o.groupby("PatientID").risk.mean().reindex(dev_ids).values
        for setname, y_, risk_, tt in (("lock", yl, lp.risk.values, "Test set"), ("dev", yd, avg, "Development set, out-of-fold risk")):
            p, hr, lo, hi = km_figure(y_["time"], y_["event"], risk_, f"{label}\n{tt}", out / f"km_{r.tag}_{setname}.png")
            m = tdauc_figure(yd, y_, risk_, f"Time-dependent AUC: {tt}", out / f"tdauc_{r.tag}_{setname}.png")
            rows.append(dict(category=NAMES[r.tag], tag=r.tag, set=setname, model=label, logrank_p=p, hr=hr, hr_lo=lo, hr_hi=hi, mean_td_auc=m))
    pd.DataFrame(rows).to_csv(out / "km_summary.csv", index=False); print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
