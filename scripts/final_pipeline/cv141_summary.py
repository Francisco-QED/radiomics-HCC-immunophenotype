"""Pooled out-of-fold performance on all 141 patients from saved OOF files: AUC of split-averaged probabilities (classification) or Harrell C of split-averaged risk (recurrence), patient-bootstrap 95% CI.
usage: cv141_summary.py --kind auc|c --glob '<run dirs>/oof_*.csv' --out table.csv [--endpoint tils_ge3|tls_ge2 for label only]"""
import argparse, glob, sys
sys.dont_write_bytecode = True
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score


def cindex(t, e, r):
    t, e, r = map(np.asarray, (t, e, r)); num = den = 0.0
    for i in np.flatnonzero(e == 1):
        m = t > t[i]; den += m.sum(); num += (r[m] < r[i]).sum() + 0.5 * (r[m] == r[i]).sum()
    return num / den if den else np.nan


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--kind", choices=["auc", "c"], required=True); ap.add_argument("--glob", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--split", default=None, help="recurrence split csv (PatientID,event,time_days) for --kind c"); ap.add_argument("--boot", type=int, default=2000); a = ap.parse_args()
    import run_guard
    files = run_guard.usable_files(a.glob, verbose=False); assert files, "no finished runs"; o = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    col = "prob" if a.kind == "auc" else "risk"; rng = np.random.default_rng(20261029); rows = []
    sp = pd.read_csv(a.split).set_index("PatientID") if a.kind == "c" else None
    if "sequence_set" not in o: o["sequence_set"] = "clinical"      # clinical-only survival runs have no sequence column
    for (ss, fam), d in o[o[col].notna()].groupby(["sequence_set", "family"]):
        avg = d.groupby("PatientID")[col].mean()
        if a.kind == "auc": y = d.groupby("PatientID").y.first().loc[avg.index].values; stat = lambda idx: roc_auc_score(y[idx], avg.values[idx])
        else: t = sp.loc[avg.index, "time_days"].values; e = sp.loc[avg.index, "event"].values.astype(int); stat = lambda idx: cindex(t[idx], e[idx], avg.values[idx])
        n = len(avg); est = stat(np.arange(n)); bs = []
        for _ in range(a.boot):
            idx = rng.integers(0, n, n)
            if a.kind == "auc" and len(set(y[idx])) < 2: continue
            bs.append(stat(idx))
        rows.append(dict(sequence_set=ss, family=fam, n_patients=n, n_repeats=d.repeat.nunique(), estimate=est, ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5)))
    pd.DataFrame(rows).sort_values("estimate", ascending=False).to_csv(a.out, index=False); print("saved", a.out, len(rows), "models")


if __name__ == "__main__":
    main()
