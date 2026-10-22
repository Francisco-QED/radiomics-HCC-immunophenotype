"""Dev/lock split for the recurrence endpoint (separate from the TILs/TLS split: different usable pool and different stratification).

Usable pool = patients with a recurrence outcome (exclude == NaN). Double-reader (ICC) patients are forced into dev. The remaining patients are
split with test_size=30, stratified on event x cohort (SA1/SA2), seed 997 (as in the earlier recurrence pipeline). Lock outcomes are never evaluated here.
"""
import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path("/Users/Francisco/Library/CloudStorage/OneDrive-TheMountSinaiHospital/Regeneron retrospective radiomics")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outcomes", default=str(ROOT / "databases/recurrence_outcomes_v3.xlsx"))
    ap.add_argument("--test-size", type=int, default=30)
    ap.add_argument("--seed", type=int, default=997)
    ap.add_argument("--out", default=str(ROOT / "Results/main_pipeline_claude/main_pipeline_recurrence/split/recurrence_split.csv"))
    a = ap.parse_args()
    o = pd.read_excel(a.outcomes); o = o[o.exclude.isna()].sort_values("PatientID").reset_index(drop=True)
    icc = set(pd.read_csv(ROOT / "Results/Radiomics_sheets/inter-reader/Reader1/R-Reader1_I-sitkBSpline_S-1mm_W-perseq.csv", index_col=0, usecols=[0]).index)
    o["strata"] = o.event.astype(int).astype(str) + "_" + o.cohort
    must = o.PatientID.isin(icc); rest = o[~must]
    _, lock = train_test_split(rest, test_size=a.test_size, stratify=rest.strata, random_state=a.seed)
    o["set"] = "dev"; o.loc[o.PatientID.isin(lock.PatientID), "set"] = "lock"
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); o[["PatientID", "cohort", "event", "time_days", "set"]].to_csv(a.out, index=False)
    print(f"usable {len(o)} | ICC patients forced to dev: {int(must.sum())}")
    print(o.groupby(["set", "cohort"]).agg(n=("event", "size"), events=("event", "sum")).to_string())
    print("dev events: %d/%d (%.0f%%) | lock events: %d/%d (%.0f%%)" % (o[o.set == "dev"].event.sum(), (o.set == "dev").sum(), 100 * o[o.set == "dev"].event.mean(), o[o.set == "lock"].event.sum(), (o.set == "lock").sum(), 100 * o[o.set == "lock"].event.mean()))
    print("wrote", a.out)
