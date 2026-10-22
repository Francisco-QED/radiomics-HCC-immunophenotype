"""One dev/lock split per run, shared by every design (cutoff / ordinal) and every sequence combo.

Rules (same as the original pipeline, except the stratification variable):
  * inter-reader (ICC) patients are always dev -- they are needed for the ICC filter;
  * the remaining patients are split stratified on the 4 TILs levels (0/1/2/3), which also balances
    every binary cutoff derived from them;
  * deterministic: patients sorted by ID before splitting, fixed seed.
Writes split.csv (PatientID,set,TILs_resection) and split_summary.csv next to it.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


def make_split(outcomes_xlsx, icc_ids, label_col, test_size, seed, id_col="PatientID", feature_ids=None):
    o = pd.read_excel(outcomes_xlsx)[[id_col, label_col]]
    n0 = len(o)
    o = o[o[label_col].notna()]
    if feature_ids is not None:
        missing = sorted(set(feature_ids) - set(o[id_col]))
        if missing:
            raise ValueError(f"Patients with features but no {label_col}: {missing}")
        o = o[o[id_col].isin(feature_ids)]
    o = o.sort_values(id_col).reset_index(drop=True)
    o["level"] = o[label_col].round().astype(int)
    must = o[id_col].isin(set(icc_ids))
    rest = o[~must]
    counts = rest.level.value_counts()
    if (counts < 2).any() or test_size >= len(rest):
        raise ValueError(f"Cannot stratify: level counts in the splittable pool = {counts.to_dict()}")
    _, lock = train_test_split(rest, test_size=test_size, stratify=rest.level, random_state=seed)
    o["set"] = "dev"
    o.loc[o[id_col].isin(lock[id_col]), "set"] = "lock"
    return o.drop(columns="level"), n0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    root = Path(os.environ.get("RADIOMICS_ROOT", cfg["paths"]["root"])).expanduser()
    db = root / cfg["paths"]["inputs"]["databases"]
    sheets = root / cfg["paths"]["inputs"]["radiomics_sheets"]
    bw = cfg["data"]["binwidth"]
    icc = pd.read_csv(sheets / f"{cfg['data']['reader1_cohort']}/R-Reader1_I-sitkBSpline_S-1mm_W-{bw}.csv",
                      index_col=0, usecols=[0]).index
    feats = []
    for c in cfg["data"]["features"]:
        n = c["cohort"]
        stem = {"SA2_pre": "SA2", "SA2_post": "SA2-post"}.get(n, n)  # folder name -> file stem
        feats += pd.read_csv(sheets / f"{n}/R-{stem}_I-sitkBSpline_S-1mm_W-{bw}.csv", index_col=0, usecols=[0]).index.tolist()
    if cfg["data"].get("exclude_unscored", False):
        o = pd.read_excel(db / cfg["data"]["outcomes_xlsx"]).set_index("PatientID")[cfg["data"]["label_col"]]
        feats = [f for f in feats if pd.notna(o.get(f, np.nan))]
    split, n0 = make_split(db / cfg["data"]["outcomes_xlsx"], icc, cfg["data"]["label_col"],
                           cfg["preprocessing"]["split"]["test_size"], cfg["run"]["seeds"]["partition"],
                           feature_ids=feats)
    out = root / cfg["paths"]["outputs"] / cfg["run"]["name"] / "split"
    out.mkdir(parents=True, exist_ok=True)
    split.to_csv(out / "split.csv", index=False)
    lab = cfg["data"]["label_col"]
    summ = split.groupby(["set", lab]).size().unstack(fill_value=0)
    summ["n"] = summ.sum(axis=1)
    summ.to_csv(out / "split_summary.csv")
    print(f"patients with features: {len(feats)}; outcomes rows: {n0}; assigned: {len(split)}")
    print(summ.to_string())
    for c in (3, 2):
        t = split.assign(y=(split[lab] >= c).astype(int)).groupby("set").y.agg(["sum", "count"])
        print(f"cutoff {c}: " + "; ".join(f"{s}: {int(r['sum'])}/{int(r['count'])} positive" for s, r in t.iterrows()))
    print("ICC patients in lock:", sorted(set(icc) & set(split[split.set == 'lock'].PatientID)))
    print("wrote", out / "split.csv")


if __name__ == "__main__":
    main()
