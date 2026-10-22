"""Numeric clinical / qualitative-MRI matrix for the recurrence analysis (all 141 patients; the split decides who is dev).

Source: databases/SA1-2_clinical_imaging_v4.xlsx (built from mastersheet.xlsx tabs SA1 + SA2-post; v3 used SA2-pre and has stale SA2 qualitative values). Writes <out>/clinical_matrix.csv and tiers.json.
Tiers (stated in advance):
  pre   known before surgery: demographics, etiology, cirrhosis, log AFP, max tumor size on PVP, qualitative MRI, treatment as a categorical (untreated reference; IO; IO+SBRT)
  path  known only after resection: tumor grade, pathologic T stage (SA2 tumors are ypT, i.e. post-treatment)
Dropped by policy: tumor necrosis (almost always none), enhancement / morphology / restricted diffusion (minority < 10%), race / ethnicity /
anatomic location (sparse free-text levels), free-text comments, pathology size (96 missing), MVI from comments (73 missing).
Macrovascular invasion (9.6% minority) is KEPT despite the 10% rule: it is a recognized recurrence predictor.
Missing values stay NaN here; imputation (median) happens inside every CV fold.
"""
import argparse, json, re, sys
from pathlib import Path
sys.dont_write_bytecode = True
import numpy as np, pandas as pd

ROOT = Path("/Users/Francisco/Library/CloudStorage/OneDrive-TheMountSinaiHospital/Regeneron retrospective radiomics")


def yn(s):
    """Yes/No coding robust to case and suffixes (e.g. 'No FF'); anything else -> NaN."""
    t = s.astype(str).str.strip().str.lower()
    return pd.Series(np.where(t.str.startswith("y"), 1.0, np.where(t.str.startswith("n") & (t != "nan"), 0.0, np.nan)), index=s.index)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mastersheet", default=str(ROOT / "databases/mastersheet.xlsx"))
    ap.add_argument("--clinical", default=str(ROOT / "databases/SA1-2_clinical_imaging_v4.xlsx"))
    ap.add_argument("--split", default=str(ROOT / "Results/main_pipeline_claude/main_pipeline_recurrence/split/recurrence_split.csv"))
    ap.add_argument("--out", default=str(ROOT / "Results/main_pipeline_claude/main_pipeline_recurrence/clinical"))
    a = ap.parse_args()
    c = pd.read_excel(a.clinical).set_index("PatientID"); sp = pd.read_csv(a.split).set_index("PatientID")
    col = lambda start: next(k for k in c.columns if k.startswith(start))
    M = pd.DataFrame(index=c.index)
    M["age"] = c["Age"].astype(float)
    M["male"] = (c["Gender"] == "M").astype(float)
    M["bmi"] = c["BMI"].astype(float)
    et = c[col("Etiology")].astype(str).str.upper()
    M["etiology_hbv"] = et.str.contains("HBV").astype(float)
    M["etiology_hcv"] = (et.str.contains("HCV") & ~et.str.contains("HBV")).astype(float)
    M["cirrhosis"] = yn(c["Cirrhosis"])
    M["log_afp"] = np.log10(c["Serum AFP"].astype(float).clip(lower=0.5))
    M["tumor_size_pvp_cm"] = c[col("Maximum tumor size on T1")].astype(float)
    M["capsule"] = yn(c[col("Capsule")])
    M["margin_irregular"] = c[col("Margin")].map(lambda v: np.nan if pd.isna(v) else float(str(v).lower().replace("-", " ").strip() != "well defined"))   # the SA2-post tab spells it "well-defined"
    M["mosaic"] = yn(c[col("Mosaic")])
    M["internal_arteries"] = yn(c[col("Internal arteries")])
    M["fat"] = yn(c[col("Fat content")])
    M["t2_hyper"] = c[col("T2 signal")].map(lambda v: np.nan if pd.isna(v) else (0.0 if str(v).strip().lower() in ("iso", "hypo") else 1.0))   # Mild / Mild Hyper / Hyper = hyperintense (the sheet defines Mild as mild hyperintensity)
    M["hbp_hypo"] = c[col("HBP signal")].map(lambda v: np.nan if pd.isna(v) else float(str(v).lower() == "hypo"))
    M["macrovascular_invasion"] = yn(c[col("Macrovascular")])
    M["lirads5"] = c[col("LI-RADS")].map(lambda v: np.nan if pd.isna(v) else float(str(v).strip().startswith("5")))
    tr = c["Treatment"].astype(str).str.strip().str.upper()
    assert tr.isin(["N", "IO", "IO+SBRT"]).all(), tr.value_counts().to_dict()
    M["treat_io"] = (tr == "IO").astype(float); M["treat_io_sbrt"] = (tr == "IO+SBRT").astype(float)
    pre = list(M.columns)
    M["cohort_sa2"] = sp["cohort"].reindex(M.index).map({"SA1": 0.0, "SA2": 1.0})      # for fold stratification only; not a model variable
    # tumor grade: mastersheet column BM ('Tumor_differentiation', Second pathology section), SA1 + SA2-post tabs; G1-G4 -> 1-4, 'G2/G3' -> 2.5
    gm = {}
    for tab in ("SA1", "SA2-post"):
        raw = pd.read_excel(a.mastersheet, sheet_name=tab, header=None); assert str(raw.iloc[1, 64]) == "Tumor_differentiation", str(raw.iloc[1, 64])
        body = raw[raw.iloc[:, 0].astype(str).str.match(r"^P\d{3}$")]; gm.update({str(i): v for i, v in zip(body.iloc[:, 0], body.iloc[:, 64])})
    conv = lambda v: np.nan if pd.isna(v) else (np.mean([float(x[1:]) for x in str(v).strip().upper().split("/")]))
    M["grade"] = pd.Series({k: conv(v) for k, v in gm.items()}).reindex(M.index)
    M["pT_stage"] = c["Tumor stage"].map(lambda v: np.nan if pd.isna(v) else (float(m.group(1)) if (m := re.search(r"T(\d)", str(v))) else np.nan))
    path = ["grade", "pT_stage"]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    M.to_csv(out / "clinical_matrix.csv"); (out / "tiers.json").write_text(json.dumps(dict(pre=pre, pre_grade=pre + ["grade"], pre_plus_path=pre + path, path_only=path), indent=2))
    print(M.shape, "| missing per column:"); print(M.isna().sum()[M.isna().sum() > 0].to_string())
    dev = sp.index[sp["set"] == "dev"]
    print("dev minority fraction (binary):"); print(M.loc[dev, [k for k in M if M[k].dropna().isin([0, 1]).all()]].mean().round(2).to_string())


if __name__ == "__main__":
    main()
