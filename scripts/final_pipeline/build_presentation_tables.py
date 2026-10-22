"""Cohort tables for the presentation: clinical/demographic and qualitative-MRI variables, overall and by dev / lock set, for both splits.

Splits: TILs/TLS (split.csv, 111 dev / 30 lock) and recurrence (recurrence_split.csv, 110 dev / 30 lock; P121 has no follow-up and is excluded).
Continuous: median [IQR], Mann-Whitney U. Categorical: n (%) of non-missing, Fisher's exact for 2x2, chi-square otherwise (dagger = any expected count < 5).
Missing values are reported as a row. Category collapsing rules (stated here, applied identically to every table):
  Race: Asian = Chinese, Korean, Asian, Asian (Pacific Islander), Asian Indian, Pakistani, Vietnamese, Filipino; Black = all 'Black' spellings;
        Other = Other, Ugandan, Hispanic/Latino; Unknown = Unknown or blank.   Ethnicity: Hispanic/Latino = any Hispanic national origin.
  Etiology: a single cause is kept (HBV, HCV, MASH/MASLD/NASH/NAFLD, Alcohol); two or more causes or any other disease = Multiple/other; Unknown/blank = missing.
  LI-RADS: 2-5 kept; TIV, FNH, M and mixed labels = Other/atypical.   Restricted diffusion: Yes vs No/Mild.
  Treatment: untreated (SA1), immunotherapy, immunotherapy + SBRT (SA2), from the Treatment column of the v4 clinical table (mastersheet tabs SA1 + SA2-post).
"""
import argparse, re, sys, warnings
from pathlib import Path
sys.dont_write_bytecode = True
import numpy as np, pandas as pd
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu

ROOT = Path("/Users/Francisco/Library/CloudStorage/OneDrive-TheMountSinaiHospital/Regeneron retrospective radiomics")
ASIAN = {"chinese", "korean", "asian", "asian (pacific islander)", "asian indian", "pakistani", "vietnamese", "filipino"}
HISP = {"hispanic", "dominican", "spaniard", "puerto rican", "colombian", "mexican", "mexcian", "cuban", "salvadoran", "hispanic/latino"}


def race(v):
    if pd.isna(v) or str(v).strip().lower() == "unknown": return "Unknown"
    k = str(v).strip().lower()
    return "Asian" if k in ASIAN else "Black or African-American" if k.startswith("black") else "White" if k == "white" else "Other"


def ethnicity(v):
    if pd.isna(v) or str(v).strip().lower() == "unknown": return "Unknown"
    k = str(v).strip().lower()
    return "Non-Hispanic" if k == "non-hispanic" else "Hispanic/Latino" if k in HISP else "Unknown"


def etiology(v):
    if pd.isna(v) or str(v).lower().startswith("unk"): return np.nan
    toks = [t.strip().upper() for t in re.split(r"[,/]", str(v)) if t.strip()]
    if len(toks) != 1: return "Multiple/other"
    t = toks[0]
    return "HBV" if t == "HBV" else "HCV" if t == "HCV" else "MASH/MASLD/NASH/NAFLD" if t in ("MASH", "MASLD", "NASH", "NAFLD") else "Alcohol" if t.startswith("ETOH") or t == "ALCOHOL" else "Multiple/other"


def lirads(v):
    if pd.isna(v): return np.nan
    s = str(v).strip()
    return f"LR-{s}" if s in ("2", "3", "4", "5") else "Other/atypical"


def yn(s):
    t = s.astype(str).str.strip().str.lower()
    return pd.Series(np.where(t.str.startswith("y"), "Yes", np.where(t.str.startswith("n") & (t != "nan"), "No", None)), index=s.index, dtype=object)


def build_frame():
    c = pd.read_excel(ROOT / "databases/SA1-2_clinical_imaging_v4.xlsx").set_index("PatientID")
    col = lambda p: next(k for k in c.columns if k.startswith(p))
    o = pd.read_excel(ROOT / "databases/immunophenotype_outcomes_v4.xlsx").set_index("PatientID")
    r = pd.read_excel(ROOT / "databases/recurrence_outcomes_v4.xlsx").set_index("PatientID")
    D = pd.DataFrame(index=c.index)
    # clinical / demographic
    D["Age, years"] = c["Age"].astype(float)
    D["Gender"] = c["Gender"].map({"M": "Male", "F": "Female"})
    D["BMI, kg/m2"] = c["BMI"].astype(float)
    D["Race"] = c["Race"].map(race); D["Ethnicity"] = c["Ethnicity"].map(ethnicity)
    D["Etiology of liver disease"] = c[col("Etiology")].map(etiology)
    D["Cirrhosis"] = yn(c["Cirrhosis"]); D["Serum AFP, ng/mL"] = c["Serum AFP"].astype(float)
    D["Tumor grade (pathology)"] = c["Tumor grade"].map(lambda v: np.nan if pd.isna(v) else f"G{int(v)}")
    # qualitative MRI
    D["Maximum tumor size on T1 PVP, cm"] = c[col("Maximum tumor size on T1")].astype(float)
    D["Morphology"] = c[col("Morphology")]; D["Capsule (on PVP)"] = yn(c[col("Capsule")])
    D["Margin"] = c[col("Margin")].map(lambda v: np.nan if pd.isna(v) else ("Infiltrative" if str(v).lower().startswith(("infil", "inflit")) else "Irregular" if str(v).lower().startswith("irreg") else "Well defined"))
    D["Mosaic appearance"] = yn(c[col("Mosaic")]); D["Internal arteries"] = yn(c[col("Internal arteries")])
    D["Macrovascular invasion"] = yn(c[col("Macrovascular")]); D["Tumor necrosis"] = yn(c[col("Tumor necrosis")]); D["Fat content"] = yn(c[col("Fat content")])
    D["T2 signal"] = c[col("T2 signal")].map(lambda v: np.nan if pd.isna(v) else {"hypo": "Hypo", "iso": "Iso", "mild": "Mild hyper (incl. 'Mild Hyper')", "mild hyper": "Mild hyper (incl. 'Mild Hyper')", "hyper": "Marked hyper"}.get(str(v).strip().lower(), str(v)))
    D["Restricted diffusion"] = c[col("Restricted diffusion")].map(lambda v: np.nan if pd.isna(v) else ("Yes" if v == "Yes" else "No/Mild"))
    D["Arterial enhancement"] = c[col("Enhancement")]; D["LI-RADS v2018"] = c[col("LI-RADS")].map(lirads); D["Arterial enhancement"] = c[col("Enhancement")].astype(str).str.strip().str.capitalize().where(c[col("Enhancement")].notna()); D["LI-RADS v2018"] = c[col("LI-RADS")].map(lirads); D["HBP signal"] = c[col("HBP signal")].astype(str).str.strip().str.capitalize().where(c[col("HBP signal")].notna())
    D["Treatment cohort"] = c["Treatment"].astype(str).str.strip().str.upper().map({"N": "Untreated (SA1)", "IO": "Immunotherapy (SA2)", "IO+SBRT": "Immunotherapy + SBRT (SA2)"})
    # outcomes
    D["TILs score"] = o["TILs_resection"].reindex(D.index).map(lambda v: np.nan if pd.isna(v) else str(int(v)))
    D["TLS score"] = o["TLS_resection"].reindex(D.index).map(lambda v: np.nan if pd.isna(v) else str(int(v)))
    D["Recurrence"] = r["event"].reindex(D.index).map({1: "Yes", 0: "No", 1.0: "Yes", 0.0: "No"})
    D["Follow-up / time to recurrence, days"] = r["time_days"].reindex(D.index).astype(float)
    return D, r


GROUPS = {"Clinical / demographic": ["Age, years", "Gender", "BMI, kg/m2", "Race", "Ethnicity", "Etiology of liver disease", "Cirrhosis", "Serum AFP, ng/mL", "Treatment cohort"],
          "Qualitative MRI": ["Maximum tumor size on T1 PVP, cm", "Morphology", "Capsule (on PVP)", "Margin", "Mosaic appearance", "Internal arteries", "Macrovascular invasion", "Tumor necrosis",
                              "Fat content", "T2 signal", "Restricted diffusion", "Arterial enhancement", "LI-RADS v2018", "HBP signal"]}
ORDER = {"T2 signal": ["Hypo", "Iso", "Mild hyper (incl. 'Mild Hyper')", "Marked hyper"], "HBP signal": ["Hypo", "Iso", "Hyper", "Mixed"], "Margin": ["Well defined", "Irregular", "Infiltrative"],
         "Race": ["White", "Black or African-American", "Asian", "Other", "Unknown"], "Ethnicity": ["Non-Hispanic", "Hispanic/Latino", "Unknown"],
         "Etiology of liver disease": ["HBV", "HCV", "MASH/MASLD/NASH/NAFLD", "Alcohol", "Multiple/other"], "LI-RADS v2018": ["LR-2", "LR-3", "LR-4", "LR-5", "Other/atypical"]}


def fmt_cont(x):
    x = x.dropna(); return "-" if len(x) == 0 else f"{x.median():.1f} [{x.quantile(.25):.1f}-{x.quantile(.75):.1f}]"


def pvalue(var, a, b, cont):
    if cont:
        a, b = a.dropna(), b.dropna(); return mannwhitneyu(a, b).pvalue if len(a) and len(b) else np.nan, ""
    levels = sorted(set(a.dropna()) | set(b.dropna()))
    if len(levels) < 2: return np.nan, ""
    T = np.array([[ (a == l).sum() for l in levels], [(b == l).sum() for l in levels]])
    if T.shape[1] == 2:
        return fisher_exact(T.T)[1], ""
    chi, p, _, exp = chi2_contingency(T, correction=False); return p, ("†" if (exp < 5).any() else "")


def table(D, split, name):
    sub = D.loc[split.index].copy()
    sets = {"Overall": sub, f"Development": sub[split["set"] == "dev"], "Lock": sub[split["set"] == "lock"]}
    rows = []
    n_row = {"Group": "", "Variable": "N", "Level": ""}; n_row.update({k: str(len(v)) for k, v in sets.items()}); n_row.update({"p": "", "note": ""}); rows.append(n_row)
    groups = dict(GROUPS); groups["Outcomes"] = ["TILs score", "TLS score", "Tumor grade (pathology)", "Recurrence", "Follow-up / time to recurrence, days"]
    for g, vars_ in groups.items():
        for v in vars_:
            cont = pd.api.types.is_numeric_dtype(sub[v]) and v != "Treatment cohort"
            p, dag = pvalue(v, sets["Development"][v], sets["Lock"][v], cont)
            if cont:
                r = {"Group": g, "Variable": v, "Level": "median [IQR]"}; r.update({k: fmt_cont(s[v]) for k, s in sets.items()})
                r["missing"] = "; ".join(f"{int(s[v].isna().sum())}" for s in sets.values()); r.update({"p": p, "note": dag}); rows.append(r)
            else:
                levels = ORDER.get(v) or sorted(sub[v].dropna().unique(), key=str)
                for i, lv in enumerate(levels):
                    r = {"Group": g, "Variable": v if i == 0 else "", "Level": lv}
                    for k, s in sets.items():
                        d = s[v].dropna(); r[k] = f"{(d == lv).sum()} ({100 * (d == lv).mean():.0f}%)" if len(d) else "-"
                    r.update({"p": p if i == 0 else "", "note": dag if i == 0 else ""}); rows.append(r)
                miss = {k: int(s[v].isna().sum()) for k, s in sets.items()}
                if any(miss.values()):
                    r = {"Group": g, "Variable": "", "Level": "missing"}; r.update({k: str(m) for k, m in miss.items()}); r.update({"p": "", "note": ""}); rows.append(r)
    T = pd.DataFrame(rows); T["p"] = T["p"].map(lambda x: "" if x == "" or pd.isna(x) else ("<0.001" if x < 0.001 else f"{x:.3f}")); return T


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(ROOT / "Results/main_pipeline_claude/presentation_20261020/tables")); a = ap.parse_args(); warnings.filterwarnings("ignore")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    D, r = build_frame()
    tils = pd.read_csv(ROOT / "Results/main_pipeline_claude/main_pipeline/final/cohort_v3_141/split/split.csv").set_index("PatientID")
    coh = pd.read_csv(ROOT / "Results/main_pipeline_claude/main_pipeline_recurrence/split/recurrence_split.csv").set_index("PatientID")["cohort"]
    tils["cohort"] = coh.reindex(tils.index)
    # P141 moved from SA2 to SA1: the recurrence split already carries the current cohort; fill any gap from the outcome table
    tils["cohort"] = tils["cohort"].fillna(r["cohort"].reindex(tils.index))
    rec = pd.read_csv(ROOT / "Results/main_pipeline_claude/main_pipeline_recurrence/split/recurrence_split.csv").set_index("PatientID")
    with pd.ExcelWriter(out / "cohort_tables.xlsx") as xw:
        for name, sp in {"TILs_TLS_split": tils, "Recurrence_split": rec}.items():
            T = table(D, sp, name); T.to_csv(out / f"table_{name}.csv", index=False); T.to_excel(xw, sheet_name=name, index=False)
            print(name, T.shape, "| N row:", T.iloc[0][["Overall", "Development", "Lock"]].tolist())
    (out / "README.txt").write_text(__doc__)


if __name__ == "__main__":
    main()
