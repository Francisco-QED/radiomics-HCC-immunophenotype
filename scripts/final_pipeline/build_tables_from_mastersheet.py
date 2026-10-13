"""Build the outcomes + clinical tables the pipeline reads, straight from mastersheet.xlsx.

The mastersheet is the single source of truth. Tabs: SA1 (untreated), SA2-pre, SA2-post. Layout per tab:
row 0 = section labels, row 1 = field names, rows 2+ = one patient per row (PatientID in column 0).
Only rows whose PatientID matches P### are kept; anything else (stray notes, blank rows) is ignored.

Outputs (written next to the mastersheet unless --out-dir is given):
  immunophenotype_outcomes_v2.xlsx   PatientID, TILs_resection, TLS_resection, Grade   (same schema as the old file)
  SA1-2_clinical_imaging_v2.xlsx     mastersheet columns up to 'Qualitative Imaging Comments', SA1 naming
  mastersheet_build_manifest.csv     per-patient TILs change vs the previous outcomes file + build parameters
"""
import argparse
import difflib
import re
from pathlib import Path

import pandas as pd

ID_RE = r"^P\d{3}$"
LAST_CLINICAL_COL = "qualitative imaging comments"
DEFAULT_ROOT = Path.home() / ("Library/CloudStorage/OneDrive-TheMountSinaiHospital/"
                              "Regeneron retrospective radiomics")


def read_tab(path, tab):
    raw = pd.read_excel(path, sheet_name=tab, header=None)
    names = raw.iloc[1].astype(str).str.strip()
    body = raw.iloc[2:].copy()
    body.columns = names.where(names != "nan", [f"col{i}" for i in range(len(names))])
    body = body[body.iloc[:, 0].astype(str).str.match(ID_RE)].reset_index(drop=True)
    if body.iloc[:, 0].duplicated().any():
        raise ValueError(f"{tab}: duplicate PatientIDs {body.iloc[:, 0][body.iloc[:, 0].duplicated()].tolist()}")
    return body


def harmonize(tab_df, canon_cols, tab):
    """Map a tab's clinical block onto SA1's column names by position, verifying each name pair is similar."""
    cols = list(tab_df.columns[:len(canon_cols)])
    for c_new, c_ref in zip(cols, canon_cols):
        sim = difflib.SequenceMatcher(None, c_new.lower(), c_ref.lower()).ratio()
        if sim < 0.45:
            raise ValueError(f"{tab}: column '{c_new}' does not look like SA1's '{c_ref}' (sim={sim:.2f})")
    out = tab_df.iloc[:, :len(canon_cols)].copy()
    out.columns = canon_cols
    other = tab_df.iloc[:, len(canon_cols):]
    return pd.concat([out, other], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mastersheet", default=str(DEFAULT_ROOT / "databases/mastersheet_2026-10-13.xlsx"))
    ap.add_argument("--sa2-tab", choices=["SA2-pre", "SA2-post"], default="SA2-pre")
    ap.add_argument("--previous-outcomes", default=str(DEFAULT_ROOT / "databases/immunophenotype_outcomes.xlsx"))
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    out_dir = Path(a.out_dir) if a.out_dir else Path(a.mastersheet).parent

    sa1 = read_tab(a.mastersheet, "SA1")
    sa2 = read_tab(a.mastersheet, a.sa2_tab)
    stop = [i for i, c in enumerate(sa1.columns) if c.lower() == LAST_CLINICAL_COL]
    if not stop:
        raise ValueError(f"SA1 has no '{LAST_CLINICAL_COL}' column to end the clinical block on")
    canon = list(sa1.columns[:stop[0] + 1])
    sa2 = harmonize(sa2, canon, a.sa2_tab)
    sa1 = sa1.rename(columns={sa1.columns[0]: "PatientID"})
    sa2 = sa2.rename(columns={sa2.columns[0]: "PatientID"})
    canon[0] = "PatientID"

    def pick(df, name):
        hit = [c for c in df.columns if c.lower() == name.lower()]
        if not hit:
            raise ValueError(f"missing column '{name}' (have: {list(df.columns)[-12:]})")
        return df[hit[0]]

    def outcome_block(df):
        return pd.DataFrame({
            "PatientID": df["PatientID"],
            "TILs_resection": pd.to_numeric(pick(df, "TILs_resection"), errors="coerce"),
            "TLS_resection": pd.to_numeric(pick(df, "TLS_resection"), errors="coerce"),
            "Grade": pick(df, "Tumor_differentiation"),
        })

    outcomes = pd.concat([outcome_block(sa1), outcome_block(sa2)], ignore_index=True)
    clinical = pd.concat([sa1[canon], sa2[canon]], ignore_index=True)
    if outcomes.PatientID.duplicated().any():
        raise ValueError(f"PatientID in both SA1 and SA2: {outcomes.PatientID[outcomes.PatientID.duplicated()].tolist()}")

    prev = pd.read_excel(a.previous_outcomes).set_index("PatientID")["TILs_resection"]
    man = outcomes.set_index("PatientID")[["TILs_resection"]].rename(columns={"TILs_resection": "TILs_new"})
    man["TILs_previous"] = prev.reindex(man.index)
    man["status"] = "unchanged"
    man.loc[man.TILs_previous.isna() & man.TILs_new.notna(), "status"] = "newly_scored_or_added"
    man.loc[man.TILs_previous.notna() & man.TILs_new.notna() & (man.TILs_previous != man.TILs_new), "status"] = "CHANGED"
    man.loc[man.TILs_new.isna(), "status"] = "no_TILs"
    man["sa2_tab"] = a.sa2_tab
    man.loc[sa1.PatientID, "sa2_tab"] = "SA1"
    removed = sorted(set(prev.index) - set(man.index))

    out_dir.mkdir(parents=True, exist_ok=True)
    outcomes.to_excel(out_dir / "immunophenotype_outcomes_v2.xlsx", index=False)
    clinical.to_excel(out_dir / "SA1-2_clinical_imaging_v2.xlsx", index=False)
    man.reset_index().to_csv(out_dir / "mastersheet_build_manifest.csv", index=False)

    t = outcomes.TILs_resection
    print(f"SA1={len(sa1)}  {a.sa2_tab}={len(sa2)}  total={len(outcomes)}  with TILs={int(t.notna().sum())}")
    print("TILs distribution:", t.value_counts().sort_index().to_dict())
    print("status counts:", man.status.value_counts().to_dict())
    print("changed:", man[man.status == 'CHANGED'].index.tolist())
    print("new:", man[man.status == 'newly_scored_or_added'].index.tolist())
    print("in previous outcomes but not in mastersheet:", removed)
    print("wrote:", [p.name for p in (out_dir / "immunophenotype_outcomes_v2.xlsx", out_dir / "SA1-2_clinical_imaging_v2.xlsx", out_dir / "mastersheet_build_manifest.csv")])


if __name__ == "__main__":
    main()
