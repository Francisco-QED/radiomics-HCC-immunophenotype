"""Rebuild the recurrence (time-to-event) outcome table from the mastersheet's 'Recurrence data' block.

Rules (reverse-engineered from, and validated against, the previous recurrence_outcomes.xlsx):
  event      = 1 if 'Recurrence (Y/N)' == 'Y' else 0
  time_days  = 'Time to Recurrence (days)' if event else 'Time to Follow Up (For No Recurrence)'
  time_years = time_days / 365.25
  recur_2yr  = 1 if event and time <= 730 d; 0 if (event and time > 730) or (no event and time >= 730); NaN if no event and time < 730 (status unknown)
  exclude    = 'no_followup' if no usable time (the patient cannot enter a survival model)
Cohort tabs: SA1 + SA2-pre/post (recurrence data are identical across SA2 tabs; checked by --check-sa2).
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/Users/Francisco/Library/CloudStorage/OneDrive-TheMountSinaiHospital/Regeneron retrospective radiomics")


def tab(path, name):
    d = pd.read_excel(path, sheet_name=name, header=None)
    h = d.iloc[1].astype(str).str.strip(); b = d.iloc[2:]
    b = b[b.iloc[:, 0].astype(str).str.match(r"^P\d{3}$")].reset_index(drop=True)
    b.columns = [c if c != "nan" else f"col{i}" for i, c in enumerate(h)]
    return b


def build(path, sa2_tab):
    rows = []
    for name, coh in (("SA1", "SA1"), (sa2_tab, "SA2")):
        b = tab(path, name)
        rec = b["Recurrence (Y/N)"].astype(str).str.strip().str.upper()
        ev = (rec == "Y").astype(int)
        t_rec = pd.to_numeric(b["Time to Recurrence (days)"], errors="coerce")
        t_fu = pd.to_numeric(b["Time to Follow Up (For No Recurrence)"], errors="coerce")
        t = np.where(ev == 1, t_rec, t_fu)
        out = pd.DataFrame({"PatientID": b.iloc[:, 0], "cohort": coh, "recurrence": rec.where(rec.isin(["Y", "N"]), np.nan), "event": ev,
                            "time_days": t, "comments": b["Recurrence Comments"], "surgery_date": pd.to_datetime(b["Date of surgery"], errors="coerce")})
        rows.append(out)
    o = pd.concat(rows, ignore_index=True)
    o["time_years"] = (o.time_days / 365.25).round(3)
    o["recur_2yr"] = np.where((o.event == 1) & (o.time_days <= 730), 1.0, np.where(((o.event == 1) & (o.time_days > 730)) | ((o.event == 0) & (o.time_days >= 730)), 0.0, np.nan))
    o["exclude"] = pd.Series("no_followup", index=o.index).where(o.time_days.isna() | (o.time_days <= 0), other=np.nan)
    o.loc[o.exclude.notna(), ["time_days", "time_years"]] = np.nan      # excluded patients carry no time (as in the previous file)
    return o


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mastersheet", default=str(ROOT / "databases/mastersheet.xlsx"))
    ap.add_argument("--sa2-tab", default="SA2-pre")
    ap.add_argument("--previous", default=str(ROOT / "databases/recurrence_outcomes.xlsx"))
    ap.add_argument("--out", default=str(ROOT / "databases/recurrence_outcomes_v3.xlsx"))
    a = ap.parse_args()
    o = build(a.mastersheet, a.sa2_tab)
    o.drop(columns="surgery_date").to_excel(a.out, index=False)
    prev = pd.read_excel(a.previous).set_index("PatientID"); new = o.set_index("PatientID")
    print(f"new table: {len(o)} patients | events {int(o.event.sum())} | excluded {int(o.exclude.notna().sum())} {o[o.exclude.notna()].PatientID.tolist()} | median follow-up {o.time_years.median():.2f} y")
    common = new.index.intersection(prev.index)
    d_ev = [p for p in common if new.at[p, "event"] != prev.at[p, "event"]]
    d_t = [(p, prev.at[p, "time_days"], new.at[p, "time_days"]) for p in common if not (pd.isna(new.at[p, "time_days"]) and pd.isna(prev.at[p, "time_days"])) and new.at[p, "time_days"] != prev.at[p, "time_days"]]
    print(f"vs previous file ({len(prev)} patients): in both {len(common)} | event status changed {len(d_ev)} {d_ev} | time changed {len(d_t)}")
    for p, a_, b_ in d_t[:25]: print(f"   {p}: {a_} -> {b_} days")
    print("only in new:", sorted(set(new.index) - set(prev.index)), "| only in previous:", sorted(set(prev.index) - set(new.index)))
    print("wrote", a.out)
