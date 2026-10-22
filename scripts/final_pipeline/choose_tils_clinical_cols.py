"""Choose the clinico-radiologic columns for the TILs models by the pre-stated prevalence rule (outcome-independent):
binary variables with minority < 10% in the TILs development set are dropped; treatment dummies are always kept. Writes clinical/tils_cols.json."""
import json, sys
from pathlib import Path
sys.dont_write_bytecode = True
import pandas as pd
ROOT = Path("/Users/Francisco/Library/CloudStorage/OneDrive-TheMountSinaiHospital/Regeneron retrospective radiomics")
d = ROOT / "Results/main_pipeline_claude/main_pipeline_recurrence/clinical"
M = pd.read_csv(d / "clinical_matrix.csv", index_col=0); tiers = json.load(open(d / "tiers.json"))
dev = pd.read_csv(ROOT / "Results/main_pipeline_claude/main_pipeline/final/cohort_v3_141/split/split.csv").query("set == 'dev'").PatientID
cols, dropped = [], []
for c in tiers["pre"]:
    v = M.loc[dev, c].dropna()
    if set(v.unique()) <= {0.0, 1.0} and not c.startswith("treat_"):
        m = min(v.mean(), 1 - v.mean())
        if m < 0.10: dropped.append((c, round(float(m), 3))); continue
    cols.append(c)
json.dump(dict(tils_pre_grade=cols + ["grade"], tils_pre=cols, dropped=dropped, rule="binary variables with minority <10% in the TILs dev set are dropped; treatment dummies always kept"), open(d / "tils_cols.json", "w"), indent=1)
print("kept", len(cols), cols); print("dropped", dropped)
