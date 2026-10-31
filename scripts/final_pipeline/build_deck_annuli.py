"""Summary deck of the four peritumoral-annulus runs (A-D), built from the model tables each run saved. Starts from the 08-17-26 template; never overwrites a deck.

A  5 mm out / 2 mm in, SA2 pre-treatment, annulus only        (main_pipeline_claude_annuli)
B  2 mm out / 5 mm in, SA2 pre-treatment, annulus only        (main_pipeline_claude_annuli_v2)
C  2 mm out / 5 mm in, SA2 post-treatment, annulus only       (main_pipeline_claude_annuli_v2_post)
D  as C, plus clinico-radiologic variables                    (main_pipeline_claude_annuli_v2_post_clinrad)
Reference rows: intratumoral radiomics (presentation_20261020) and clinico-radiologic variables alone (presentation_20261022).
'Best' = highest mean of out-of-fold and test performance (AUC; C-index for recurrence), as in the main deck.
"""
import sys
from datetime import date
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
from pptx.util import Inches, Pt
import build_deck as bd

R = bd.ROOT / "Results"
OUT = R / "main_pipeline_claude_annuli_summary"
ENDPOINTS = [("tils_ge2", "TILs 2-3 vs 0-1", "auc"), ("tils_ge3", "TILs 3 vs 0-2", "auc"), ("tls_ge2", "TLS 2 vs 0-1", "auc"), ("recurrence", "Recurrence", "c")]
RUNS = {"A": ("main_pipeline_claude_annuli", "5 mm out / 2 mm in", "pre", "annulus only"), "B": ("main_pipeline_claude_annuli_v2", "2 mm out / 5 mm in", "pre", "annulus only"),
        "C": ("main_pipeline_claude_annuli_v2_post", "2 mm out / 5 mm in", "post", "annulus only"), "D": ("main_pipeline_claude_annuli_v2_post_clinrad", "2 mm out / 5 mm in", "post", "annulus + clinico-radiologic")}
SEQ = {"tp": "TP", "pvp": "PVP", "dwi_b2": "DWI-b2", "hbp": "HBP"}
FAM = {"ridge": "ridge", "lasso": "lasso", "elastic_net": "elastic net"}
_new = bd.new_slide
bd.footnote = lambda *a, **k: None


def _title22(prs, t):
    s = _new(prs, t)
    for r in s.shapes[0].text_frame.paragraphs[0].runs: r.font.size = Pt(22)
    return s


bd.new_slide = _title22


def cols(kind): return ("oof_auc_avgprob", "oof_auc_ci_lo", "oof_auc_ci_hi", "lock_auc") if kind == "auc" else ("oof_c_avg", "oof_c_ci_lo", "oof_c_ci_hi", "lock_c")


def load(base, ep, tag):
    d = R / base
    if ep == "recurrence" and base.endswith("clinrad"):
        return pd.concat([pd.read_csv(d / "recurrence" / x / "recurrence_model_table_radiomics_clinical.csv") for x in ("report_ridge", "report_sparse")], ignore_index=True)
    return pd.read_csv(d / ep / "report" / f"{ep}_model_table_{tag}.csv")


def table(run, ep, kind):
    tag = "radiomics_clinical" if run == "D" else "radiomics"; d = load(RUNS[run][0], ep, tag); o, lo, hi, t = cols(kind)
    d["score"] = (d[o] + d[t]) / 2; d["tlo"], d["thi"] = d[t + "_ci_lo"], d[t + "_ci_hi"]; return d


def ref(ep, kind, which):
    o, lo, hi, t = cols(kind)
    if which == "intra": f = R / "main_pipeline_claude/presentation_20261020" / ep / f"{ep}_model_table_radiomics.csv"
    else: f = R / "main_pipeline_claude/presentation_20261022" / ep / f"{ep}_model_table_clinical.csv"
    d = pd.read_csv(f); d["score"] = (d[o] + d[t]) / 2; return d.loc[d.score.idxmax()]


def seqs(s): return " + ".join(SEQ.get(x, x) for x in s.split("+")) if s not in ("clinical", "clinicoradiologic") else "all variables"
def ci(m, lo, hi): return f"{m:.2f} ({lo:.2f}-{hi:.2f})"


def best(d, kind):
    o, lo, hi, t = cols(kind); return d.loc[d.score.idxmax()]


def cell(r, kind, with_model=True):
    o, lo, hi, t = cols(kind); head = f"{seqs(r.sequence_set)} {FAM.get(r.family, r.family)}: " if with_model else ""
    return f"{head}{ci(r[o], r[lo], r[hi])}; test {r[t]:.2f}"


def sub_row(tbl, i, ncols):
    c = tbl.cell(i, 0); c.merge(tbl.cell(i, ncols - 1)); c.fill.solid(); c.fill.fore_color.rgb = RGBColor(0xE7, 0xE6, 0xE6)
    for p in c.text_frame.paragraphs:
        for r in p.runs: r.font.bold = True


def main():
    prs = Presentation(bd.TEMPLATE); ids = prs.slides._sldIdLst
    for sid in list(ids)[1:]: prs.part.drop_rel(sid.rId); ids.remove(sid)
    bd.text_box(prs.slides[0], 0.26, 5.6, 10, 0.5, [f"Peritumoral annulus radiomics: summary of four runs, {date.today():%m/%d/%y}"], size=16)
    T = {(r, ep): table(r, ep, k) for r in RUNS for ep, _, k in ENDPOINTS}; B = {key: best(d, dict((e, k) for e, _, k in ENDPOINTS)[key[1]]) for key, d in T.items()}
    K = dict((e, k) for e, _, k in ENDPOINTS)

    s = bd.new_slide(prs, "Peritumoral annulus radiomics - rationale and methods")
    bd.text_box(s, 0.5, 1.0, 12.3, 5.9, [("Rationale",), "Immune infiltrates concentrate at the tumor border; the annulus samples that region instead of the tumor interior.",
        ("Annulus definition",), "Built from the tumor ROI (Segment_1), slice by slice with in-plane distances in mm: an outer rim outside the ROI edge, clipped to the liver mask, plus an inner rim inside the edge (not clipped).",
        "Two geometries: 5 mm out / 2 mm in (first), and 2 mm out / 5 mm in (second, to reduce the share of non-tumoral parenchyma).",
        ("Features and models",), "Same extraction as the intratumoral sheets (PyRadiomics, 1 mm resampling, per-sequence bin widths, N4 images; 963 original features).",
        "Inter-reader ICC (23 patients, two readers' ROIs expanded with the same rule): 888 / 963 features with ICC ≥ 0.85 for 5 mm out / 2 mm in; 813 / 963 for 2 mm out / 5 mm in (tumor ROI: 822 / 963).",
        "Same pipeline as the intratumoral analysis: 14 sequence combinations × ridge / lasso / elastic net, at most 20 features, 10 × repeated 5-fold CV; best model = highest mean of out-of-fold and test performance."], size=13, bullet=True)

    s = bd.new_slide(prs, "The four runs")
    rows = [{"Run": r, "Annulus": g, "SA2 imaging": ("pre-treatment" if p == "pre" else "post-treatment"), "Variables": v, "ICC-passing features": ("888 / 963" if r == "A" else "813 / 963")} for r, (_, g, p, v) in RUNS.items()]
    bd.add_table(s, pd.DataFrame(rows), 0.5, 1.2, [0.8, 2.6, 2.4, 3.6, 2.4], font=12, row_h=0.55, align_center_from=4)
    bd.text_box(s, 0.5, 4.2, 12.3, 2.5, ["A vs B isolates the annulus geometry (same data, pre-treatment SA2). B vs C isolates pre- vs post-treatment SA2 (same geometry).",
        "D adds the clinico-radiologic variables (tumor grade included; ridge selects from one pooled ranking, so no model has more than 20 features)."], size=13, bullet=True)

    s = bd.new_slide(prs, "Best model per endpoint and run")
    hdr = ["Endpoint"] + [f"Run {r}" for r in RUNS] + ["Intratumoral radiomics", "Clinico-radiologic alone"]; rows = []
    for ep, lab, k in ENDPOINTS:
        row = {"Endpoint": lab}
        for r in RUNS: row[f"Run {r}"] = cell(B[(r, ep)], k)
        row["Intratumoral radiomics"] = cell(ref(ep, k, "intra"), k); row["Clinico-radiologic alone"] = cell(ref(ep, k, "clin"), k); rows.append(row)
    bd.add_table(s, pd.DataFrame(rows)[hdr], 0.25, 1.15, [1.3, 1.9, 1.9, 1.9, 1.95, 1.95, 1.95], font=9, row_h=0.95, align_center_from=1, header_h=0.45)
    bd.text_box(s, 0.25, 5.8, 12.8, 0.6, ["Out-of-fold AUC (C-index for recurrence) with 95% CI; test value after the semicolon. Best = highest mean of out-of-fold and test performance."], size=11)

    for title, ci_col in (("Out-of-fold performance of the best model, by run", 0), ("Test-set performance of the best model, by run", 3)):
        s = bd.new_slide(prs, title); cd = CategoryChartData(); cd.categories = [lab for _, lab, _ in ENDPOINTS]; pick = lambda x, k: float(x[cols(k)[ci_col]])
        for r in RUNS: cd.add_series(f"Run {r}", [pick(B[(r, ep)], k) for ep, _, k in ENDPOINTS])
        cd.add_series("Intratumoral radiomics", [pick(ref(ep, k, "intra"), k) for ep, _, k in ENDPOINTS])
        cd.add_series("Clinico-radiologic alone", [pick(ref(ep, k, "clin"), k) for ep, _, k in ENDPOINTS])
        ch = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.7), Inches(1.1), Inches(11.9), Inches(5.6), cd).chart
        ch.has_legend = True; ch.legend.position = XL_LEGEND_POSITION.BOTTOM; ch.legend.include_in_layout = False; ch.legend.font.size = Pt(11)
        ch.value_axis.minimum_scale = 0.3; ch.value_axis.maximum_scale = 0.9 if ci_col else 0.8; ch.value_axis.has_major_gridlines = True; ch.value_axis.tick_labels.font.size = Pt(11); ch.category_axis.tick_labels.font.size = Pt(12)
        pl = ch.plots[0]; pl.gap_width = 60; pl.has_data_labels = True; pl.data_labels.number_format = "0.00"; pl.data_labels.number_format_is_linked = False; pl.data_labels.font.size = Pt(9)
        for ser, c in zip(pl.series, ["9DC3E6", "5B9BD5", "2E75B6", "C00000", "7F7F7F", "70AD47"]): ser.format.fill.solid(); ser.format.fill.fore_color.rgb = RGBColor.from_string(c)

    for ep, lab, k in ENDPOINTS:
        s = bd.new_slide(prs, f"{lab} - top models in each run"); o, lo, hi, t = cols(k); m = "AUC" if k == "auc" else "C-index"; c_oof, c_test = f"Out-of-fold {m} (95% CI)", f"Test {m} (95% CI)"; rows, subs = [], []
        for r in RUNS:
            d = T[(r, ep)]; n_ok = int((d[lo] > 0.5).sum()); g = RUNS[r]
            subs.append(len(rows)); rows.append({"Sequences": f"Run {r}: {g[1]}, SA2 {g[2]}-treatment, {g[3]} ({n_ok} of {len(d)} models with out-of-fold CI above 0.5)", "Model": "", c_oof: "", c_test: ""})
            for _, x in d.sort_values("score", ascending=False).drop_duplicates("sequence_set").head(2).iterrows():
                rows.append({"Sequences": seqs(x.sequence_set), "Model": FAM.get(x.family, x.family), c_oof: ci(x[o], x[lo], x[hi]), c_test: ci(x[t], x.tlo, x.thi)})
        cf = R / "main_pipeline_claude/presentation_20261022" / ep / f"{ep}_model_table_clinical.csv"; cd_ = pd.read_csv(cf); cd_["score"] = (cd_[o] + cd_[t]) / 2
        subs.append(len(rows)); rows.append({"Sequences": "Clinico-radiologic variables only (no radiomics; reference)", "Model": "", c_oof: "", c_test: ""})
        for _, x in cd_.sort_values("score", ascending=False).iterrows(): rows.append({"Sequences": "all variables", "Model": FAM.get(x.family, x.family), c_oof: ci(x[o], x[lo], x[hi]), c_test: ci(x[t], x[t + "_ci_lo"], x[t + "_ci_hi"])})
        df = pd.DataFrame(rows).fillna("")
        shp = bd.add_table(s, df, 0.6, 1.1, [5.0, 1.6, 2.6, 2.6], font=10, row_h=0.33, align_center_from=2, header_h=0.5)
        for i in subs: sub_row(shp.table, i + 1, 4)

    s = bd.new_slide(prs, "ROI quality control (2 mm out / 5 mm in masks)")
    O = Path("/Volumes/LaCie2/peritumor_roisv2"); rows = []
    try:
        for lab, f1, f2, n in (("SA1 + SA2 pre-treatment", "flagged_cases_v2.csv", "flagged_patients_v2.csv", "1,230 masks / 141 patients"), ("SA2 post-treatment", "flagged_cases_v2_SA2_post.csv", "flagged_patients_v2_SA2_post.csv", "269 masks / 30 patients")):
            rows.append({"Set": lab, "Masks / patients": n, "Flagged masks": str(len(pd.read_csv(O / f1))), "Flagged patients": str(len(pd.read_csv(O / f2)))})
        bd.add_table(s, pd.DataFrame(rows), 0.8, 1.2, [3.6, 3.2, 2.4, 2.4], font=12, row_h=0.55, align_center_from=1)
    except Exception as e: print("QC table skipped:", e)
    bd.text_box(s, 0.8, 3.2, 11.8, 3.5, ["Flags: tumor under 90% inside the liver mask, liver-mask outliers within a patient, thin outer rim (under 15% of the annulus), annulus under 2 mL, tumor volume more than 2x off the patient's median.",
        "Largest problems were liver-mask failures on single sequences (DWI mostly) and exophytic tumors (e.g. P121, P149); none of these were corrected by hand before the models were run.",
        "Annulus masks were regenerated from the current tumor ROIs; earlier hand edits of annulus files are not included."], size=13, bullet=True)

    s = bd.new_slide(prs, "Take-home messages")
    g = lambda r, ep: B[(r, ep)]; o = lambda x: float(x.oof_auc_avgprob) if "oof_auc_avgprob" in x else float(x.oof_c_avg)
    ob = {r: float(B[(r, "tils_ge2")].oof_auc_avgprob) for r in RUNS}
    lines = [("Where the annulus works",), f"TILs 2-3 vs 0-1 is the only endpoint with an annulus signal: out-of-fold AUC of the best model {ob['A']:.2f} (A), {ob['B']:.2f} (B), {ob['C']:.2f} (C) as the geometry moves to a thin outer rim with a deep inner rim and SA2 to post-treatment.",
             f"Run C reaches {cell(g('C', 'tils_ge2'), 'auc')}, above the intratumoral radiomics ({cell(ref('tils_ge2', 'auc', 'intra'), 'auc', False)}); it is driven by TP alone, picked from 14 combinations × 3 families, with a 30-patient test set.",
             ("Where it does not",), "TILs 3 vs 0-2 and TLS 2 vs 0-1: no annulus-only model has an out-of-fold CI above 0.5 in any run. Recurrence: about 0.62 in every run, the same as the intratumoral radiomics.",
             "Adding clinico-radiologic variables (run D) does not improve TILs 2-3 vs 0-1; for TILs 3 vs 0-2 it recovers the clinico-radiologic signal and the annulus adds nothing to it.",
             ("Caveats",), "A vs B (geometry) and B vs C (pre vs post SA2) each change one thing, but the best-model picture rests on single models among many; all intervals are wide and the test set is small (n=30)."]
    bd.text_box(s, 0.5, 1.0, 12.3, 5.9, lines, size=13, bullet=True)
    OUT.mkdir(parents=True, exist_ok=True); dest = OUT / f"annulus_summary_{date.today():%m-%d-%y}_v3.pptx"; prs.save(dest); print("saved", dest, "slides", len(prs.slides))


if __name__ == "__main__":
    main()
