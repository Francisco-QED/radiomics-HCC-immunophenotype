"""Build the results deck (python-pptx) from the tables/figures produced by build_presentation_tables.py, build_tils_deliverables.py and make_shap_plots.py.

Starts from the 08-17-26 deck (theme, fonts, slide size) with all its slides removed; every slide uses the 'Blank' layout with an Arial title box like the original.
Sections are added only when their input files exist, so the cohort/method slides can be built before the results are final.
Writes <out>/updates-<date>_draft.pptx (never overwrites the source deck).
"""
import argparse, copy, sys
from datetime import date
from pathlib import Path
sys.dont_write_bytecode = True
import pandas as pd
from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

ROOT = Path("/Users/Francisco/Library/CloudStorage/OneDrive-TheMountSinaiHospital/Regeneron retrospective radiomics")
TEMPLATE = "/Users/Francisco/Desktop/RestrepoProjects/Active/Regeneron retrospective radiomics/presentations/updates-08-17-26.pptx"
HDR = RGBColor(0xDD, 0xE6, 0xEE); GREY = RGBColor(0x59, 0x59, 0x59); LINE = "BFBFBF"
NS = "http://schemas.openxmlformats.org/drawingml/2006/main"


def clear_slides(prs):
    ids = prs.slides._sldIdLst
    for sid in list(ids):
        prs.part.drop_rel(sid.rId); ids.remove(sid)


def new_slide(prs, title):
    s = prs.slides.add_slide(next(l for l in prs.slide_layouts if l.name == "Blank"))
    for ph in list(s.placeholders):                                   # drop footer / slide-number placeholders copied from the layout; re-add the number below
        if ph.placeholder_format.type is not None and "SLIDE_NUMBER" not in str(ph.placeholder_format.type) and "FOOTER" in str(ph.placeholder_format.type):
            ph._element.getparent().remove(ph._element)
    tb = s.shapes.add_textbox(Inches(0.5), Inches(0.35), Inches(12.33), Inches(0.6)); tf = tb.text_frame; tf.word_wrap = True
    r = tf.paragraphs[0].add_run(); r.text = title; r.font.name = "Arial"; r.font.size = Pt(24); r.font.bold = True
    return s


def text_box(slide, left, top, width, height, paras, size=14, bold_first=False, color=None, bullet=False):
    tb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height)); tf = tb.text_frame; tf.word_wrap = True
    for i, t in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        head = isinstance(t, tuple); txt = t[0] if head else t
        r = p.add_run(); r.text = ("•  " if (bullet and not head) else "") + txt; r.font.size = Pt(size + (1 if head else 0)); r.font.bold = head; r.font.name = "Arial"
        if color: r.font.color.rgb = color
        p.space_after = Pt(5)
    return tb


def _border(cell, bottom=True):
    tcPr = cell._tc.get_or_add_tcPr()
    for tag in ("lnL", "lnR", "lnT", "lnB"):
        for e in tcPr.findall(f"{{{NS}}}{tag}"): tcPr.remove(e)
    for tag in ("lnL", "lnR", "lnT", "lnB"):
        ln = etree.SubElement(tcPr, f"{{{NS}}}{tag}", w="6350" if (tag == "lnB" and bottom) else "0")
        if tag == "lnB" and bottom:
            sf = etree.SubElement(ln, f"{{{NS}}}solidFill"); etree.SubElement(sf, f"{{{NS}}}srgbClr", val=LINE)
        else:
            etree.SubElement(ln, f"{{{NS}}}noFill")
    # fill element must precede borders? schema: lnL lnR lnT lnB ... then fill. move any fill to the end
    for e in list(tcPr):
        if e.tag.endswith(("solidFill", "noFill")) and e.getparent() is tcPr and e.tag.split("}")[1] in ("solidFill", "noFill"):
            tcPr.remove(e); tcPr.append(e)


def add_table(slide, df, left, top, widths, font=8, row_h=0.2, bold_rows=(), align_center_from=1, header_h=None):
    rows, cols = len(df) + 1, len(df.columns)
    shp = slide.shapes.add_table(rows, cols, Inches(left), Inches(top), Inches(sum(widths)), Inches(row_h * rows)); tbl = shp.table
    tblPr = tbl._tbl.tblPr
    for k in ("bandRow", "firstRow"): tblPr.attrib.pop(k, None)
    sid = tblPr.find(f"{{{NS}}}tableStyleId")
    if sid is None: sid = etree.SubElement(tblPr, f"{{{NS}}}tableStyleId")
    sid.text = "{2D5ABB26-0587-4C30-8999-92F81FD0307C}"                  # No Style, No Grid
    for j, w in enumerate(widths): tbl.columns[j].width = Inches(w)
    for i in range(rows): tbl.rows[i].height = Inches(header_h if (i == 0 and header_h) else row_h)
    def put(cell, text, bold=False, fill=None, center=False, size=font):
        cell.margin_left = cell.margin_right = Inches(0.04); cell.margin_top = cell.margin_bottom = Inches(0.01); cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        tf = cell.text_frame; tf.word_wrap = True; p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER if center else PP_ALIGN.LEFT
        r = p.add_run(); r.text = "" if (text is None or (isinstance(text, float) and pd.isna(text))) else str(text); r.font.size = Pt(size); r.font.bold = bold; r.font.name = "Arial"
        if fill is not None: cell.fill.solid(); cell.fill.fore_color.rgb = fill
        _border(cell)
    for j, c in enumerate(df.columns): put(tbl.cell(0, j), c, bold=True, fill=HDR, center=j >= align_center_from)
    for i, (_, r) in enumerate(df.iterrows(), 1):
        for j, v in enumerate(r): put(tbl.cell(i, j), v, bold=(i - 1) in bold_rows, center=j >= align_center_from)
    return shp


def picture(slide, path, left, top, width=None, height=None):
    kw = {}
    if width: kw["width"] = Inches(width)
    if height: kw["height"] = Inches(height)
    return slide.shapes.add_picture(str(path), Inches(left), Inches(top), **kw)


def footnote(slide, text, top=7.05):
    text_box(slide, 0.5, top, 12.3, 0.4, [text], size=9, color=GREY)


def cohort_table_slides(prs, csv, split_label, n_note):
    T = pd.read_csv(csv).fillna(""); n = T.iloc[0]; hdr = {"Overall": f"Overall (N={n['Overall']})", "Development": f"Development (N={n['Development']})", "Lock": f"Lock (N={n['Lock']})"}
    T = T.iloc[1:].copy(); T["Variable"] = T["Variable"].where(T["Variable"] != "", "")
    pages, cur, size = [], [], 0
    blocks = []; blk = []
    for _, r in T.iterrows():
        if r["Variable"] != "" and blk: blocks.append(blk); blk = []
        blk.append(r)
    if blk: blocks.append(blk)
    by_group = {}
    for b in blocks: by_group.setdefault(b[0]["Group"], []).append(b)
    for g, bl in by_group.items():
        cur, size, part = [], 0, 1
        for b in bl:
            if size + len(b) > 27 and cur:
                pages.append((g, part, cur)); cur, size, part = [], 0, part + 1
            cur += b; size += len(b)
        if cur: pages.append((g, part, cur))
    for g, part, rows in pages:
        multi = sum(1 for p in pages if p[0] == g) > 1
        s = new_slide(prs, f"{g}{f' ({part})' if multi else ''}: development vs lock, {split_label}")
        df = pd.DataFrame(rows)[["Variable", "Level", "Overall", "Development", "Lock", "p", "note"]]
        df["p"] = df["p"].astype(str) + df["note"].astype(str); df = df.drop(columns="note")
        df.columns = ["Variable", "", hdr["Overall"], hdr["Development"], hdr["Lock"], "p"]
        add_table(s, df, 0.5, 1.1, [3.4, 2.4, 1.8, 1.8, 1.7, 0.9], font=9, row_h=0.205, align_center_from=2)
        footnote(s, "Continuous: median [IQR], Mann-Whitney U. Categorical: n (% of non-missing); Fisher's exact (2x2) or chi-square († = expected count < 5). Missing values shown per variable. " + n_note)
    return pages


def methods_slides(prs):
    s = new_slide(prs, "Methods: study design and validation")
    text_box(s, 0.5, 1.15, 12.3, 5.6, [("Data and split",), "141 HCC patients (SA1 untreated n=111, SA2 post-immunotherapy n=30); 9 MRI sequences segmented in 3D; PyRadiomics, per-sequence bin widths, N4 bias-field correction; post-treatment imaging for SA2 (SA1 untreated) is used for both radiomics and radiologic variables, so the two describe the same time point.",
        "Lock set (n=30) held out from all training, tuning and model selection; evaluated once. Development set: TILs/TLS n=111; recurrence n=110 (one patient without follow-up excluded).",
        ("Validation",), "10 x repeated stratified 5-fold cross-validation on the development set (strata: outcome x cohort), 3-fold inner CV for tuning; all feature selection, imputation and scaling fit inside each training fold.",
        "Performance = pooled out-of-fold (OOF) predictions: AUC mean +/- SD over the 10 repeats, and AUC of the repeat-averaged OOF probability with a patient-level bootstrap 95% CI (2,000 resamples).",
        ("Endpoint (primary)",), "TILs >= 2 vs 0-1 at resection (positive = TILs >= 2; 90/111 development, 23/30 lock)."], size=14, bullet=True)
    s = new_slide(prs, "Methods: radiomics model development (inside every training fold)")
    text_box(s, 0.5, 1.15, 6.2, 5.6, [("Feature pipeline",), "963 original features (107 per sequence x 9); LoG and wavelet excluded.", "Inter-reader ICC(3,1) >= 0.85 (23 double-segmented patients; estimated once on development data, outcome-independent).",
        "QC: >= 80% observed, CV > 0.10; median imputation.", "Within-sequence complete-linkage clusters, |Spearman rho| >= 0.85; one outcome-ranked representative per cluster (univariate AUC).", "Standardization."], size=13, bullet=True)
    text_box(s, 6.9, 1.15, 5.9, 5.6, [("Models",), "Ridge logistic regression on the top-k clusters (k = 10, 15, 20).", "Lasso and elastic net (l1 ratio 0.25 / 0.5 / 0.75) constrained to 10-20 non-zero coefficients; regularization tuned by inner 3-fold AUC.",
        ("Sequence combinations",), "All 14 combinations of 1-3 sequences from TP, PVP, DWI (b2) and HBP. One model per combination: the family with the highest dev OOF AUC (selected on development data only)."], size=13, bullet=True)
    s = new_slide(prs, "Methods: clinico-radiologic variables, combined models and metrics")
    text_box(s, 0.5, 1.15, 6.2, 5.6, [("Clinico-radiologic variables (pre-operative only)",), "Age, sex, BMI, HBV / HCV etiology, cirrhosis, log AFP, max tumor size (PVP), capsule, mosaic, internal arteries, fat, T2 hyperintensity, HBP hypointensity, LI-RADS 5, treatment (untreated / immunotherapy / immunotherapy + SBRT, categorical); SA2 radiologic variables are read on the post-treatment MRI.",
        "Binary variables with minority < 10% in the development set dropped (macrovascular invasion for TILs); necrosis not used; no pathology / resection variables.", ("Combined models",), "Radiomics pipeline unchanged; clinical variables bypass ICC and clustering and are added to the selected radiomics features (ridge) or to the candidate pool (lasso / elastic net)."], size=13, bullet=True)
    text_box(s, 6.9, 1.15, 5.9, 5.6, [("Reported measures (all with 95% CI)",), "OOF AUC (bootstrap CI); lock AUC (stratified bootstrap).", "Sensitivity / specificity (Wilson CI) at the Youden threshold of the dev OOF predictions, applied unchanged to the lock set.",
        "Models are reported without selection on lock results. SHAP: exact linear SHAP values of the final development-fit model (coefficient x centered standardized feature)."], size=13, bullet=True)


def tils_results_slides(prs, d):
    foot = "OOF = out-of-fold, development set (n=111); lock set n=30 (23 TILs>=2, 7 TILs<2). CIs: bootstrap (AUC) and Wilson (sens/spec). Sens/spec at the Youden threshold of dev OOF predictions. † = degenerate at that threshold (sens or spec <= 0.05 in OOF or lock)."
    specs = [("table_radiomics_top10.csv", "TILs >= 2: radiomics models, top 10 of 14 sequence combinations", [0.45, 1.7, 1.0, 1.1, 1.3, 1.4, 1.4, 1.3, 1.4, 1.4], 9, None),
             ("table_radiomics_plus_clinical_14.csv", "TILs >= 2: radiomics + clinico-radiologic variables, all 14 combinations", [0.4, 1.55, 0.9, 0.75, 0.65, 0.9, 1.2, 1.25, 1.25, 1.2, 1.25, 1.25], 8, None),
             ("table_clinical_only.csv", "TILs >= 2: clinico-radiologic variables only", [0.45, 1.9, 1.0, 1.05, 1.3, 1.4, 1.4, 1.25, 1.4, 1.4], 9, None)]
    for csv, title, widths, font, _ in specs:
        if not (d / csv).exists(): continue
        s = new_slide(prs, title); df = pd.read_csv(d / csv)
        add_table(s, df, 0.4, 1.15, widths, font=font, row_h=0.36 if len(df) <= 10 else 0.32, align_center_from=3 if csv != "table_radiomics_plus_clinical_14.csv" else 3, header_h=0.6)
        footnote(s, foot, top=6.85)
    if (d / "roc_best_per_category.png").exists():
        s = new_slide(prs, "TILs >= 2: ROC curves, best model of each category"); pic = picture(s, d / "roc_best_per_category.png", 0.9, 1.0, height=5.85); pic.left = int((prs.slide_width - pic.width) / 2)
        footnote(s, "Best model per category chosen by dev OOF AUC (mean over repeats). Left: repeat-averaged OOF probabilities (n=111). Right: one-shot lock set (n=30).", top=6.95)
    best = pd.read_csv(d / "best_per_category.csv") if (d / "best_per_category.csv").exists() else None
    if best is not None:
        s = new_slide(prs, "TILs >= 2: SHAP explainability, best model of each category"); x = 0.35
        for _, r in best.iterrows():
            name = f"{r.sequences.replace('+', '_')}__{r.family}" if r.tag != "clinical" else f"clinicoradiologic__{r.family}"
            img = d / "models" / r.tag / name / "shap_beeswarm.png"
            if img.exists(): picture(s, img, x, 1.15, width=4.2)
            x += 4.3
        footnote(s, "Linear SHAP values on the final development-fit models (standardized selected features, dev patients). SHAP beeswarm and bar plots for all 42 + 42 + 3 models are saved under tils_ge2/models/.", top=6.95)



def recurrence_results_slides(prs, d):
    foot = "Time to recurrence; development n=110 (56 events), lock n=30 (16 events). C = Harrell C-index; AUC(t) = time-dependent AUC (censoring-weighted); sens/spec for 2-year recurrence at the Youden threshold of dev OOF risk (patients with unknown 2-year status excluded). CIs: patient bootstrap (C, AUC), Wilson (sens/spec). † = degenerate threshold (sens or spec <= 0.05)."
    lockfoot = foot + " Lock set was opened earlier in this project: lock numbers are exploratory."
    cats = [("radiomics_top10", "radiomics models, top 10 of 14 sequence combinations", 10), ("radiomics_plus_clinical_14", "radiomics + clinico-radiologic variables, all 14 combinations", 14), ("clinical_only", "clinico-radiologic variables only", 3)]
    for k, label, n in cats:
        for kind, title_kind, widths in (("OOF", "development out-of-fold", [0.45, 1.7, 1.0, 1.1, 1.3, 1.4, 1.4, 1.4, 1.3, 1.3]), ("lock", "lock set (exploratory)", [0.45, 1.9, 1.1, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5])):
            f = d / f"table_{k}_{kind}.csv"
            if not f.exists(): continue
            s = new_slide(prs, f"Recurrence, {title_kind}: {label}"); df = pd.read_csv(f)
            add_table(s, df, 0.4, 1.15, widths, font=9 if n <= 10 else 8.5, row_h=0.36 if n <= 10 else 0.32, align_center_from=3, header_h=0.6)
            footnote(s, lockfoot if kind == "lock" else foot, top=6.8)
    if (d / "roc2y_best_per_category.png").exists():
        s = new_slide(prs, "Recurrence: ROC for 2-year recurrence, best model of each category"); pic = picture(s, d / "roc2y_best_per_category.png", 0.9, 1.0, height=5.85); pic.left = int((prs.slide_width - pic.width) / 2)
        footnote(s, "Binary 2-year recurrence status among patients with known status. Best model per category chosen by dev OOF C-index. Lock set was opened earlier in this project (exploratory).", top=6.95)
    f = d / "best_per_category.csv"
    if f.exists():
        best = pd.read_csv(f); s = new_slide(prs, "Recurrence: SHAP explainability, best model of each category"); x = 0.35
        for _, r in best.iterrows():
            name = f"{r.sequences.replace('+', '_')}__{r.family}" if r.tag != "clinical" else f"clinicoradiologic__{r.family}"
            img = d / "models" / r.tag / name / "shap_beeswarm.png"
            if img.exists(): picture(s, img, x, 1.15, width=4.2)
            x += 4.3
        footnote(s, "Linear SHAP values of the Cox linear predictor in the final development-fit models. Plots for all models are saved under recurrence/models/.", top=6.95)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "Results/main_pipeline_claude/presentation_20261020"))
    a = ap.parse_args(); out = Path(a.out)
    prs = Presentation(TEMPLATE); clear_slides(prs)
    s = new_slide(prs, "Cohort summary and data splits")
    text_box(s, 0.5, 1.2, 12.3, 5, [("Cohorts",), "SA1: 111 resected, untreated HCC.  SA2: 30 resected HCC after neoadjuvant immunotherapy.  141 patients, 9 MRI sequences segmented in 3D.",
                                    ("TILs / TLS splits (development / lock = 111 / 30)",), "Stratified on outcome x cohort; reader-ICC patients forced into development. Lock set never used for training or model selection.",
                                    ("Recurrence split (110 / 30)",), "Stratified on event x cohort. One patient (P121) has no follow-up and is excluded."], size=15, bullet=True)
    methods_slides(prs)
    t = out / "tables"
    for csv, label, note in (("table_TILs_TLS_split.csv", "TILs/TLS split", ""), ("table_Recurrence_split.csv", "recurrence split", "P121 excluded (no follow-up).")):
        if (t / csv).exists(): cohort_table_slides(prs, t / csv, label, note)
    tils_results_slides(prs, out / "tils_ge2")
    recurrence_results_slides(prs, out / "recurrence")
    dest = out / f"updates-{date.today():%m-%d-%y}_draft.pptx"; prs.save(dest); print("saved", dest, "slides:", len(prs.slides))


if __name__ == "__main__":
    main()
