"""Results deck v5 = v4 with the corrected ridge combination models (clinico-radiologic variables compete with radiomics clusters; <=20 features); see merge_ep().

v4 text: Results deck v4 = v3 with: (i) SHAP shown as bar charts, (ii) radiomics-only and radiomics + clinico-radiologic top-5 models merged in one table per endpoint,
(iii) no mean-of-OOF-and-lock column in any table (the score is still used for selection), (iv) tumor grade in the dev/lock split tables and the whole-cohort table.
Selection rule unchanged: mean of OOF and lock AUC (C-index for recurrence); lock numbers of selected models are not independent validation.
"""
import argparse, json, shutil, subprocess, sys
from datetime import date
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION
from pptx.util import Inches, Pt
import build_deck as bd
import build_presentation_tables as bt
import build_deck_v2 as v2
import build_deck_v3 as v3
from build_deck_v3 import ep_cfg, per_combo, load_tables, roc_fig, shap_top, pretty, model_folder, CAT, FAM, ci, NEW, OLD, BASE, SEL_FOOT

bd.footnote = lambda *a, **k: None      # v5: no grey footnotes
_new_slide = bd.new_slide


def _new_slide22(prs, title):
    s = _new_slide(prs, title)
    for sh in s.shapes:
        for p in sh.text_frame.paragraphs:
            for r in p.runs: r.font.size = Pt(22)
        break
    return s


bd.new_slide = _new_slide22
GRADE = {1: "G1", 2: "G2", 2.5: "G3", 3: "G3", 4: "G4"}   # the single G2/G3 patient is counted as G3
CAT[:] = [c for k in ("radiomics", "clinical", "radiomics_clinical") for c in CAT if c[0] == k]   # display order: radiomics, clinico-radiologic, combination
GRADE_ROW = ("Tumor grade", "Tumor grade", "cat")
TOP_N = 5
NF = "Features: clinico-radiologic / total"
REC_COMB_OOF = str(BASE / "main_pipeline_recurrence/grade_ladder14_clinrad_post_20261022/recB_post_forced_*/oof_risk.csv")
COMB_NOTE = (" Features = clinico-radiologic variables / all features in the final model (at most 20). Radiomics + clinico-radiologic models: ridge keeps the top k (10, 15 or 20) of one pooled ranking of radiomics clusters and clinico-radiologic variables; "
             "lasso and elastic net select from one penalized pool (10-20 non-zero coefficients).")


def cls_rows(df, n=None, nfeat=False):
    rows = []
    for _, r in (df.head(n) if n else df).iterrows():
        rows.append({"Sequences": r.sequence_set.replace("+", " + ") if r.sequence_set != "clinicoradiologic" else "all variables", "Model": FAM[r.family],
                     "Out-of-fold AUC (95% CI)": ci(r.oof_auc_avgprob, r.oof_auc_ci_lo, r.oof_auc_ci_hi), "Test AUC (95% CI)": ci(r.lock_auc, r.lock_auc_ci_lo, r.lock_auc_ci_hi)})
        if nfeat: rows[-1][NF] = f"{int(r.get('n_clinical_in_model', 0))} / {int(r.n_features)}"
    return pd.DataFrame(rows)


def rec_rows(df, n=None, nfeat=False):
    rows = []
    for _, r in (df.head(n) if n else df).iterrows():
        rows.append({"Sequences": r.sequence_set.replace("+", " + ") if r.sequence_set != "clinical" else "all variables", "Model": FAM[r.family],
                     "Out-of-fold C-index (95% CI)": ci(r.oof_c_avg, r.oof_c_ci_lo, r.oof_c_ci_hi), "Test C-index (95% CI)": ci(r.lock_c, r.lock_c_ci_lo, r.lock_c_ci_hi)})
        if nfeat: rows[-1][NF] = f"{int(r.get('n_clinical_in_model', 0))} / {int(r.n_features)}"
    return pd.DataFrame(rows)


def merged_slide(prs, title, rb, cb, kind, extra=""):
    rows = cls_rows if kind == "cls" else rec_rows
    a, b = rows(rb.head(TOP_N), nfeat=True), rows(cb.head(TOP_N), nfeat=True); blank = {c: "" for c in a.columns}
    df = pd.concat([pd.DataFrame([{**blank, "Sequences": "Radiomics only"}]), a, pd.DataFrame([{**blank, "Sequences": "Radiomics + clinico-radiologic variables"}]), b], ignore_index=True).rename(columns={"Sequences": "Radiomics sequences"}).rename(columns=lambda c: c)
    df.iloc[[0, TOP_N + 1], 0] = ["Radiomics only", "Radiomics + clinico-radiologic variables"]
    s = bd.new_slide(prs, title)
    widths = [2.4, 1.5, 2.6, 2.6, 2.1] if kind == "cls" else [2.3, 1.4, 2.9, 2.9, 1.7]
    shp = bd.add_table(s, df, 0.45 if kind == "cls" else 0.9, 1.15, widths, font=10.5, row_h=0.42, align_center_from=2, header_h=0.55)
    for i in (1, TOP_N + 2):
        shp.table.cell(i, 0).merge(shp.table.cell(i, len(df.columns) - 1)); c = shp.table.cell(i, 0)
        c.fill.solid(); c.fill.fore_color.rgb = RGBColor(0xE7, 0xE6, 0xE6)
        for p in c.text_frame.paragraphs:
            for r in p.runs: r.font.bold = True
    m = "AUC" if kind == "cls" else "C-index"
    bd.footnote(s, f"One model (best family) per sequence combination; top {TOP_N} combinations by the mean of out-of-fold and test {m}. Out-of-fold = {m} of the split-averaged out-of-fold predictions (development set, 10 repeated 5-fold splits); test = final development-fit model applied once to the test set; bootstrap 95% CIs." +
                (" Test sensitivity / specificity at the Youden threshold of the out-of-fold predictions; † = (almost) everyone classified one way." if kind == "cls" else "") + (" " + extra if extra else "") + COMB_NOTE, top=6.1)


def shap_slide(prs, title, best, dirs, lists):
    s = bd.new_slide(prs, title); x = 0.3
    for tag, name, col in CAT:
        r = best[tag]; sh = shap_top(dirs, tag, r)
        label = f"{name}: {r.sequence_set.replace('+', ' + ') if tag != 'clinical' else 'all variables'} / {FAM[r.family].lower()}" + (f" (n={len(sh)} features)" if sh is not None else "")
        bd.text_box(s, x, 0.95, 4.2, 0.5, [(label,)], size=10)
        if sh is not None:
            lists.append((title.split(" - ")[0], name, label, sh)); top = sh.head(10)
            cd = CategoryChartData(); cd.categories = [(pretty(f)[:38]) for f in top.feature][::-1]; cd.add_series("Mean |SHAP|", [float(v) for v in top.mean_abs_shap][::-1])
            gf = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(x), Inches(1.45), Inches(4.25), Inches(5.3), cd); ch = gf.chart
            ch.has_legend = False; ch.has_title = False; pl = ch.plots[0]; pl.gap_width = 40; pl.has_data_labels = True
            dl = pl.data_labels; dl.number_format = "0.00"; dl.number_format_is_linked = False; dl.font.size = Pt(8); dl.position = XL_LABEL_POSITION.OUTSIDE_END
            ser = pl.series[0]; ser.format.fill.solid(); h = col.lstrip("#"); ser.format.fill.fore_color.rgb = RGBColor(int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
            ch.category_axis.tick_labels.font.size = Pt(8); ch.value_axis.tick_labels.font.size = Pt(8); ch.value_axis.has_major_gridlines = False
        x += 4.35
    bd.footnote(s, "Mean absolute linear SHAP value per feature in the final development-fit models (standardized features; top 10 shown; scales are model-specific). Complete feature lists follow in the appendix.", top=6.9)


v3.shap_slide = shap_slide


def split_spec_add_grade(spec):
    i = next(k for k, t in enumerate(spec) if t[0].startswith("Maximum tumor size")); return spec[:i + 1] + [GRADE_ROW] + spec[i + 1:]


def classification_section(prs, key, D, split, spec_extra, fallback, lists, summary):
    cfg = ep_cfg(key, fallback); tabs = load_tables(cfg, key, "cls", fallback)
    if tabs is None: print("skip", key); return None
    dev_n, lock_n = int((split["set"] == "dev").sum()), int((split["set"] == "lock").sum())
    rb, cb = per_combo(tabs["radiomics"]), per_combo(tabs["radiomics_clinical"]); clb = tabs["clinical"].sort_values("score", ascending=False).reset_index(drop=True)
    merged_slide(prs, f"{cfg['label']} - radiomics vs radiomics + clinico-radiologic variables (top {TOP_N} each)", rb, cb, "cls")
    best = {"radiomics": rb.iloc[0], "radiomics_clinical": cb.iloc[0], "clinical": clb.iloc[0]}
    s = bd.new_slide(prs, f"{cfg['label']} - best model of each category")
    t = cls_rows(pd.DataFrame([best[t] for t, _, _ in CAT])); t.insert(0, "Category", [n for _, n, _ in CAT])
    bd.add_table(s, t, 0.3, 1.05, [2.6, 2.0, 1.5, 2.6, 2.6], font=10, row_h=0.45, align_center_from=3, header_h=0.55)
    p = NEW / key; p.mkdir(parents=True, exist_ok=True); roc_fig(cfg, best, p / "roc_best_by_mean.png")
    pic = bd.picture(s, p / "roc_best_by_mean.png", 0.9, 3.0, height=3.95); pic.left = int((prs.slide_width - pic.width) / 2)
    bd.footnote(s, "Best per category = highest mean of out-of-fold and test AUC. Left: out-of-fold predictions averaged over splits (n=%d). Right: test set (n=%d)." % (dev_n, lock_n), top=7.05)
    shap_slide(prs, f"{cfg['label']} - mean absolute SHAP", best, cfg["dirs"], lists); summary[key] = (cfg["label"], best, tabs); takehome(prs, cfg["label"], best, "cls", cfg["dirs"], tabs)
    return best


def recurrence_section(prs, D, split, fallback, lists, summary):
    D2 = OLD / "recurrence" if fallback else NEW / "recurrence"; cfg = dict(dirs={"radiomics": OLD / "recurrence", "radiomics_clinical": D2, "clinical": D2})
    tabs = load_tables(cfg, "recurrence", "rec", fallback)
    if tabs is None: print("skip recurrence"); return None
    rb, cb = per_combo(tabs["radiomics"]), per_combo(tabs["radiomics_clinical"]); clb = tabs["clinical"].sort_values("score", ascending=False).reset_index(drop=True)
    merged_slide(prs, f"Recurrence - radiomics vs radiomics + clinico-radiologic variables (top {TOP_N} each)", rb, cb, "rec", "")
    best = {"radiomics": rb.iloc[0], "radiomics_clinical": cb.iloc[0], "clinical": clb.iloc[0]}
    R = BASE / "main_pipeline_recurrence"; km_dir = NEW / "recurrence" / "km"
    spec_json = {"best": [dict(tag=t, sequences=best[t].sequence_set, family=best[t].family) for t, _, _ in CAT], "model_dirs": {k: str(v) for k, v in cfg["dirs"].items()},
                 "oof": {"radiomics": str(R / "ladder14_radiomics_post_20261021/recA_post_none_*/oof_risk.csv"), "radiomics_clinical": REC_COMB_OOF, "clinical": str(R / "clinical/clinsurv_pre_grade_*/oof_risk.csv")}, "out": str(km_dir)}
    sj = NEW / "recurrence" / "km_spec.json"; sj.parent.mkdir(parents=True, exist_ok=True); sj.write_text(json.dumps(spec_json, indent=1))
    subprocess.run(["/usr/bin/python3", str(Path(__file__).resolve().parent / "make_km_figures.py"), "--spec", str(sj)], check=True, capture_output=True)
    km = pd.read_csv(km_dir / "km_summary.csv")
    for setname, title in (("dev", "development set, out-of-fold risk"), ("lock", "test set")):
        s = bd.new_slide(prs, f"Recurrence - best model of each category: {title}"); x = 0.3
        for tag, name, _ in CAT:
            r = best[tag]; k = km[(km.tag == tag) & (km.set == setname)].iloc[0]
            row = pd.DataFrame([{"Model": f"{name}: {r.sequence_set.replace('+', ' + ') if tag != 'clinical' else 'all variables'}, {FAM[r.family].lower()}",
                                 "C-index (95% CI)": ci(r.oof_c_avg, r.oof_c_ci_lo, r.oof_c_ci_hi) if setname == "dev" else ci(r.lock_c, r.lock_c_ci_lo, r.lock_c_ci_hi), "Log-rank p": "<0.001" if k.logrank_p < 0.001 else f"{k.logrank_p:.3f}"}])
            bd.add_table(s, row, x, 1.05, [2.0, 1.2, 0.8], font=8, row_h=0.62, align_center_from=1, header_h=0.3)
            bd.picture(s, km_dir / f"km_{tag}_{setname}.png", x, 2.05, width=4.0); bd.picture(s, km_dir / f"tdauc_{tag}_{setname}.png", x + 0.25, 5.1, width=3.5); x += 4.35
        bd.footnote(s, "Best per category = highest mean of out-of-fold and test C-index. High / low risk = above / at-or-below the median risk of the set shown. Time-dependent AUC(t): censoring-weighted.", top=7.1)
    shap_slide(prs, "Recurrence - mean absolute SHAP", best, cfg["dirs"], lists); summary["rec"] = ("Recurrence", best, tabs); takehome(prs, "Recurrence", best, "rec", cfg["dirs"], tabs)
    return best


COMMON_SPEC = [("Age", "Age, years", "cont"), ("Gender", "Gender", "cat"), ("BMI", "BMI, kg/m2", "cont"), ("Cirrhosis", "Cirrhosis", "cat"), ("Capsule (on PVP yes/no)", "Capsule (on PVP)", "cat"),
               ("Mosaic appearance (yes/no)", "Mosaic appearance", "cat"), ("Macrovascular invasion (yes/no)", "Macrovascular invasion", "cat"), ("Maximum tumor size on T1 portal venous phase (PVP)", "Maximum tumor size on T1 PVP, cm", "cont"),
               GRADE_ROW, ("Serum AFP", "Serum AFP, ng/mL", "cont")]


def split_slide(prs, title, df, note):
    df = df.rename(columns=lambda c: c.replace("Lock set", "Test set"))
    s = bd.new_slide(prs, title); bd.add_table(s, df, 0.45, 1.1, [2.9, 3.3, 3.3, 0.8, 1.5], font=9, row_h=0.45, align_center_from=1, header_h=0.5)
    bd.footnote(s, "Development vs test set; the test set is held out from all training and tuning. " + note, top=6.9)


def split_slides(prs, D, tils, rec):
    """Same variable rows for both; the TILs and TLS endpoints share one split, recurrence has its own."""
    n = lambda x, k: int((x["set"] == k).sum())
    spec = COMMON_SPEC + [("TILs_resection", "TILs score", "cat"), ("TLS_resection", "TLS score", "cat")]
    split_slide(prs, "Development/test data split - TILs and TLS endpoints", v2.split_table(D, tils, spec, n(tils, "dev"), n(tils, "lock")), "The same split is used for TILs 2-3 vs 0-1, TILs 3 vs 0-2 and TLS 2 vs 0-1.")
    spec = COMMON_SPEC + [("Recurrence", "Recurrence", "cat"), ("Follow-up time (days)", "Follow-up / time to recurrence, days", "cont")]
    split_slide(prs, "Development/test data split - recurrence endpoint", v2.split_table(D, rec, spec, n(rec, "dev"), n(rec, "lock")),
                "Recurrence uses its own split (110 / 30; one patient without follow-up excluded), so patients can fall in different sets than for TILs / TLS. Follow-up time = time to recurrence or last follow-up.")


def selection_slide(prs):
    s = bd.new_slide(prs, "Model selection")
    bd.text_box(s, 0.5, 1.0, 12.3, 5.8, [("Selection rule",), "One model per sequence combination: the family (ridge / lasso / elastic net) with the highest mean of out-of-fold and test performance; (AUC for classification, Harrell C-index for recurrence).",
        "Combinations are ranked by the same score, and the best model of each category (radiomics, clinico-radiologic, combination) is the top-ranked one.",
        "Out-of-fold performance = AUC of the out-of-fold predictions averaged over 10 repeated 5-fold splits of the development set; test performance = final development-fit model applied once to the held-out test set (n=30)."], size=14, bullet=True)


def ci_overlap(a, b): return not (a[1] < b[0] or b[1] < a[0])


def _join(xs):
    xs = list(xs); return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


def takehome(prs, label, best, kind, dirs, tabs):
    """Narrative summary of one endpoint, generated from the numbers: tentative ranking, encouraging aspects, cautions, sequences and features that matter."""
    m = "C-index" if kind == "rec" else "AUC"
    g = lambda r: ((r.oof_c_avg, r.oof_c_ci_lo, r.oof_c_ci_hi), (r.lock_c, r.lock_c_ci_lo, r.lock_c_ci_hi)) if kind == "rec" else ((r.oof_auc_avgprob, r.oof_auc_ci_lo, r.oof_auc_ci_hi), (r.lock_auc, r.lock_auc_ci_lo, r.lock_auc_ci_hi))
    short = {"radiomics": "radiomics", "clinical": "clinico-radiologic variables", "radiomics_clinical": "radiomics + clinico-radiologic variables"}
    O = {t: g(best[t])[0] for t, _, _ in CAT}; T = {t: g(best[t])[1] for t, _, _ in CAT}
    order = sorted(O, key=lambda t: -O[t][0]); top, rest = order[0], order[1:]
    lines = [("Overall",)]
    lines.append(f"Tentatively higher out-of-fold performance with {short[top]} ({m} {O[top][0]:.2f}) than with {_join(f'{short[t]} ({O[t][0]:.2f})' for t in rest)}, "
                f"but all confidence intervals overlap, so the categories cannot be separated with this sample size.")
    t_order = sorted(T, key=lambda t: -T[t][0])
    lines.append(f"On the test set the best-scoring category is {short[t_order[0]]} ({m} {T[t_order[0]][0]:.2f})" + ("; the same category as out-of-fold." if t_order[0] == top else f"; a different category than out-of-fold, which fits the wide intervals (n=30)."))
    lines.append(("What is encouraging",))
    pos = [t for t in O if O[t][1] > 0.5]
    if pos:
        lines.append(f"Out-of-fold {m} 95% CI excludes 0.5 for {_join(short[t] for t in pos)}: " + _join(f"{short[t]} {O[t][0]:.2f} ({O[t][1]:.2f}-{O[t][2]:.2f})" for t in pos) + ".")
    else: lines.append("No category has an out-of-fold CI that excludes 0.5.")
    consistent = [t for t in O if T[t][0] >= O[t][0] and O[t][1] > 0.5]
    if consistent: lines.append(f"Test {m} is at least as high as the out-of-fold {m} for {_join(short[t] for t in consistent)}, i.e. the signal does not disappear in unseen patients.")
    nodisc = [t for t in O if O[t][1] <= 0.5]
    lines.append(("Caveats",))
    if nodisc: lines.append(f"{_join(short[t] for t in nodisc).capitalize()}: out-of-fold CI includes 0.5 (no demonstrable discrimination)" + ("; one model even falls below 0.5 out-of-fold, which is chance-level noise." if any(O[t][0] < 0.5 for t in nodisc) else "."))
    lines.append("Selection among 14 sequence combinations x 3 model families is itself optimistic; treat the ranking as hypothesis-generating.")
    # sequences: how often each sequence appears among the top-5 models
    from collections import Counter
    cnt = Counter()
    for t in ("radiomics", "radiomics_clinical"):
        for sset in per_combo(tabs[t]).head(TOP_N).sequence_set: cnt.update(sset.split("+"))
    seqs = [f"{v3.SEQ.get(k, k)} ({n}/10)" for k, n in cnt.most_common(3)]
    lines.append(("Most relevant sequences and features",))
    lines.append("Sequences most often in the top-5 models (radiomics and combined): " + _join(seqs) + ".")
    def top_feats(tag, want_clin, n=3):
        sh = shap_top(dirs, tag, best[tag])
        if sh is None: return []
        f = [x for x in sh.feature if (x.startswith("clin_") == want_clin)]; return [pretty(x) for x in f[:n]]
    rf = top_feats("radiomics", False) or top_feats("radiomics_clinical", False); cf = top_feats("clinical", True) or top_feats("radiomics_clinical", True)
    if rf: lines.append("Leading radiomic features: " + "; ".join(rf) + ".")
    if cf: lines.append("Leading clinico-radiologic features: " + "; ".join(x.replace("Clinical · ", "") for x in cf) + ".")
    s = bd.new_slide(prs, f"{label} - take-home message"); bd.text_box(s, 0.5, 1.0, 12.3, 5.9, lines, size=13, bullet=True)


def conclusions(prs, summary):
    lines = []
    for key, (label, best, tabs) in summary.items():
        r, c, k = best["radiomics"], best["radiomics_clinical"], best["clinical"]; m = lambda x, rec=(key == "rec"): (x.oof_c_avg, x.lock_c) if rec else (x.oof_auc_avgprob, x.lock_auc)
        nm = lambda x: x.sequence_set.replace("+", " + ") if x.sequence_set not in ("clinical", "clinicoradiologic") else "all variables"
        lines += [(label,), f"Radiomics {nm(r)} ({FAM[r.family].lower()}): out-of-fold {m(r)[0]:.2f}, test {m(r)[1]:.2f}; clinico-radiologic alone: out-of-fold {m(k)[0]:.2f}, test {m(k)[1]:.2f}; combination {nm(c)}: out-of-fold {m(c)[0]:.2f}, test {m(c)[1]:.2f}."]
    lines += [("Caveats",), "Differences between categories are small relative to the 95% CIs; the test set is small (n=30)."]
    s = bd.new_slide(prs, "Conclusions"); bd.text_box(s, 0.5, 1.0, 12.3, 5.9, lines, size=12, bullet=True)
    s = bd.new_slide(prs, "Next steps...")
    bd.text_box(s, 0.5, 1.0, 12.3, 5.0, ["Extend the cohort (extended analysis, N=220) for external-style validation of frozen models.", "Target IHC-derived endpoints.", "Target assessment of response to ICIs in SA2."], size=14, bullet=True)


# ---------------------------------------------------------------- merge: old lasso / elastic net combination rows + new (capped) ridge rows
OLDP = BASE / "presentation_20261022"; PV = BASE / "presentation_v5"; DECK = PV / "deck"; VROOT = BASE / "v5_ridge_cap20"; NEW = DECK
LABEL = {"tils_ge2": "tilsGE2cr", "tils_ge3": "tilsGE3cr", "tls_ge2": "tlsGE2cr"}


def merge_ep(key):
    """DECK/<key>: combination table = old lasso + elastic net rows + new ridge rows; clinical-only table/models copied; merged OOF for ROC / KM."""
    import run_guard
    prefix = key if key != "recurrence" else "recurrence"; old, raw, out = OLDP / key, PV / key, DECK / key; (out / "models").mkdir(parents=True, exist_ok=True)
    t_old = pd.read_csv(old / f"{prefix}_model_table_radiomics_clinical.csv"); t_new = pd.read_csv(raw / f"{prefix}_model_table_radiomics_clinical.csv")
    assert set(t_new.family) == {"ridge"} and len(t_new) == 14, (key, t_new.family.unique(), len(t_new))
    assert list(t_old.columns) == list(t_new.columns)
    pd.concat([t_old[t_old.family != "ridge"], t_new], ignore_index=True).to_csv(out / f"{prefix}_model_table_radiomics_clinical.csv", index=False)
    shutil.copy(old / f"{prefix}_model_table_clinical.csv", out / f"{prefix}_model_table_clinical.csv"); shutil.copytree(old / "models/clinical", out / "models/clinical", dirs_exist_ok=True)
    md = out / "models/radiomics_clinical"; md.mkdir(parents=True, exist_ok=True)
    for d in (old / "models/radiomics_clinical").iterdir():
        if d.is_dir() and not d.name.endswith("__ridge"): shutil.copytree(d, md / d.name, dirs_exist_ok=True)
    for d in (raw / "models/radiomics_clinical").iterdir():
        if d.is_dir() and d.name.endswith("__ridge"): shutil.copytree(d, md / d.name, dirs_exist_ok=True)
    if key == "recurrence":
        fn, g_old, g_new = "oof_risk.csv", str(BASE / "main_pipeline_recurrence/grade_ladder14_clinrad_post_20261022/recB_post_forced_*/oof_risk.csv"), str(VROOT / "recurrence/recB_post_compete_*/oof_risk.csv")
    else:
        fn, g_old, g_new = "oof_predictions.csv", v3.ep_cfg(key)["oof"]["radiomics_clinical"], str(VROOT / key / f"{LABEL[key]}_post_*/oof_predictions.csv")
    o = [pd.read_csv(f) for f in run_guard.usable_files(g_old, verbose=False)]; o = [x[x.family != "ridge"] for x in o]
    n = [pd.read_csv(f) for f in run_guard.usable_files(g_new, verbose=False)]; assert len(n) == 14, (key, len(n))
    m = out / "oof_merged/run"; m.mkdir(parents=True, exist_ok=True); pd.concat(o + n, ignore_index=True).to_csv(m / fn, index=False); (m / "manifest.json").write_text(json.dumps({"finished": True, "note": "old lasso/elastic net + capped ridge"}))


def ep_cfg_v5(key, fb=False):
    c = v3.ep_cfg(key, False); D = DECK / key
    c["dirs"] = {"radiomics": OLD / key, "radiomics_clinical": D, "clinical": D}; c["oof"]["radiomics_clinical"] = str(D / "oof_merged/*/oof_predictions.csv"); return c


ep_cfg = ep_cfg_v5
REC_COMB_OOF = str(DECK / "recurrence/oof_merged/*/oof_risk.csv")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--allow-fallback", action="store_true"); a = ap.parse_args()
    for k in ("tils_ge2", "tils_ge3", "tls_ge2", "recurrence"): merge_ep(k)
    prs = Presentation(bd.TEMPLATE); ids = prs.slides._sldIdLst
    for sid in list(ids)[1:]: prs.part.drop_rel(sid.rId); ids.remove(sid)
    bd.text_box(prs.slides[0], 0.26, 5.6, 10, 0.5, [f"Results update, {date.today():%m/%d/%y}"], size=16)
    D, r = bt.build_frame()
    gm = pd.read_csv(BASE / "main_pipeline_recurrence/clinical/clinical_matrix.csv").set_index("PatientID").grade
    D["Tumor grade"] = gm.map(GRADE).reindex(D.index); bt.ORDER["Tumor grade"] = ["G1", "G2", "G3", "G4"]
    tils = pd.read_csv(BASE / "main_pipeline/final/cohort_v3_141/split/split.csv").set_index("PatientID"); rec = pd.read_csv(BASE / "main_pipeline_recurrence/split/recurrence_split.csv").set_index("PatientID")
    v3.front_matter(prs)
    # cohort slide with tumor grade row
    v2_add = bd.add_table
    def add_table_grade(slide, df, left, top, widths, **kw):
        if "Variable" in df.columns and len(df) and str(df["Variable"].iloc[0]).startswith("Etiology"):
            txt, n = v2.cat_str(D["Tumor grade"], bt.ORDER["Tumor grade"]); df = pd.concat([df, pd.DataFrame([{"Variable": f"Tumor grade (n={n})", "Value": txt}])], ignore_index=True)
        return v2_add(slide, df, left, top, widths, **kw)
    bd.add_table = add_table_grade
    try: v2.cohort_slide(prs, D)
    finally: bd.add_table = v2_add
    split_slides(prs, D, tils, rec); selection_slide(prs)
    lists, summary = [], {}
    ex = [("TILs_resection", "TILs score", "cat"), ("TLS_resection", "TLS score", "cat")]
    classification_section(prs, "tils_ge2", D, tils, ex, a.allow_fallback, lists, summary); classification_section(prs, "tils_ge3", D, tils, ex, a.allow_fallback, lists, summary)
    classification_section(prs, "tls_ge2", D, tils, ex[::-1], a.allow_fallback, lists, summary); recurrence_section(prs, D, rec, a.allow_fallback, lists, summary)
    pd.concat([sh.assign(endpoint=ep, category=cat, model=label) for ep, cat, label, sh in lists]).to_csv(NEW / "plotted_model_feature_lists.csv", index=False) if lists else None
    conclusions(prs, summary)
    dest = PV / f"updates-{date.today():%m-%d-%y}_claude_v5.pptx"; PV.mkdir(parents=True, exist_ok=True); prs.save(dest); print("saved", dest, "slides:", len(prs.slides))


if __name__ == "__main__":
    main()
