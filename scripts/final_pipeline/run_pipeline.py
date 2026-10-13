"""Config-driven orchestrator for the TILs radiomics analysis.

    python run_pipeline.py --config config.yaml --dry-run      # print the cells and commands, run nothing
    python run_pipeline.py --config config.yaml                # run everything (resumes finished cells)
    python run_pipeline.py --config config.yaml --only tils_ge3 --max-cells 2 --force

What it does, in order (each step aborts on a violated guard rail instead of warning):
  1. Cohort: merges the per-cohort radiomics sheets, checks IDs/columns, and checks that the
     mastersheet-derived outcomes + clinical tables cover exactly the same patients.
  2. Split: one dev/lock split per run (make_split.py), stratified on the 4 TILs levels, shared by every cell.
  3. Cells: design x sequence-combo. Each cell runs the existing stages of main_pipeline.py (binary) or
     tils_regression_pipeline.py (ordinal): partition (nested-CV OOF + feature selection) and, only for
     cells listed in evaluation.lock.preregistered, gridsearch + train (lock evaluation).
  4. Summary: cell_summary.csv (one row per cell) and run_manifest.json (config, git commit, cohort).

Not implemented yet: the permutation null (evaluation.permutation_null). Until it is, every summary is
stamped with the number of cells so the best-of-N OOF value is read as uncorrected.
"""
import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
PIPELINES = {"binary": SCRIPTS / "tils_classification" / "main_pipeline.py",
             "ordinal": SCRIPTS / "tils_regression" / "tils_regression_pipeline.py"}
sys.path.insert(0, str(HERE))
from make_split import make_split  # noqa: E402


# ----------------------------------------------------------------------------- config / paths
def load_config(path):
    cfg = yaml.safe_load(open(path))
    # RADIOMICS_ROOT overrides paths.root so the same config runs on another machine
    # (OneDrive mounts at different paths per machine).
    root = Path(os.environ.get("RADIOMICS_ROOT", cfg["paths"]["root"])).expanduser()
    cfg["_root"] = root
    cfg["_sheets"] = root / cfg["paths"]["inputs"]["radiomics_sheets"]
    cfg["_db"] = root / cfg["paths"]["inputs"]["databases"]
    cfg["_run_dir"] = root / cfg["paths"]["outputs"] / cfg["run"]["name"]
    return cfg


def sheet_path(cfg, cohort):
    stem = cohort.split("/")[-1].replace("_pre_new", "")      # SA2_pre_new folder holds R-SA2_*.csv
    return cfg["_sheets"] / cohort / f"R-{stem}_I-sitkBSpline_S-1mm_W-{cfg['data']['binwidth']}.csv"


def reader_path(cfg, key, label):
    c = cfg["data"][key]
    return cfg["_sheets"] / c / f"R-{label}_I-sitkBSpline_S-1mm_W-{cfg['data']['binwidth']}.csv"


# ----------------------------------------------------------------------------- 1. cohort
def build_cohort(cfg):
    frames = []
    for c in cfg["data"]["features"]:
        p = sheet_path(cfg, c["cohort"])
        if not p.exists():
            raise FileNotFoundError(f"{c['cohort']}: {p} not found (binwidth={cfg['data']['binwidth']})")
        frames.append(pd.read_csv(p, index_col=0))
    cols = frames[0].columns
    for c, f in zip(cfg["data"]["features"][1:], frames[1:]):
        if set(f.columns) != set(cols):
            raise ValueError(f"{c['cohort']}: {len(set(f.columns) ^ set(cols))} columns differ from the first cohort")
    merged = pd.concat([f.reindex(columns=cols) for f in frames])
    merged.index.name = cfg["data"]["id_col"]
    if merged.index.duplicated().any():
        raise ValueError(f"Patient in more than one cohort: {merged.index[merged.index.duplicated()].tolist()}")

    outcomes = pd.read_excel(cfg["_db"] / cfg["data"]["outcomes_xlsx"])
    clinical = pd.read_excel(cfg["_db"] / cfg["data"]["clinical_xlsx"])
    ids = set(merged.index)
    problems = []
    if ids != set(outcomes.PatientID):
        problems.append(f"features vs outcomes: only features={sorted(ids - set(outcomes.PatientID))}, "
                        f"only outcomes={sorted(set(outcomes.PatientID) - ids)}")
    if ids != set(clinical.PatientID):
        problems.append(f"features vs clinical: only features={sorted(ids - set(clinical.PatientID))}, "
                        f"only clinical={sorted(set(clinical.PatientID) - ids)}")
    no_label = sorted(set(outcomes.loc[outcomes[cfg["data"]["label_col"]].isna(), "PatientID"]) & ids)
    if no_label and cfg["data"].get("require_outcome_and_clinical_rows", True):
        problems.append(f"patients with features but no {cfg['data']['label_col']}: {no_label}")
    exp = cfg["data"].get("expected_n_features_patients")
    if exp is not None and len(ids) != exp:
        problems.append(f"expected {exp} patients with features, found {len(ids)}")
    if problems:
        raise SystemExit("COHORT GUARD RAILS FAILED:\n  - " + "\n  - ".join(problems))
    return merged


# ----------------------------------------------------------------------------- 3. cells
def enumerate_combos(cfg):
    s = cfg["sequences"]
    if s["mode"] == "explicit":
        return [list(c) for c in s["explicit"]]
    e = s["enumerate"]
    out = []
    for n in range(e["min_size"], e["max_size"] + 1):
        for c in itertools.combinations(s["pool"], n):
            if e.get("require_any_of") and not set(c) & set(e["require_any_of"]):
                continue
            out.append(list(c))
    return out


def cell_dir(cfg, design_id, combo):
    return cfg["_run_dir"] / "cells" / design_id / "_".join(combo)


def selection_args(cfg, design):
    sel = cfg["selection"]
    method = {"stability": "stability", "fold_consensus": "fold-consensus"}.get(sel["method"])
    if method is None:
        raise SystemExit(f"selection.method '{sel['method']}' is not supported by the wrapped pipelines yet "
                         f"(supported: stability, fold_consensus)")
    a = ["--feature-selection-method", method,
         "--stability-threshold", str(sel["stability"]["threshold"]),
         "--stability-n-resamples", str(sel["stability"]["n_resamples"]),
         "--stability-resample-frac", str(sel["stability"]["resample_frac"]),
         "--decorrelation-threshold", str(sel["decorrelation_threshold"]),
         "--target-feature-count", str(sel["target_feature_count"])]
    a += ["--cluster-threshold", str(sel["stability"]["cluster_threshold"])]
    if sel["penalty"] not in ("l1", "elasticnet"):
        raise SystemExit("selection.penalty must be l1 or elasticnet")
    a += ["--penalty", sel["penalty"], "--l1-ratios"] + [str(x) for x in sel["l1_ratios"]]
    return a


def common_args(cfg, design, features_csv, split_csv, combo, clinical_dir):
    d, p = cfg["data"], cfg["preprocessing"]
    a = ["--sequences"] + combo + [
        "--exclude-feature-families"] + p["exclude_feature_families"] + [
        "--features-path", str(features_csv),
        "--path-r1", str(reader_path(cfg, "reader1_cohort", "Reader1")),
        "--path-r2", str(reader_path(cfg, "reader2_cohort", "Reader2")),
        "--outcomes-path", str(cfg["_db"] / d["outcomes_xlsx"]),
        "--clinical-data-path", str(cfg["_db"] / d["clinical_xlsx"]),
        "--label-col", d["label_col"],
        "--split-file", str(split_csv),
        "--cv-threshold", str(p["cv_threshold"]),
        "--icc-threshold", str(p["icc_threshold"]),
        "--feature-set", "radiomics",
        "--clinical-output-dir", str(clinical_dir),
        "--n-jobs", str(cfg["run"]["n_jobs"]),
    ]
    if design["type"] == "binary":
        a += ["--cutoff", str(design["cutoff"])]
    return a


def run(cmd, log):
    t = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(" ".join(cmd) + "\n\n" + proc.stdout + "\n--- STDERR ---\n" + proc.stderr)
    return proc.returncode, time.time() - t


def read_oof(part_dir, design):
    p = part_dir / "nested_cv_diagnostics.csv"
    if not p.exists():
        return {}
    d = pd.read_csv(p)
    key, lo, hi = (("oof_auc", "auc_ci_lower", "auc_ci_upper") if design["type"] == "binary"
                   else ("oof_spearman_rho", "rho_ci_lower", "rho_ci_upper"))
    b = d.loc[d[key].idxmax()]
    return {"oof_metric": key, "oof_best_model": b["model"], "oof_best": round(float(b[key]), 4),
            "oof_ci_lower": round(float(b[lo]), 4), "oof_ci_upper": round(float(b[hi]), 4),
            "oof_median_all_models": round(float(d[key].median()), 4), "n_dev": int(b["n_dev"])}


def read_lock(model_dir):
    p = model_dir / "model_comparison.csv"
    if not p.exists():
        return {}
    d = pd.read_csv(p)
    col = "AUC" if "AUC" in d.columns else "Spearman_rho"
    v = pd.to_numeric(d[col], errors="coerce")
    b = d.loc[v.idxmax()]
    return {"lock_metric": col, "lock_best_model": b["Model"], "lock_best": b[col]}


# ----------------------------------------------------------------------------- main
def git_commit():
    try:
        return subprocess.run(["git", "-C", str(SCRIPTS), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", nargs="*", default=None, help="design ids to run")
    ap.add_argument("--max-cells", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="rerun cells whose outputs already exist")
    a = ap.parse_args()
    cfg = load_config(a.config)
    run_dir = cfg["_run_dir"]
    py = cfg["run"]["python"] if Path(cfg["run"]["python"]).exists() else sys.executable

    features = build_cohort(cfg)
    print(f"cohort OK: {len(features)} patients x {features.shape[1]} columns "
          f"({[c['cohort'] for c in cfg['data']['features']]}, binwidth={cfg['data']['binwidth']})")

    icc = pd.read_csv(reader_path(cfg, "reader1_cohort", "Reader1"), index_col=0, usecols=[0]).index
    split, _ = make_split(cfg["_db"] / cfg["data"]["outcomes_xlsx"], icc, cfg["data"]["label_col"],
                          cfg["preprocessing"]["split"]["test_size"], cfg["run"]["seeds"]["partition"],
                          feature_ids=features.index)
    lab = cfg["data"]["label_col"]
    print("split:", split.groupby("set")[lab].value_counts().unstack(fill_value=0).to_dict("index"))

    designs = [d for d in cfg["designs"] if a.only is None or d["id"] in a.only]
    combos = enumerate_combos(cfg)
    cells = [(d, c) for d in designs for c in combos]
    if a.max_cells:
        cells = cells[:a.max_cells]
    prereg = set(cfg["evaluation"]["lock"].get("preregistered") or [])
    lock_mode = cfg["evaluation"]["lock"]["run_for"]
    print(f"{len(cells)} cells = {len(designs)} designs x {len(combos)} combos; "
          f"lock evaluation: {lock_mode} ({len(prereg)} preregistered)")
    if cfg["evaluation"]["permutation_null"]["enabled"]:
        print("NOTE: permutation null is enabled in the config but NOT implemented yet -- "
              "best-of-N OOF values below are uncorrected for the number of cells.")

    if a.dry_run:
        for d, c in cells:
            print(f"  {d['id']}/{'_'.join(c)}  [{d['type']}]")
        return

    (run_dir / "inputs").mkdir(parents=True, exist_ok=True)
    features_csv = run_dir / "inputs" / f"features_merged_{cfg['data']['binwidth']}.csv"
    features.to_csv(features_csv)
    split_csv = run_dir / "split" / "split.csv"
    split_csv.parent.mkdir(parents=True, exist_ok=True)
    split.to_csv(split_csv, index=False)
    clinical_dir = run_dir / "split"
    (run_dir / "run_manifest.json").write_text(json.dumps({
        "started": datetime.now().isoformat(timespec="seconds"), "git_commit": git_commit(),
        "n_patients": len(features), "n_cells": len(cells), "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
    }, indent=2, default=str))

    rows = []
    summary_csv = run_dir / "cell_summary.csv"
    for i, (d, combo) in enumerate(cells, 1):
        name = f"{d['id']}/{'_'.join(combo)}"
        base = cell_dir(cfg, d["id"], combo)
        part, params, model = base / "partition", base / "params", base / "model"
        script = str(PIPELINES[d["type"]])
        row = {"design": d["id"], "sequences": "_".join(combo), "n_sequences": len(combo)}
        done = (part / "nested_cv_diagnostics.csv").exists() and (part / "consensus_features.csv").exists()
        if done and not a.force:
            print(f"[{i}/{len(cells)}] {name}: reusing finished partition", flush=True)
            row["partition_status"] = "reused"
        else:
            print(f"[{i}/{len(cells)}] {name}: partition ...", flush=True)
            cmd = ([py, script, "--stage", "partition", "--output-dir", str(part)]
                   + common_args(cfg, d, features_csv, split_csv, combo, clinical_dir) + selection_args(cfg, d))
            rc, sec = run(cmd, base / "logs" / "partition.log")
            row.update(partition_status="ok" if rc == 0 else "failed", partition_sec=round(sec))
            if rc != 0:
                tail = (base / "logs" / "partition.log").read_text().strip().splitlines()
                row["error"] = next((l for l in reversed(tail) if l.strip() and "STDERR" not in l), "")[:200]
                if "No consensus features" in row["error"]:
                    row["partition_status"] = "no_stable_features"   # a legitimate result, not a crash
                print(f"    {row['partition_status'].upper()} ({row['error']})")
        cf = part / "consensus_features.csv"
        row["n_features"] = len(pd.read_csv(cf)) if cf.exists() else 0
        row.update(read_oof(part, d))

        do_lock = lock_mode == "all" or (lock_mode == "preregistered_only" and name in prereg)
        if do_lock and row.get("partition_status") in ("ok", "reused") and row["n_features"] > 0:
            models = cfg["models"][d["type"]]
            for stage, extra in (("gridsearch", ["--input-dir", str(part), "--output-dir", str(params)]),
                                 ("train", ["--input-data-dir", str(part), "--input-params-dir", str(params),
                                            "--output-dir", str(model), "--bootstrap",
                                            str(cfg["evaluation"]["nested_cv"]["bootstrap_ci"])])):
                rc, sec = run([py, script, "--stage", stage] + extra + ["--models"] + models,
                              base / "logs" / f"{stage}.log")
                row[f"{stage}_status"] = "ok" if rc == 0 else "failed"
                if rc != 0:
                    break
            row.update(read_lock(model))
        rows.append(row)
        out = pd.DataFrame(rows)
        out["n_cells_in_run"] = len(cells)
        out.to_csv(summary_csv, index=False)

    print("\n" + "=" * 100)
    print(f"RUN COMPLETE: {len(rows)} cells.  Summary: {summary_csv}")
    show = [c for c in ("design", "sequences", "partition_status", "n_features", "oof_best_model", "oof_best",
                        "oof_ci_lower", "oof_ci_upper", "n_dev", "lock_best_model", "lock_best") if c in out.columns]
    print(out[show].to_string(index=False))
    print(f"\nbest-of-{len(rows)} OOF values are UNCORRECTED for multiplicity (permutation null not implemented).")


if __name__ == "__main__":
    main()
