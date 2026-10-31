"""Repeated nested-CV classification restricted to the untreated SA1 roster.

This is a standalone runner. It imports the established pipeline components but
does not alter their source files. All paths are explicit or relative to
RADIOMICS_PROJECT_ROOT / RADIOMICS_SHEETS_DIR.
"""
from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
import shutil
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedKFold


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def regular(path: Path) -> Path:
    path = path.expanduser().absolute()
    if path.is_symlink():
        raise RuntimeError(f"Symlink prohibited: {path}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=os.environ.get("RADIOMICS_PROJECT_ROOT"))
    ap.add_argument("--sheets-dir", default=os.environ.get("RADIOMICS_SHEETS_DIR"))
    ap.add_argument("--gpt-dir", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--clinical-matrix", required=True)
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--families", nargs="+", default=["ridge", "lasso", "elastic_net"])
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--output-parent", required=True)
    ap.add_argument("--label", default="tilsGE2_sa1_untreated")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    if not args.project_root:
        raise RuntimeError("Provide --project-root or RADIOMICS_PROJECT_ROOT")
    project = Path(args.project_root).expanduser().absolute()
    sheets = Path(args.sheets_dir).expanduser().absolute() if args.sheets_dir else project / "Results/Radiomics_sheets/Annuli_v2"
    sa1_sheet = regular(sheets / "SA1/R-SA1_I-sitkBSpline_S-1mm_W-perseq.csv")
    reader1 = regular(sheets / "inter-reader/Reader1/R-Reader1_I-sitkBSpline_S-1mm_W-perseq.csv")
    reader2 = regular(sheets / "inter-reader/Reader2/R-Reader2_I-sitkBSpline_S-1mm_W-perseq.csv")
    split_path = regular(Path(args.split))
    clinical_path = regular(Path(args.clinical_matrix))

    sys.path.insert(0, args.gpt_dir)
    import run_compact_ridge_sequence_search as core
    import run_sparse_dynamic_extensions as sparse

    ids = pd.read_csv(sa1_sheet, usecols=["PatientID"])["PatientID"].astype(str).tolist()
    if len(ids) != 111 or len(set(ids)) != 111:
        raise RuntimeError(f"Expected 111 unique SA1 IDs, got {len(ids)} rows / {len(set(ids))} unique")
    clinical = pd.read_csv(clinical_path).set_index("PatientID")
    untreated = set(clinical.index[(clinical.treat_io == 0) & (clinical.treat_io_sbrt == 0)].astype(str))
    if set(ids) != untreated:
        raise RuntimeError(f"SA1/treatment roster mismatch: SA1-only={sorted(set(ids)-untreated)}, untreated-only={sorted(untreated-set(ids))}")
    labels = pd.read_csv(split_path).set_index("PatientID")["TILs_resection"].reindex(ids)
    if labels.isna().any():
        raise RuntimeError(f"Missing TIL labels: {labels[labels.isna()].index.tolist()}")
    y = (labels >= 2).astype(int)
    if (int(y.sum()), int((1-y).sum())) != (90, 21):
        raise RuntimeError(f"Expected TILs>=2 distribution 90/21, got {int(y.sum())}/{int((1-y).sum())}")

    X = pd.read_csv(sa1_sheet, index_col=0, usecols=lambda c: c == "PatientID" or "_original_" in c).loc[ids]
    X = X.reindex(sorted(X.columns), axis=1).astype(float)
    if X.shape != (111, 963):
        raise RuntimeError(f"Expected (111,963), got {X.shape}")
    for rp in (reader1, reader2):
        R = pd.read_csv(rp, index_col=0, usecols=lambda c: c == "PatientID" or "_original_" in c)
        if R.shape != (23, 963) or np.isinf(R.to_numpy(float)).any():
            raise RuntimeError(f"Unexpected reader sheet {rp}: {R.shape}")
        if not set(R.index.astype(str)).issubset(set(ids)):
            raise RuntimeError(f"Reader IDs outside SA1 roster: {rp}")

    run = Path(args.output_parent) / f"{args.label}_{'-'.join(s.replace('+','_') for s in args.sets)}_{datetime.now():%Y%m%d_%H%M%S}"
    run.mkdir(parents=True, exist_ok=False)
    inputs = [sa1_sheet, reader1, reader2, split_path, clinical_path]
    audit = {
        "status": "PASS", "population": "111 untreated SA1 patients", "n": 111,
        "tils_ge2": 90, "tils_lt2": 21, "original_features": 963,
        "inputs": [{"path": str(p), "size": p.stat().st_size, "mtime": p.stat().st_mtime, "sha256": sha256(p)} for p in inputs],
    }
    (run / "input_preflight.json").write_text(json.dumps(audit, indent=2) + "\n")
    (run / "patient_roster.csv").write_text("PatientID,TILs_resection,tils_ge2\n" + "\n".join(f"{i},{labels.loc[i]},{y.loc[i]}" for i in ids) + "\n")
    manifest = {
        "started": datetime.now().isoformat(), "population": "SA1 untreated only", "n": 111,
        "endpoint": "TILs>=2", "sets": args.sets, "families": args.families,
        "repeats": args.repeats, "outer": "5-fold stratified, seed 42+repeat",
        "inner": "3-fold, seed 420+fold+1000*repeat", "icc": "fold-wise reader ICC",
        "project_root": str(project), "sheets_dir": str(sheets), "sheet": str(sa1_sheet),
        "readers": [str(reader1), str(reader2)], "sklearn": sklearn.__version__,
        "source_runner": str(Path(__file__).absolute()), "lock_evaluated": False,
    }
    (run / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("RUN_DIRECTORY=" + str(run), flush=True)

    memory_dir = Path("/private/tmp") / run.name
    memory = joblib.Memory(memory_dir, verbose=0)
    atexit.register(shutil.rmtree, str(memory_dir), True)
    rows: list[dict] = []
    oof_rows: list[pd.DataFrame] = []
    selected_rows: list[dict] = []
    readers = [str(reader1), str(reader2)]

    for set_name in args.sets:
        cols = sorted(c for c in X.columns if any(c.startswith(seq + "_original_") for seq in set_name.split("+")))
        Xs = X[cols]
        if not cols:
            raise RuntimeError(f"No columns for sequence set {set_name}")
        for family in args.families:
            for repeat in range(args.repeats):
                started = time.time()
                pred = np.full(len(y), np.nan)
                selected_counts = []
                try:
                    outer = StratifiedKFold(5, shuffle=True, random_state=42 + repeat)
                    for fold, (tr, va) in enumerate(outer.split(Xs, y), 1):
                        inner = list(StratifiedKFold(3, shuffle=True, random_state=420 + fold + 1000 * repeat).split(Xs.iloc[tr], y.iloc[tr]))
                        icc, estimator, params, score, fit_audit, _ = sparse.fit_one(family, Xs.iloc[tr], y.iloc[tr], inner, readers, memory, args.jobs)
                        pred[va] = estimator.predict_proba(icc.transform(Xs.iloc[va]))[:, 1]
                        features = list(fit_audit.get("selected_features", []))
                        selected_counts.append(len(features))
                        selected_rows.extend({"sequence_set": set_name, "family": family, "repeat": repeat, "fold": fold, "feature": f} for f in features)
                except (ValueError, RuntimeError) as exc:
                    rows.append({"sequence_set": set_name, "family": family, "repeat": repeat, "auc": np.nan, "average_precision": np.nan, "brier": np.nan, "log_loss": np.nan, "prevalence": float(y.mean()), "mean_selected_features": np.nan, "failed_reason": str(exc)[:200]})
                    pd.DataFrame(rows).to_csv(run / "per_repeat_auc.csv", index=False)
                    print(f"[{set_name}|{family}] repeat {repeat}: FAILED {str(exc)[:100]}", flush=True)
                    continue
                metrics = core.classification_metrics(y, pred)
                rows.append({"sequence_set": set_name, "family": family, "repeat": repeat, **metrics, "mean_selected_features": float(np.mean(selected_counts))})
                oof_rows.append(pd.DataFrame({"sequence_set": set_name, "family": family, "repeat": repeat, "PatientID": ids, "y": y.values, "prob": pred}))
                pd.DataFrame(rows).to_csv(run / "per_repeat_auc.csv", index=False)
                pd.concat(oof_rows, ignore_index=True).to_csv(run / "oof_predictions.csv", index=False)
                pd.DataFrame(selected_rows).to_csv(run / "selected_features_by_fold.csv", index=False)
                print(f"[{set_name}|{family}] repeat {repeat}: AUC {metrics['auc']:.3f} ({time.time()-started:.0f}s)", flush=True)

    detail = pd.DataFrame(rows)
    summary = detail.groupby(["sequence_set", "family"]).auc.agg(mean_auc="mean", sd_auc="std", min_auc="min", max_auc="max", n_ok=lambda z: int(z.notna().sum()), n_failed=lambda z: int(z.isna().sum())).reset_index().round(4)
    summary.to_csv(run / "summary.csv", index=False)
    manifest["finished"] = datetime.now().isoformat()
    manifest["n_models"] = int(len(summary))
    (run / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("COMPLETED=" + str(run), flush=True)


if __name__ == "__main__":
    main()
