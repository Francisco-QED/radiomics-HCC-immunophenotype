"""Dispatch the complete SA1-only TIL and recurrence sweeps with bounded concurrency."""
from __future__ import annotations

import concurrent.futures
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


SETS = [
    "tp", "pvp", "dwi_b2", "hbp", "tp+pvp", "tp+dwi_b2", "tp+hbp",
    "pvp+dwi_b2", "pvp+hbp", "dwi_b2+hbp", "tp+pvp+dwi_b2",
    "tp+pvp+hbp", "tp+dwi_b2+hbp", "pvp+dwi_b2+hbp",
]
FAMILIES = ["ridge", "lasso", "elastic_net"]


def combine(root: Path, kind: str, filename: str, output: str) -> None:
    files = sorted((root / kind).glob(f"*/{filename}"))
    if files:
        pd.concat([pd.read_csv(p).assign(run_directory=str(p.parent)) for p in files], ignore_index=True).to_csv(root / output, index=False)


def main() -> None:
    if len(sys.argv) != 6:
        raise SystemExit("usage: dispatch.py PROJECT_ROOT SHEETS_ROOT REPO_ROOT OUTPUT_ROOT SURVIVAL_PY")
    project, sheets, repo, output_base, survival_py = map(Path, sys.argv[1:])
    run_root = output_base / f"full_run_{datetime.now():%Y%m%d_%H%M%S}"
    (run_root / "logs").mkdir(parents=True, exist_ok=False)
    (run_root / "tils_runs").mkdir()
    (run_root / "recurrence_runs").mkdir()
    gpt = repo / "scripts/final_pipeline/vendored_functions"
    split = project / "Results/main_pipeline_claude/main_pipeline/final/cohort_v3_141/split/split.csv"
    clinical = project / "Results/main_pipeline_gpt/presentation_20261022/v7_grade_inclusive_20261022/shared_inputs/clinical_matrix.csv"
    recurrence_split = output_base / "sa1_recurrence_all.csv"
    env = os.environ.copy()
    env["RADIOMICS_PROJECT_ROOT"] = str(project)
    env["RADIOMICS_SHEETS_DIR"] = str(sheets)
    tasks = []
    for seq in SETS:
        tasks.append(("tils", seq, [
            sys.executable, str(repo / "scripts/final_pipeline/run_sa1_untreated_tils.py"),
            "--project-root", str(project), "--sheets-dir", str(sheets), "--gpt-dir", str(gpt),
            "--split", str(split), "--clinical-matrix", str(clinical), "--sets", seq,
            "--families", *FAMILIES, "--repeats", "10", "--jobs", "1",
            "--output-parent", str(run_root / "tils_runs"), "--label", "tilsGE2_sa1_untreated",
        ]))
    for seq in SETS:
        tasks.append(("recurrence", seq, [
            str(survival_py), str(repo / "scripts/final_pipeline/run_sa1_untreated_recurrence.py"),
            "--gpt-dir", str(gpt), "--sheet", "post", "--sets", seq,
            "--families", *FAMILIES, "--clinical", "none", "--repeats", "10", "--jobs", "1",
            "--split", str(recurrence_split), "--output-parent", str(run_root / "recurrence_runs"),
            "--label", "recurrence_sa1_untreated",
        ]))

    manifest = {
        "started": datetime.now().isoformat(), "run_root": str(run_root), "population": "SA1 untreated only",
        "sets": SETS, "families": FAMILIES, "repeats": 10, "max_concurrent_processes": 8,
        "project_root": str(project), "sheets_root": str(sheets), "repo_root": str(repo),
    }
    (run_root / "dispatch_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    statuses = []

    def run_one(task):
        endpoint, seq, cmd = task
        log_path = run_root / "logs" / f"{endpoint}__{seq.replace('+','_')}.log"
        started = datetime.now()
        with log_path.open("w") as log:
            log.write("COMMAND=" + json.dumps(cmd) + "\n")
            log.flush()
            result = subprocess.run(cmd, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
        return {
            "endpoint": endpoint, "sequence_set": seq, "returncode": result.returncode,
            "started": started.isoformat(), "finished": datetime.now().isoformat(), "log": str(log_path),
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        future_map = {pool.submit(run_one, task): task for task in tasks}
        for future in concurrent.futures.as_completed(future_map):
            status = future.result()
            statuses.append(status)
            pd.DataFrame(statuses).sort_values(["endpoint", "sequence_set"]).to_csv(run_root / "task_status.csv", index=False)
            print(json.dumps(status), flush=True)

    combine(run_root, "tils_runs", "summary.csv", "tils_summary_all.csv")
    combine(run_root, "tils_runs", "per_repeat_auc.csv", "tils_per_repeat_all.csv")
    combine(run_root, "tils_runs", "oof_predictions.csv", "tils_oof_all.csv")
    combine(run_root, "tils_runs", "selected_features_by_fold.csv", "tils_selected_features_all.csv")
    combine(run_root, "recurrence_runs", "summary.csv", "recurrence_summary_all.csv")
    combine(run_root, "recurrence_runs", "per_repeat.csv", "recurrence_per_repeat_all.csv")
    combine(run_root, "recurrence_runs", "oof_risk.csv", "recurrence_oof_all.csv")
    combine(run_root, "recurrence_runs", "selected_features_by_fold.csv", "recurrence_selected_features_all.csv")
    manifest["finished"] = datetime.now().isoformat()
    manifest["tasks_ok"] = sum(s["returncode"] == 0 for s in statuses)
    manifest["tasks_failed"] = sum(s["returncode"] != 0 for s in statuses)
    (run_root / "dispatch_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (run_root / "DISPATCH_COMPLETE.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("COMPLETED=" + str(run_root), flush=True)
    if manifest["tasks_failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
