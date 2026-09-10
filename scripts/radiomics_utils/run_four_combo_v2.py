"""
Run the v2 development-only pilot across four prespecified sequence combinations.

This script intentionally writes logs and summary CSVs to the OneDrive results tree, not to the
repository. It runs only the v2 "dev" stage so workflow comparison is based on nested/OOF AUC,
leaving lock-set evaluation for a later, explicit final run.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


PYTHON = "/usr/bin/python3"
PIPELINE = str(Path(__file__).resolve().parent / "main_pipeline_v2.py")

FOUR_COMBOS = [
    ["ap", "pvp", "hbp", "dwi_b1", "adc"],
    ["ap", "pvp", "hbp", "dwi_b1", "adc", "t2"],
    ["ap", "tp", "hbp", "dwi_b1", "adc"],
    ["ap", "pvp", "hbp", "dwi_b2", "adc"],
]


def results_root(cutoff, feature_set):
    cutoff_val = int(cutoff)
    cutoff_label = f"tils_{cutoff_val}_vs_0-{cutoff_val - 1}"
    feature_set_label = "clinical+radiomics" if feature_set == "both" else "radiomics"
    return (
        Path.home() / "Library" / "CloudStorage" / "OneDrive-TheMountSinaiHospital"
        / "Regeneron retrospective radiomics" / "Results" / "main_pipeline"
        / "immunophenotype" / cutoff_label / feature_set_label / "v2_stability"
    )


def combo_dev_dir(root, sequences):
    return root / "combinations" / "_".join(sequences) / "dev"


def run_combo(args, root, sequences):
    out_dir = combo_dev_dir(root, sequences)
    log_dir = root / "summary_results" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        PYTHON, PIPELINE,
        "--stage", "dev",
        "--cutoff", str(args.cutoff),
        "--sequences", *sequences,
        "--output-dir", str(out_dir),
        "--clinical-output-dir", str(root / "summary_results"),
        "--stability-n-resamples", str(args.stability_n_resamples),
        "--full-stability-n-resamples", str(args.full_stability_n_resamples),
        "--stability-threshold", str(args.stability_threshold),
        "--target-feature-count", str(args.target_feature_count),
        "--feature-set", args.feature_set,
        "--exclude-feature-families", *args.exclude_feature_families,
        "--l1-ratios", *[str(x) for x in args.l1_ratios],
        "--selector-input", args.selector_input,
        "--selector-c-grid-size", str(args.selector_c_grid_size),
    ]
    if args.oof_models:
        cmd += ["--oof-models", *args.oof_models]
    if args.workflows:
        cmd += ["--workflows", *args.workflows]

    log_path = log_dir / f"dev_{'_'.join(sequences)}.log"
    start = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start
    log_path.write_text(proc.stdout + "\n--- STDERR ---\n" + proc.stderr)

    row = {
        "sequences": "_".join(sequences),
        "status": "success" if proc.returncode == 0 else "failed",
        "elapsed_sec": round(elapsed, 1),
        "output_dir": str(out_dir),
        "log_path": str(log_path),
    }
    summary_path = out_dir / "nested_oof_summary.csv"
    if proc.returncode == 0 and summary_path.exists():
        summary = pd.read_csv(summary_path)
        best = summary.loc[summary["oof_auc"].idxmax()]
        row.update({
            "best_workflow": best["workflow"],
            "best_model": best["model"],
            "best_oof_auc": round(float(best["oof_auc"]), 4),
            "best_auc_ci_lower": round(float(best["auc_ci_lower"]), 4),
            "best_auc_ci_upper": round(float(best["auc_ci_upper"]), 4),
        })
        for workflow in ["lasso", "elasticnet"]:
            sig_path = out_dir / f"signature_{workflow}_features.csv"
            if sig_path.exists():
                row[f"n_{workflow}_signature_features"] = len(pd.read_csv(sig_path))
    else:
        row["error"] = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "unknown"
    return row


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutoff", type=int, default=3, choices=[2, 3])
    parser.add_argument("--feature-set", choices=["radiomics", "both"], default="radiomics")
    parser.add_argument("--exclude-feature-families", nargs="*", default=["wavelet"])
    parser.add_argument("--stability-threshold", type=float, default=0.70)
    parser.add_argument("--target-feature-count", type=int, default=15)
    parser.add_argument("--stability-n-resamples", type=int, default=50,
                        help="Nested-CV stability resamples per outer fold.")
    parser.add_argument("--full-stability-n-resamples", type=int, default=200,
                        help="Full-development stability resamples for final signature tables.")
    parser.add_argument("--workflows", nargs="+", choices=["ridge", "lasso", "elasticnet"],
                        default=["ridge", "lasso", "elasticnet"])
    parser.add_argument("--oof-models", nargs="+", default=["signature_logistic"])
    parser.add_argument("--l1-ratios", nargs="+", type=float, default=[0.5],
                        help="Elastic-net ratios for the four-combo pilot. Default is a single midpoint for speed.")
    parser.add_argument("--selector-input", choices=["clusters", "all"], default="clusters")
    parser.add_argument("--selector-c-grid-size", type=int, default=7)
    parser.add_argument("--analysis-label", type=str, default=None,
                        help="Optional subfolder under v2_stability so parameter experiments do not overwrite each other.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_arguments()
    root = results_root(args.cutoff, args.feature_set)
    if args.analysis_label:
        root = root / args.analysis_label
    summary_dir = root / "summary_results"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = summary_dir / "four_combo_v2_dev_summary.csv"

    print(f"V2 four-combo dev run will write outside the repo:")
    print(f"  {root}")
    for combo in FOUR_COMBOS:
        print(f"  {' '.join(combo)}")
    if args.dry_run:
        return

    rows = []
    for idx, sequences in enumerate(FOUR_COMBOS, start=1):
        print(f"\n[{idx}/{len(FOUR_COMBOS)}] {' '.join(sequences)}", flush=True)
        row = run_combo(args, root, sequences)
        rows.append(row)
        pd.DataFrame(rows).to_csv(summary_csv, index=False)
        print(f"  -> {row}")

    df = pd.DataFrame(rows)
    print("\nV2 FOUR-COMBO DEV SUMMARY")
    if "best_oof_auc" in df.columns:
        print(df.sort_values("best_oof_auc", ascending=False).to_string(index=False))
    else:
        print(df.to_string(index=False))
    print(f"\nSummary CSV: {summary_csv}")


if __name__ == "__main__":
    main()
