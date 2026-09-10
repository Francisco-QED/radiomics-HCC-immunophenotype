"""
Focused overnight v2 elastic-net run.

Runs a fuller elastic-net analysis on a small set of chosen sequence combinations. Outputs are
written to the OneDrive results tree, never to the repository.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


PYTHON = "/usr/bin/python3"
PIPELINE = str(Path(__file__).resolve().parent / "main_pipeline_v2.py")

FOCUSED_COMBOS = [
    ["ap", "pvp", "hbp", "dwi_b2", "adc"],
    ["ap", "pvp", "hbp", "dwi_b1", "adc", "t2"],
    ["ap", "tp", "hbp", "dwi_b1", "adc"],
    ["ap", "pvp", "dwi_b2", "adc"],
]


def results_root(cutoff, feature_set, analysis_label):
    cutoff_val = int(cutoff)
    cutoff_label = f"tils_{cutoff_val}_vs_0-{cutoff_val - 1}"
    feature_set_label = "clinical+radiomics" if feature_set == "both" else "radiomics"
    return (
        Path.home() / "Library" / "CloudStorage" / "OneDrive-TheMountSinaiHospital"
        / "Regeneron retrospective radiomics" / "Results" / "main_pipeline"
        / "immunophenotype" / cutoff_label / feature_set_label / "v2_stability"
        / analysis_label
    )


def run_combo(args, root, sequences):
    out_dir = root / "combinations" / "_".join(sequences) / "dev"
    log_dir = root / "summary_results" / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        PYTHON, PIPELINE,
        "--stage", "dev",
        "--cutoff", str(args.cutoff),
        "--sequences", *sequences,
        "--output-dir", str(out_dir),
        "--clinical-output-dir", str(root / "summary_results"),
        "--stability-threshold", str(args.stability_threshold),
        "--target-feature-count", str(args.target_feature_count),
        "--stability-n-resamples", str(args.stability_n_resamples),
        "--full-stability-n-resamples", str(args.full_stability_n_resamples),
        "--workflows", "elasticnet",
        "--oof-models", *args.oof_models,
        "--l1-ratios", *[str(x) for x in args.l1_ratios],
        "--selector-input", args.selector_input,
        "--selector-c-grid-size", str(args.selector_c_grid_size),
        "--feature-set", args.feature_set,
        "--exclude-feature-families", *args.exclude_feature_families,
    ]

    log_path = log_dir / f"dev_{'_'.join(sequences)}.log"
    start = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start
    log_path.write_text(proc.stdout + "\n--- STDERR ---\n" + proc.stderr)

    row = {
        "sequences": "_".join(sequences),
        "n_sequences": len(sequences),
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
        sig_path = out_dir / "signature_elasticnet_features.csv"
        if sig_path.exists():
            row["n_elasticnet_signature_features"] = len(pd.read_csv(sig_path))
    else:
        stderr = proc.stderr.strip()
        row["error"] = stderr.splitlines()[-1] if stderr else "unknown"
    return row


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutoff", type=int, default=3, choices=[2, 3])
    parser.add_argument("--feature-set", choices=["radiomics", "both"], default="radiomics")
    parser.add_argument("--exclude-feature-families", nargs="*", default=["wavelet"])
    parser.add_argument("--analysis-label", default="focused_elasticnet_thr050_k15_full")
    parser.add_argument("--stability-threshold", type=float, default=0.50)
    parser.add_argument("--target-feature-count", type=int, default=15)
    parser.add_argument("--stability-n-resamples", type=int, default=50)
    parser.add_argument("--full-stability-n-resamples", type=int, default=200)
    parser.add_argument("--l1-ratios", nargs="+", type=float, default=[0.05, 0.1, 0.2, 0.3, 0.5, 0.7])
    parser.add_argument("--selector-input", choices=["clusters", "all"], default="clusters")
    parser.add_argument("--selector-c-grid-size", type=int, default=15)
    parser.add_argument("--oof-models", nargs="+", default=["signature_logistic"])
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_arguments()
    root = results_root(args.cutoff, args.feature_set, args.analysis_label)
    summary_dir = root / "summary_results"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = summary_dir / "focused_elasticnet_v2_dev_summary.csv"

    print("Focused v2 elastic-net run will write outside the repo:")
    print(f"  {root}")
    print(f"  l1_ratios={args.l1_ratios}")
    print(f"  target_feature_count={args.target_feature_count}")
    print(f"  stability_threshold={args.stability_threshold}")
    for combo in FOCUSED_COMBOS:
        print(f"  {' '.join(combo)}")
    if args.dry_run:
        return

    rows = []
    for idx, sequences in enumerate(FOCUSED_COMBOS, start=1):
        print(f"\n[{idx}/{len(FOCUSED_COMBOS)}] {' '.join(sequences)}", flush=True)
        row = run_combo(args, root, sequences)
        rows.append(row)
        pd.DataFrame(rows).to_csv(summary_csv, index=False)
        print(f"  -> {row}", flush=True)

    df = pd.DataFrame(rows)
    print("\nFOCUSED V2 ELASTIC-NET DEV SUMMARY")
    if "best_oof_auc" in df.columns:
        print(df.sort_values("best_oof_auc", ascending=False).to_string(index=False))
    else:
        print(df.to_string(index=False))
    print(f"\nSummary CSV: {summary_csv}")


if __name__ == "__main__":
    main()
