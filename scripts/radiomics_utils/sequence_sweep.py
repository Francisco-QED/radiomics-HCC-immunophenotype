##### ------------ Sequence/cutoff combinatorial sweep for the HCC TILs radiomics pipeline -------------- #####
# Runs main_pipeline.py --stage partition once per (cutoff, sequence-subset) combination, then builds
# a single comparison table across all combinations from their nested_cv_diagnostics.csv outputs.
#
# Combination rules (agreed with the user, not arbitrary):
#   - At least 2 contrast-enhanced sequences from {ap, pvp, tp, hbp}.
#   - Exactly one diffusion "slot": dwi_b1+adc OR dwi_b2+adc -- never dwi_b1 and dwi_b2 together,
#     since the two b-values are highly correlated. ADC always rides along with whichever raw
#     b-value is chosen, since it's a derived map rather than competing raw signal.
#   - t2 optional, included only when it doesn't push the total above MAX_TOTAL_SEQS.
#   - Total sequence count capped at MAX_TOTAL_SEQS (5), matching the original baseline's size.
#
# Usage:
#   caffeinate -i python3 sequence_sweep.py
#   python3 sequence_sweep.py --dry-run   # just print the combinations, don't execute anything

import argparse
import itertools
import re
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

PYTHON = "/usr/bin/python3"
PIPELINE = str(Path(__file__).resolve().parent / "main_pipeline.py")

CONTRAST_SEQS = ["ap", "pvp", "tp", "hbp"]
DIFFUSION_OPTIONS = [["dwi_b1", "adc"], ["dwi_b2", "adc"]]
CUTOFFS = [2, 3]
MIN_CONTRAST = 2
MAX_TOTAL_SEQS = 5

LOG_DIR = Path(__file__).resolve().parent / "sweep_logs"
RESULTS_CSV = Path(__file__).resolve().parent / "sweep_results.csv"


def generate_sequence_sets():
    combos = []
    for contrast_size in range(MIN_CONTRAST, len(CONTRAST_SEQS) + 1):
        for contrast_subset in itertools.combinations(CONTRAST_SEQS, contrast_size):
            for diffusion in DIFFUSION_OPTIONS:
                base = list(contrast_subset) + diffusion
                if len(base) > MAX_TOTAL_SEQS:
                    continue
                combos.append(base)
                if len(base) + 1 <= MAX_TOTAL_SEQS:
                    combos.append(base + ["t2"])
    # de-duplicate (order-independent)
    seen = set()
    unique = []
    for c in combos:
        key = tuple(sorted(c))
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


def run_one(cutoff, sequences):
    cmd = [PYTHON, PIPELINE, "--stage", "partition", "--cutoff", str(cutoff), "--sequences"] + sequences
    label = f"cutoff{cutoff}_{'_'.join(sequences)}"
    log_path = LOG_DIR / f"{label}.log"

    start = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start

    log_path.write_text(proc.stdout + "\n--- STDERR ---\n" + proc.stderr)

    output_dir = None
    m = re.search(r"^Output directory: (.+)$", proc.stdout, re.MULTILINE)
    if m:
        output_dir = m.group(1).strip()

    row = {
        "cutoff": cutoff,
        "sequences": "_".join(sequences),
        "n_sequences": len(sequences),
        "output_dir": output_dir,
        "elapsed_sec": round(elapsed, 1),
        "status": "success" if proc.returncode == 0 else "failed",
        "log_path": str(log_path),
    }

    if proc.returncode != 0:
        row["error"] = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "unknown error"
        return row

    try:
        nested_cv = pd.read_csv(Path(output_dir) / "nested_cv_diagnostics.csv")
        consensus = pd.read_csv(Path(output_dir) / "consensus_features.csv")
        best = nested_cv.loc[nested_cv["oof_auc"].idxmax()]
        row.update({
            "n_consensus_features": len(consensus),
            "best_model": best["model"],
            "best_oof_auc": round(best["oof_auc"], 4),
            "best_auc_ci_lower": round(best["auc_ci_lower"], 4),
            "best_auc_ci_upper": round(best["auc_ci_upper"], 4),
            "significant": bool(best["auc_ci_lower"] > 0.5),
            "mean_oof_auc_all_models": round(nested_cv["oof_auc"].mean(), 4),
        })
    except Exception as exc:
        row["status"] = "parse_failed"
        row["error"] = str(exc)

    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Print combinations and exit.")
    args = parser.parse_args()

    sequence_sets = generate_sequence_sets()
    total_runs = len(sequence_sets) * len(CUTOFFS)
    print(f"Generated {len(sequence_sets)} sequence sets x {len(CUTOFFS)} cutoffs = {total_runs} runs.")
    for s in sequence_sets:
        print(f"  {s}")

    if args.dry_run:
        return

    LOG_DIR.mkdir(exist_ok=True)
    results = []
    run_idx = 0
    for cutoff in CUTOFFS:
        for sequences in sequence_sets:
            run_idx += 1
            print(f"\n[{run_idx}/{total_runs}] cutoff={cutoff} sequences={sequences}", flush=True)
            row = run_one(cutoff, sequences)
            print(f"  -> {row['status']} in {row['elapsed_sec']}s"
                  + (f", best={row.get('best_model')} AUC={row.get('best_oof_auc')}"
                     if row["status"] == "success" else f", error={row.get('error')}"))
            results.append(row)
            # Write incrementally so partial progress survives a crash mid-sweep.
            pd.DataFrame(results).to_csv(RESULTS_CSV, index=False)

    df = pd.DataFrame(results)
    successful = df[df["status"] == "success"].sort_values("best_oof_auc", ascending=False)
    print(f"\n{'='*100}")
    print(f"SWEEP COMPLETE: {len(successful)}/{total_runs} succeeded")
    print(f"{'='*100}")
    print(successful[["cutoff", "sequences", "best_model", "best_oof_auc",
                       "best_auc_ci_lower", "best_auc_ci_upper", "significant",
                       "n_consensus_features"]].head(15).to_string(index=False))
    print(f"\nFull results: {RESULTS_CSV}")


if __name__ == "__main__":
    main()
