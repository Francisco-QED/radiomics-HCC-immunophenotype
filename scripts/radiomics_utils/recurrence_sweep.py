##### ------------ Sequence combinatorial sweep for the HCC recurrence survival pipeline -------------- #####
# Runs recurrence_pipeline.py --stage partition once per sequence-subset combination, then builds
# a single comparison table across all combinations from their nested_cv_diagnostics.csv outputs.
#
# Combination rules (same as sequence_sweep.py for consistency):
#   - At least 2 contrast-enhanced sequences from {ap, pvp, tp, hbp}.
#   - Exactly one diffusion "slot": dwi_b1+adc OR dwi_b2+adc -- never dwi_b1 and dwi_b2 together.
#   - t2 optional, included only when it doesn't push the total above MAX_TOTAL_SEQS.
#   - Total sequence count capped at MAX_TOTAL_SEQS (5).
#
# No cutoff dimension — survival endpoint has no binary threshold to sweep.
# Selection criterion: OOF C-index from nested CV (lock set never touched).
#
# Usage:
#   caffeinate -i python3 recurrence_sweep.py
#   python3 recurrence_sweep.py --dry-run   # just print the combinations, don't execute

import argparse
import itertools
import re
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

PYTHON   = "/usr/bin/python3"
PIPELINE = str(Path(__file__).resolve().parent / "recurrence_pipeline.py")

CONTRAST_SEQS     = ["ap", "pvp", "tp", "hbp"]
DIFFUSION_OPTIONS = [["dwi_b1", "adc"], ["dwi_b2", "adc"]]
MIN_CONTRAST      = 2
MAX_TOTAL_SEQS    = 5

_RESULTS_DIR = (Path.home() / "Library" / "CloudStorage"
                / "OneDrive-TheMountSinaiHospital"
                / "Regeneron retrospective radiomics"
                / "Results" / "main_pipeline" / "recurrence_survival")

LOG_DIR     = Path(__file__).resolve().parent / "recurrence_sweep_logs"
RESULTS_CSV = _RESULTS_DIR / "recurrence_sweep_results.csv"


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
    seen, unique = set(), []
    for c in combos:
        key = tuple(sorted(c))
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


def run_one(sequences):
    cmd = [PYTHON, PIPELINE, "--stage", "partition", "--sequences"] + sequences
    label = "_".join(sequences)
    log_path = LOG_DIR / f"{label}.log"

    start = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start

    log_path.write_text(proc.stdout + "\n--- STDERR ---\n" + proc.stderr)

    output_dir = None
    m = re.search(r"^Output: (.+)$", proc.stdout, re.MULTILINE)
    if m:
        output_dir = m.group(1).strip()

    row = {
        "sequences":   "_".join(sequences),
        "n_sequences": len(sequences),
        "output_dir":  output_dir,
        "elapsed_sec": round(elapsed, 1),
        "status":      "success" if proc.returncode == 0 else "failed",
        "log_path":    str(log_path),
    }

    if proc.returncode != 0:
        row["error"] = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "unknown error"
        return row

    try:
        nested_cv  = pd.read_csv(Path(output_dir) / "nested_cv_diagnostics.csv")
        consensus  = pd.read_csv(Path(output_dir) / "consensus_features.csv")
        c_index    = float(nested_cv["oof_c_index"].iloc[0])
        ci_lower   = float(nested_cv["ci_lower"].iloc[0])
        ci_upper   = float(nested_cv["ci_upper"].iloc[0])
        row.update({
            "n_consensus_features": len(consensus),
            "oof_c_index":          round(c_index, 4),
            "ci_lower":             round(ci_lower, 4),
            "ci_upper":             round(ci_upper, 4),
            "significant":          bool(ci_lower > 0.5),
        })
    except Exception as exc:
        row["status"] = "parse_failed"
        row["error"]  = str(exc)

    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Print combinations and exit.")
    args = parser.parse_args()

    sequence_sets = generate_sequence_sets()
    print(f"Generated {len(sequence_sets)} sequence sets.")
    for s in sequence_sets:
        print(f"  {s}")

    if args.dry_run:
        return

    LOG_DIR.mkdir(exist_ok=True)
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    bar = tqdm(sequence_sets, desc="Sweep", unit="combo", dynamic_ncols=True)
    for sequences in bar:
        bar.set_postfix_str("_".join(sequences))
        row = run_one(sequences)
        status = row["status"]
        if status == "success":
            bar.write(f"  ✓ {row['sequences']}  C-index={row.get('oof_c_index')} "
                      f"[{row.get('ci_lower')}, {row.get('ci_upper')}]  "
                      f"features={row.get('n_consensus_features')}  {row['elapsed_sec']}s")
        else:
            bar.write(f"  ✗ {row['sequences']}  {status}  {row.get('error', '')}")
        results.append(row)
        pd.DataFrame(results).to_csv(RESULTS_CSV, index=False)

    df = pd.DataFrame(results)
    successful = df[df["status"] == "success"].sort_values("oof_c_index", ascending=False)
    print(f"\n{'='*90}")
    print(f"SWEEP COMPLETE: {len(successful)}/{len(sequence_sets)} succeeded")
    print(f"{'='*90}")
    print(successful[["sequences", "oof_c_index", "ci_lower", "ci_upper",
                       "significant", "n_consensus_features"]].head(15).to_string(index=False))
    print(f"\nFull results: {RESULTS_CSV}")


if __name__ == "__main__":
    main()
