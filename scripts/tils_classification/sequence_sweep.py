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
# Usage (run once per TILs cutoff -- the sweep no longer loops over cutoffs internally, so its
# sweep_results.csv lands in that cutoff's own <binwidth-label>/summary_results/ folder):
#   caffeinate -i python3 sequence_sweep.py --cutoff 2
#   caffeinate -i python3 sequence_sweep.py --cutoff 3
#   python3 sequence_sweep.py --cutoff 2 --dry-run   # just print the combinations, don't execute
#
# --binwidth-label/--features-path target a different binwidth setting (e.g. the fixed binWidth=5
# extraction instead of the default per-sequence one) without editing this file:
#   python3 sequence_sweep.py --cutoff 2 --binwidth-label binwidth5 \
#       --features-path "/Volumes/T7 Shield/Restrepo/HCC_Retrospective_radiomics/Results/Radiomics_sheets/SA1_2/R-SA1_2_I-sitkBSpline_S-1mm_W-5.csv"

import argparse
import itertools
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

PYTHON = "/usr/bin/python3"
PIPELINE = str(Path(__file__).resolve().parent / "main_pipeline.py")

CONTRAST_SEQS = ["ap", "pvp", "tp", "hbp"]
DIFFUSION_OPTIONS = [["dwi_b1", "adc"], ["dwi_b2", "adc"]]
CUTOFFS = [2, 3]  # valid --cutoff choices; the sweep runs one cutoff per invocation
MIN_CONTRAST = 2
MAX_TOTAL_SEQS = 5

_RESULTS_DIR = (Path.home() / "Library" / "CloudStorage"
                / "OneDrive-TheMountSinaiHospital"
                / "Regeneron retrospective radiomics"
                / "Results" / "main_pipeline" / "immunophenotype")

LOG_DIR = Path(__file__).resolve().parent / "sweep_logs"

# Default binwidth setting -- matches main_pipeline.py's own --features-path default (the
# per-sequence-binwidth extraction). Passing --binwidth-label/--features-path targets any other
# binwidth extraction (e.g. the fixed binWidth=5 sheet) without editing this file.
DEFAULT_BINWIDTH_LABEL = "binwidth_perseq"


# main_pipeline.py's own default --output-dir no longer has a per-sequence-combo segment (it
# routes single fixed-sequence runs to .../radiomics or .../clinical+radiomics), so the sweep must
# pass --output-dir explicitly per combo or every combo would collide into the same folder.
def combo_base_dir(cutoff, sequences, binwidth_label=DEFAULT_BINWIDTH_LABEL):
    cutoff_val = int(cutoff)
    neg_max = cutoff_val - 1
    cutoff_label = f"tils_{cutoff_val}_vs_0-{neg_max}"
    return _RESULTS_DIR / cutoff_label / "radiomics" / binwidth_label / "combinations" / "_".join(sequences)


# Shared per-cutoff location for artifacts that don't depend on which sequences are used (dev/lock
# split, clinical_metadata_dev/lock.csv, dev_test_comparison.xlsx) -- computed once per cutoff
# instead of duplicated into every combo's own folder. Sibling of "combinations/" under binwidth_label.
def summary_results_dir(cutoff, binwidth_label=DEFAULT_BINWIDTH_LABEL):
    cutoff_val = int(cutoff)
    neg_max = cutoff_val - 1
    cutoff_label = f"tils_{cutoff_val}_vs_0-{neg_max}"
    return _RESULTS_DIR / cutoff_label / "radiomics" / binwidth_label / "summary_results"


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


def run_one(cutoff, sequences, binwidth_label=DEFAULT_BINWIDTH_LABEL, features_path=None, path_r1=None, path_r2=None):
    output_dir = combo_base_dir(cutoff, sequences, binwidth_label) / "clean_partition_select"
    cmd = [PYTHON, PIPELINE, "--stage", "partition", "--cutoff", str(cutoff), "--sequences"] + sequences \
        + ["--output-dir", str(output_dir), "--clinical-output-dir", str(summary_results_dir(cutoff, binwidth_label))]
    if features_path:
        cmd += ["--features-path", str(features_path)]
    # main_pipeline.py's --path-r1/--path-r2 (reader-comparison files for the ICC stability filter)
    # default to the binwidth_perseq inter-reader sheets regardless of --features-path -- must be
    # overridden explicitly here too, or interobserver_stability() filters this binwidth's primary
    # features using binwidth_perseq's reader1-vs-reader2 reproducibility instead of this binwidth's.
    if path_r1:
        cmd += ["--path-r1", str(path_r1)]
    if path_r2:
        cmd += ["--path-r2", str(path_r2)]
    label = f"{binwidth_label}_cutoff{cutoff}_{'_'.join(sequences)}"
    log_path = LOG_DIR / f"{label}.log"

    start = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start

    log_path.write_text(proc.stdout + "\n--- STDERR ---\n" + proc.stderr)

    row = {
        "cutoff": cutoff,
        "sequences": "_".join(sequences),
        "n_sequences": len(sequences),
        "output_dir": str(output_dir),
        "elapsed_sec": round(elapsed, 1),
        "status": "success" if proc.returncode == 0 else "failed",
        "log_path": str(log_path),
    }

    if proc.returncode != 0:
        row["error"] = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "unknown error"
        return row

    try:
        nested_cv = pd.read_csv(output_dir / "nested_cv_diagnostics.csv")
        consensus = pd.read_csv(output_dir / "consensus_features.csv")
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
    parser.add_argument("--cutoff", type=int, required=True, choices=CUTOFFS,
                        help="TILs cutoff to sweep. Run once per cutoff.")
    parser.add_argument("--binwidth-label", type=str, default=DEFAULT_BINWIDTH_LABEL,
                        help="Output folder name under radiomics/ (default: binwidth_perseq).")
    parser.add_argument("--features-path", type=str, default=None,
                        help="Radiomics features CSV to pass to main_pipeline.py --features-path. "
                             "Defaults to main_pipeline.py's own default (the per-sequence-binwidth "
                             "extraction) when omitted.")
    parser.add_argument("--path-r1", type=str, default=None,
                        help="Reader-1 comparison CSV for the ICC stability filter, matching "
                             "--features-path's binwidth. Defaults to main_pipeline.py's own default "
                             "(the binwidth_perseq reader sheet) when omitted -- must be set "
                             "explicitly for any non-perseq --features-path.")
    parser.add_argument("--path-r2", type=str, default=None,
                        help="Reader-2 comparison CSV for the ICC stability filter; see --path-r1.")
    parser.add_argument("--dry-run", action="store_true", help="Print combinations and exit.")
    args = parser.parse_args()
    cutoff = args.cutoff
    binwidth_label = args.binwidth_label
    features_path = args.features_path
    path_r1 = args.path_r1
    path_r2 = args.path_r2

    sequence_sets = generate_sequence_sets()
    total_runs = len(sequence_sets)
    print(f"Generated {len(sequence_sets)} sequence sets for cutoff {cutoff} = {total_runs} runs.")
    for s in sequence_sets:
        print(f"  {s}")

    if args.dry_run:
        return

    # Per-cutoff output: sits next to this cutoff's own combinations, not at the shared root.
    results_csv = summary_results_dir(cutoff, binwidth_label) / "sweep_results.csv"
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)

    # Resume support: skip combos already marked success in a previous (interrupted) run of this
    # same cutoff, so a crash/stop mid-sweep doesn't force redoing everything from combo 1. Mirrors
    # gridsearch_train_sweep.py's resume logic.
    done_sequences = set()
    if results_csv.exists():
        existing = pd.read_csv(results_csv).drop_duplicates(subset="sequences", keep="last")
        done_sequences = set(existing.loc[existing["status"] == "success", "sequences"])
        # Only carry forward rows for combos we're about to SKIP. A combo that will be retried
        # this run must not keep its old (failed/incomplete) row, or the retry's new row becomes
        # a duplicate instead of a replacement -- this is exactly how an orphaned prior process
        # racing this script corrupted cutoff 3's sweep_results.csv (duplicate rows, and combos
        # silently dropped when the two processes' independently-tracked result lists diverged).
        results = existing[existing["sequences"].isin(done_sequences)].to_dict("records")
        print(f"Resuming: {len(done_sequences)} combos already done, skipping them.")
    else:
        results = []

    for run_idx, sequences in enumerate(sequence_sets, 1):
        sequences_str = "_".join(sequences)
        if sequences_str in done_sequences:
            print(f"\n[{run_idx}/{total_runs}] cutoff={cutoff} sequences={sequences} -- already done, skipping")
            continue
        print(f"\n[{run_idx}/{total_runs}] cutoff={cutoff} sequences={sequences}", flush=True)
        row = run_one(cutoff, sequences, binwidth_label, features_path, path_r1, path_r2)
        print(f"  -> {row['status']} in {row['elapsed_sec']}s"
              + (f", best={row.get('best_model')} AUC={row.get('best_oof_auc')}"
                 if row["status"] == "success" else f", error={row.get('error')}"))
        results.append(row)
        # Write incrementally so partial progress survives a crash mid-sweep.
        pd.DataFrame(results).to_csv(results_csv, index=False)

    df = pd.DataFrame(results)
    successful = df[df["status"] == "success"]
    print(f"\n{'='*100}")
    print(f"SWEEP COMPLETE (cutoff {cutoff}): {len(successful)}/{total_runs} succeeded")
    print(f"{'='*100}")
    if len(successful) == 0:
        print("No combos succeeded -- nothing to rank.")
        print(df["status"].value_counts().to_string())
    else:
        successful = successful.sort_values("best_oof_auc", ascending=False)
        print(successful[["cutoff", "sequences", "best_model", "best_oof_auc",
                           "best_auc_ci_lower", "best_auc_ci_upper", "significant",
                           "n_consensus_features"]].head(15).to_string(index=False))
    print(f"\nFull results: {results_csv}")


if __name__ == "__main__":
    main()
