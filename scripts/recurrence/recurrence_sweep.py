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
#
# --binwidth-label/--features-path target a different binwidth setting (must match whatever
# --binwidth-label recurrence_pipeline.py itself was given), same convention as sequence_sweep.py:
#   python3 recurrence_sweep.py --binwidth-label binwidth5 --features-path ".../W-5.csv"

import argparse
import itertools
import subprocess
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

DEFAULT_BINWIDTH_LABEL = "binwidth_perseq"  # must match recurrence_pipeline.py's own default

_RESULTS_DIR = (Path.home() / "Library" / "CloudStorage"
                / "OneDrive-TheMountSinaiHospital"
                / "Regeneron retrospective radiomics"
                / "Results" / "main_pipeline" / "recurrence_survival" / "radiomics")

LOG_DIR = Path(__file__).resolve().parent / "recurrence_sweep_logs"


# Must match recurrence_pipeline.py's own parse_arguments() path derivation exactly -- that's
# where its partition-stage output for this combo already lives.
def combo_base_dir(sequences, binwidth_label=DEFAULT_BINWIDTH_LABEL):
    return _RESULTS_DIR / binwidth_label / "combinations" / "_".join(sequences)


def summary_results_dir(binwidth_label=DEFAULT_BINWIDTH_LABEL):
    return _RESULTS_DIR / binwidth_label / "summary_results"


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


def run_one(sequences, binwidth_label=DEFAULT_BINWIDTH_LABEL, features_path=None, path_r1=None, path_r2=None,
            outcomes_path=None, event_col=None, time_col=None):
    output_dir = combo_base_dir(sequences, binwidth_label) / "partition_select"
    # --clinical-output-dir points every combo at the same per-binwidth summary_results/ folder --
    # the dev/lock split doesn't depend on which sequences are used, so this is intentionally
    # shared across combos (but never across binwidth labels, unlike before this was explicit).
    cmd = [PYTHON, PIPELINE, "--stage", "partition", "--sequences"] + sequences \
        + ["--binwidth-label", binwidth_label, "--output-dir", str(output_dir),
           "--clinical-output-dir", str(summary_results_dir(binwidth_label))]
    if features_path:
        cmd += ["--features-path", str(features_path)]
    # outcomes-path/event-col/time-col let a caller point at an alternate outcome definition (e.g.
    # 2-year landmark-censored event_2yr/time_days_2yr) without touching recurrence_pipeline.py's
    # own defaults, which stay pointed at the original any-time event/time_days columns.
    if outcomes_path:
        cmd += ["--outcomes-path", str(outcomes_path)]
    if event_col:
        cmd += ["--event-col", event_col]
    if time_col:
        cmd += ["--time-col", time_col]
    # recurrence_pipeline.py's --path-r1/--path-r2 (reader-comparison files for the ICC stability
    # filter) default to the binwidth_perseq inter-reader sheets regardless of --features-path --
    # they must be overridden explicitly here too, or the ICC filter silently compares
    # non-perseq primary features against perseq reader-comparison features.
    if path_r1:
        cmd += ["--path-r1", str(path_r1)]
    if path_r2:
        cmd += ["--path-r2", str(path_r2)]
    label = f"{binwidth_label}_{'_'.join(sequences)}"
    log_path = LOG_DIR / f"{label}.log"

    start = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start

    log_path.write_text(proc.stdout + "\n--- STDERR ---\n" + proc.stderr)

    row = {
        "sequences":   "_".join(sequences),
        "n_sequences": len(sequences),
        "output_dir":  str(output_dir),
        "elapsed_sec": round(elapsed, 1),
        "status":      "success" if proc.returncode == 0 else "failed",
        "log_path":    str(log_path),
    }

    if proc.returncode != 0:
        row["error"] = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "unknown error"
        return row

    try:
        nested_cv  = pd.read_csv(output_dir / "nested_cv_diagnostics.csv")
        consensus  = pd.read_csv(output_dir / "consensus_features.csv")
        if len(consensus) == 0:
            # recurrence_pipeline.py exits cleanly (not a nonzero returncode) when zero
            # consensus features survive -- treat that as its own distinct status rather than
            # a false "success" with an empty, unusable feature set.
            row["status"] = "no_consensus_features"
            return row
        c_index  = float(nested_cv["oof_c_index"].iloc[0])
        ci_lower = float(nested_cv["ci_lower"].iloc[0])
        ci_upper = float(nested_cv["ci_upper"].iloc[0])
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
    parser.add_argument("--binwidth-label", type=str, default=DEFAULT_BINWIDTH_LABEL,
                        help="Output folder name under recurrence_survival/ (default: binwidth_perseq).")
    parser.add_argument("--features-path", type=str, default=None,
                        help="Radiomics features CSV to pass to recurrence_pipeline.py "
                             "--features-path. Defaults to its own default when omitted.")
    parser.add_argument("--path-r1", type=str, default=None,
                        help="Reader-1 comparison CSV for the ICC stability filter, matching "
                             "--features-path's binwidth. Defaults to recurrence_pipeline.py's own "
                             "default (the binwidth_perseq reader sheet) when omitted -- must be "
                             "set explicitly for any non-perseq --features-path.")
    parser.add_argument("--path-r2", type=str, default=None,
                        help="Reader-2 comparison CSV for the ICC stability filter; see --path-r1.")
    parser.add_argument("--outcomes-path", type=str, default=None,
                        help="Outcomes xlsx to pass to recurrence_pipeline.py --outcomes-path. "
                             "Defaults to its own default (any-time event/time_days) when omitted.")
    parser.add_argument("--event-col", type=str, default=None,
                        help="Event column name in --outcomes-path, e.g. 'event_2yr'. Defaults to "
                             "recurrence_pipeline.py's own default ('event') when omitted.")
    parser.add_argument("--time-col", type=str, default=None,
                        help="Time column name in --outcomes-path, e.g. 'time_days_2yr'. Defaults to "
                             "recurrence_pipeline.py's own default ('time_days') when omitted.")
    parser.add_argument("--dry-run", action="store_true", help="Print combinations and exit.")
    args = parser.parse_args()
    binwidth_label = args.binwidth_label
    features_path = args.features_path
    path_r1 = args.path_r1
    path_r2 = args.path_r2
    outcomes_path = args.outcomes_path
    event_col = args.event_col
    time_col = args.time_col

    sequence_sets = generate_sequence_sets()
    total_runs = len(sequence_sets)
    print(f"Generated {len(sequence_sets)} sequence sets for binwidth_label={binwidth_label}.")
    for s in sequence_sets:
        print(f"  {s}")

    if args.dry_run:
        return

    LOG_DIR.mkdir(exist_ok=True)
    results_csv = summary_results_dir(binwidth_label) / "recurrence_sweep_results.csv"
    results_csv.parent.mkdir(parents=True, exist_ok=True)

    # Resume support: skip combos already marked success in a previous (interrupted) run of this
    # same binwidth_label. Mirrors sequence_sweep.py's resume logic -- only carry forward rows
    # for combos we're about to SKIP, so a retried combo's new row replaces its stale one instead
    # of duplicating it.
    done_sequences = set()
    if results_csv.exists():
        existing = pd.read_csv(results_csv).drop_duplicates(subset="sequences", keep="last")
        done_sequences = set(existing.loc[existing["status"] == "success", "sequences"])
        results = existing[existing["sequences"].isin(done_sequences)].to_dict("records")
        print(f"Resuming: {len(done_sequences)} combos already done, skipping them.")
    else:
        results = []

    bar = tqdm(sequence_sets, desc="Sweep", unit="combo", dynamic_ncols=True)
    for sequences in bar:
        sequences_str = "_".join(sequences)
        bar.set_postfix_str(sequences_str)
        if sequences_str in done_sequences:
            bar.write(f"  -- {sequences_str}  already done, skipping")
            continue
        row = run_one(sequences, binwidth_label, features_path, path_r1, path_r2,
                       outcomes_path, event_col, time_col)
        status = row["status"]
        if status == "success":
            bar.write(f"  v {row['sequences']}  C-index={row.get('oof_c_index')} "
                      f"[{row.get('ci_lower')}, {row.get('ci_upper')}]  "
                      f"features={row.get('n_consensus_features')}  {row['elapsed_sec']}s")
        else:
            bar.write(f"  x {row['sequences']}  {status}  {row.get('error', '')}")
        results.append(row)
        pd.DataFrame(results).to_csv(results_csv, index=False)

    df = pd.DataFrame(results)
    successful = df[df["status"] == "success"]
    print(f"\n{'='*90}")
    print(f"SWEEP COMPLETE: {len(successful)}/{total_runs} succeeded")
    print(f"{'='*90}")
    if len(successful) == 0:
        # "success" rows are the only ones with oof_c_index -- an all-failed/all-no-consensus run
        # has no such column at all, so sort_values("oof_c_index") would KeyError rather than just
        # printing an empty table.
        print("No combos succeeded -- nothing to rank.")
        print(df["status"].value_counts().to_string())
    else:
        successful = successful.sort_values("oof_c_index", ascending=False)
        print(successful[["sequences", "oof_c_index", "ci_lower", "ci_upper",
                           "significant", "n_consensus_features"]].head(15).to_string(index=False))
    print(f"\nFull results: {results_csv}")


if __name__ == "__main__":
    main()
