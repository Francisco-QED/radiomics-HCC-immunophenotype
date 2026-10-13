"""
Runs the partition stage (main_pipeline.py --stage partition) for the 7 pre-specified,
LI-RADS-motivated sequence combinations (see project memory / sequence_sweep_findings_report.docx)
-- NOT sequence_sweep.py's own exhaustive generate_sequence_sets(), which predates that redesign
and would produce a different, much larger set (and can't produce combo 5's 6-sequence set at all,
since it's capped at MAX_TOTAL_SEQS=5).

Reuses sequence_sweep.py's run_one()/combo_base_dir()/summary_results_dir() as-is for output-path
conventions and incremental sweep_results.csv bookkeeping -- only the combo list is overridden here.

Usage:
    caffeinate -i python3 run_seven_combo_partition.py --cutoff 2 --binwidth-label binwidth_perseq_nowavelet2
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sequence_sweep import run_one, summary_results_dir, LOG_DIR

SEVEN_COMBOS = [
    ["ap", "pvp", "hbp", "dwi_b1", "adc"],              # 1. Base
    ["ap", "pvp", "hbp", "dwi_b1", "adc", "t2"],         # 2. Base + T2
    ["ap", "tp", "hbp", "dwi_b1", "adc"],                # 3. Base, pvp->tp
    ["ap", "pvp", "tp", "dwi_b1", "adc"],                # 4. Base, hbp->tp
    ["pre", "ap", "pvp", "hbp", "dwi_b1", "adc"],        # 5. Base + pre (6 sequences)
    ["ap", "pvp", "hbp", "dwi_b2", "adc"],               # 6. Base, b1->b2
    ["pre", "ap", "pvp", "tp", "hbp"],                   # 7. T1-only
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", type=int, required=True, choices=[2, 3])
    ap.add_argument("--binwidth-label", type=str, required=True)
    args = ap.parse_args()

    results_csv = summary_results_dir(args.cutoff, args.binwidth_label) / "sweep_results.csv"
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)

    done_sequences = set()
    if results_csv.exists():
        existing = pd.read_csv(results_csv).drop_duplicates(subset="sequences", keep="last")
        done_sequences = set(existing.loc[existing["status"] == "success", "sequences"])
        results = existing[existing["sequences"].isin(done_sequences)].to_dict("records")
        print(f"Resuming: {len(done_sequences)} combos already done, skipping them.")
    else:
        results = []

    total = len(SEVEN_COMBOS)
    for idx, sequences in enumerate(SEVEN_COMBOS, 1):
        sequences_str = "_".join(sequences)
        if sequences_str in done_sequences:
            print(f"\n[{idx}/{total}] {sequences} -- already done, skipping")
            continue
        print(f"\n[{idx}/{total}] cutoff={args.cutoff} sequences={sequences}", flush=True)
        row = run_one(args.cutoff, sequences, args.binwidth_label)
        print(f"  -> {row['status']} in {row['elapsed_sec']}s"
              + (f", best={row.get('best_model')} AUC={row.get('best_oof_auc')}"
                 if row["status"] == "success" else f", error={row.get('error')}"))
        results.append(row)
        pd.DataFrame(results).to_csv(results_csv, index=False)

    df = pd.DataFrame(results)
    successful = df[df["status"] == "success"]
    print(f"\n{'='*100}\nPARTITION COMPLETE: {len(successful)}/{total} succeeded\n{'='*100}")
    print(f"Full results: {results_csv}")


if __name__ == "__main__":
    main()
