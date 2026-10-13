"""
Scaling study, not a combo hunt: runs stability selection (main_pipeline.py --stage partition
--feature-selection-method stability) for all 5 single-sequence baselines and all 10 pairwise
combinations of {ap, pvp, tp, dwi_b2, hbp} (5 + C(5,2) = 15 combos total). Goal is to characterize
how selection stability (top stability frequency, n candidate features, n consensus features)
scales with candidate-pool size/noise (n_sequences=1 vs 2, and the actual candidate feature count)
-- NOT to search for the best-performing combo. Report as a scaling relationship (sorted by
n_sequences), not a leaderboard.

This sequence set follows from the recurring tp/hbp GLDM-dependence pattern seen in the 7-combo
LoG-removed sweep -- per project discussion, that makes the dev-set stability frequencies here
exploratory/hypothesis-refining, not confirmatory (same data used to notice the pattern and to
test it). Whatever, if anything, looks informative should get a single-shot lock-set check via
gridsearch+train, not further dev-side iteration.

Usage:
    python3 run_scaling_combo_stability.py --dry-run
    caffeinate -i python3 run_scaling_combo_stability.py
"""
import argparse
import itertools
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_seven_combo_stability import combo_base_dir, summary_results_dir, run_stage, LOG_DIR

SEQUENCES = ["ap", "pvp", "tp", "dwi_b2", "hbp"]
COMBOS = [[s] for s in SEQUENCES] + [list(pair) for pair in itertools.combinations(SEQUENCES, 2)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", type=int, default=2, choices=[2, 3])
    ap.add_argument("--binwidth-label", type=str, default="binwidth_perseq_nowavelet_nologsigma_scaling")
    ap.add_argument("--stability-threshold", type=float, default=0.7)
    ap.add_argument("--stability-n-resamples", type=int, default=200)
    ap.add_argument("--exclude-feature-families", nargs="*", default=["wavelet", "log-sigma"])
    ap.add_argument("--skip-gridsearch-train", action="store_true",
        help="Skip gridsearch/train even for combos that produce consensus features -- useful for "
             "a fast first pass focused purely on the dev-side scaling relationship.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print(f"Scaling study: 5 single-sequence baselines + 10 pairs from {SEQUENCES} = "
          f"{len(COMBOS)} combos total. cutoff={args.cutoff}, binwidth-label={args.binwidth_label}, "
          f"threshold={args.stability_threshold}, n_resamples={args.stability_n_resamples}, "
          f"exclude_feature_families={args.exclude_feature_families}:")
    for s in COMBOS:
        print(f"  {s}")
    if args.dry_run:
        return

    LOG_DIR.mkdir(exist_ok=True)
    results = []
    results_csv = summary_results_dir(args.cutoff, args.binwidth_label) / "scaling_sweep_results.csv"
    results_csv.parent.mkdir(parents=True, exist_ok=True)

    extra = ['--feature-selection-method', 'stability',
             '--stability-threshold', str(args.stability_threshold),
             '--stability-n-resamples', str(args.stability_n_resamples),
             '--exclude-feature-families'] + args.exclude_feature_families

    for idx, sequences in enumerate(COMBOS, 1):
        seq_str = "_".join(sequences)
        print(f"\n[{idx}/{len(COMBOS)}] n_seq={len(sequences)}  {sequences}", flush=True)
        row = {'sequences': seq_str, 'n_sequences': len(sequences)}

        rc, elapsed, log = run_stage('partition', args.cutoff, sequences, args.binwidth_label, extra)
        row['partition_status'] = 'success' if rc == 0 else 'failed'
        row['partition_elapsed_sec'] = round(elapsed, 1)
        print(f"  partition {'OK' if rc == 0 else 'FAILED'} in {elapsed:.0f}s")

        base_dir = combo_base_dir(args.cutoff, sequences, args.binwidth_label)
        cf_path = base_dir / 'clean_partition_select' / 'consensus_features.csv'
        row['n_consensus_features'] = len(pd.read_csv(cf_path)) if cf_path.exists() else 0

        stab_path = base_dir / 'clean_partition_select' / 'stability_selection.csv'
        if stab_path.exists():
            sdf = pd.read_csv(stab_path, index_col=0)
            row['n_candidate_features_any_selection'] = len(sdf)
            stable = sdf[sdf['Sign_Stable']]
            row['n_sign_stable_ge_50pct'] = int((stable['Selection_Frequency'] >= 0.5).sum())
            if len(stable):
                row['top_stable_feature'] = stable.index[0]
                row['top_stable_freq'] = round(float(stable.iloc[0]['Selection_Frequency']), 4)

        if rc == 0 and not args.skip_gridsearch_train:
            rc_gs, t_gs, _ = run_stage('gridsearch', args.cutoff, sequences, args.binwidth_label)
            row['gridsearch_status'] = 'success' if rc_gs == 0 else 'failed'
            if rc_gs == 0:
                rc_tr, t_tr, _ = run_stage('train', args.cutoff, sequences, args.binwidth_label)
                row['train_status'] = 'success' if rc_tr == 0 else 'failed'
                if rc_tr == 0:
                    mc_path = base_dir / 'trained_model_outputs' / 'model_comparison.csv'
                    if mc_path.exists():
                        comp = pd.read_csv(mc_path)
                        comp['_auc'] = pd.to_numeric(comp['AUC'], errors='coerce')
                        best = comp.loc[comp['_auc'].idxmax()]
                        row['lock_best_model'] = best['Model']
                        row['lock_best_auc'] = best['AUC']

        results.append(row)
        pd.DataFrame(results).to_csv(results_csv, index=False)
        print(f"  -> {row}")

    df = pd.DataFrame(results).sort_values(['n_sequences', 'sequences'])
    print(f"\n{'='*100}\nSCALING STUDY COMPLETE (sorted by n_sequences, not by performance)\n{'='*100}")
    print(df.to_string(index=False))
    print(f"\nFull results: {results_csv}")


if __name__ == "__main__":
    main()
