"""
One-off driver: runs main_pipeline.py --stage partition --feature-selection-method stability for a
custom, explicitly-listed set of combos (NOT the standard 7 pre-specified LI-RADS combos -- see
run_seven_combo_stability.py for those). Built for the targeted follow-up on the recurring tp/hbp
GLDM-dependence pattern seen in the LoG-removed 7-combo stability sweep: the 5-sequence
{ap,tp,pvp,hbp,dwi_b2} set plus all 10 pairwise 2-sequence combinations within it.

Per project discussion: this combo list was chosen *because* of a pattern already observed in this
same dev set (selective inference, not independent confirmation) -- report dev-set stability
frequencies here as exploratory. Whatever combo produces something should get one single-shot
gridsearch+train lock-set evaluation, not further dev-side iteration.

Usage:
    caffeinate -i python3 run_custom_combo_stability.py --dry-run
    caffeinate -i python3 run_custom_combo_stability.py
"""
import argparse
import itertools
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_seven_combo_stability import combo_base_dir, summary_results_dir, run_stage, LOG_DIR

BASE_SEQUENCES = ["ap", "tp", "pvp", "hbp", "dwi_b2"]
COMBOS = [BASE_SEQUENCES] + [list(pair) for pair in itertools.combinations(BASE_SEQUENCES, 2)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", type=int, default=2, choices=[2, 3])
    ap.add_argument("--binwidth-label", type=str, default="binwidth_perseq_nowavelet_nologsigma_5seqpairs")
    ap.add_argument("--stability-threshold", type=float, default=0.7)
    ap.add_argument("--stability-n-resamples", type=int, default=200)
    ap.add_argument("--exclude-feature-families", nargs="*", default=["wavelet", "log-sigma"])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print(f"Will run stability-selection partition (+gridsearch/train where non-empty) for "
          f"{len(COMBOS)} combos (cutoff={args.cutoff}, binwidth-label={args.binwidth_label}, "
          f"threshold={args.stability_threshold}, n_resamples={args.stability_n_resamples}, "
          f"exclude_feature_families={args.exclude_feature_families}):")
    for s in COMBOS:
        print(f"  {s}")
    if args.dry_run:
        return

    LOG_DIR.mkdir(exist_ok=True)
    results = []
    results_csv = summary_results_dir(args.cutoff, args.binwidth_label) / "stability_sweep_results.csv"
    results_csv.parent.mkdir(parents=True, exist_ok=True)

    extra = ['--feature-selection-method', 'stability',
             '--stability-threshold', str(args.stability_threshold),
             '--stability-n-resamples', str(args.stability_n_resamples),
             '--exclude-feature-families'] + args.exclude_feature_families

    for idx, sequences in enumerate(COMBOS, 1):
        seq_str = "_".join(sequences)
        print(f"\n[{idx}/{len(COMBOS)}] {sequences}", flush=True)
        row = {'sequences': seq_str}

        print("  partition (stability selection)...", flush=True)
        rc, elapsed, log = run_stage('partition', args.cutoff, sequences, args.binwidth_label, extra)
        row['partition_status'] = 'success' if rc == 0 else 'failed'
        row['partition_elapsed_sec'] = round(elapsed, 1)
        print(f"  partition {'OK' if rc == 0 else 'FAILED'} in {elapsed:.0f}s")

        cf_path = combo_base_dir(args.cutoff, sequences, args.binwidth_label) / 'clean_partition_select' / 'consensus_features.csv'
        n_features = len(pd.read_csv(cf_path)) if cf_path.exists() else None
        row['n_consensus_features'] = n_features

        # Also record the top stability frequency regardless of success/failure -- informative
        # even when nothing clears the threshold (near-miss tracking, same as the 7-combo sweep).
        stab_path = combo_base_dir(args.cutoff, sequences, args.binwidth_label) / 'clean_partition_select' / 'stability_selection.csv'
        if stab_path.exists():
            sdf = pd.read_csv(stab_path, index_col=0)
            stable = sdf[sdf['Sign_Stable']]
            if len(stable):
                row['top_stable_feature'] = stable.index[0]
                row['top_stable_freq'] = round(float(stable.iloc[0]['Selection_Frequency']), 4)

        if rc != 0:
            row['error'] = log.read_text().strip().splitlines()[-1] if log.exists() else 'unknown'
            results.append(row)
            pd.DataFrame(results).to_csv(results_csv, index=False)
            continue

        print("  gridsearch...", flush=True)
        rc_gs, t_gs, _ = run_stage('gridsearch', args.cutoff, sequences, args.binwidth_label)
        row['gridsearch_status'] = 'success' if rc_gs == 0 else 'failed'
        print(f"  gridsearch {'OK' if rc_gs == 0 else 'FAILED'} in {t_gs:.0f}s")

        if rc_gs == 0:
            print("  train...", flush=True)
            rc_tr, t_tr, _ = run_stage('train', args.cutoff, sequences, args.binwidth_label)
            row['train_status'] = 'success' if rc_tr == 0 else 'failed'
            print(f"  train {'OK' if rc_tr == 0 else 'FAILED'} in {t_tr:.0f}s")
            if rc_tr == 0:
                mc_path = combo_base_dir(args.cutoff, sequences, args.binwidth_label) / 'trained_model_outputs' / 'model_comparison.csv'
                if mc_path.exists():
                    comp = pd.read_csv(mc_path)
                    comp['_auc'] = pd.to_numeric(comp['AUC'], errors='coerce')
                    best = comp.loc[comp['_auc'].idxmax()]
                    row['lock_best_model'] = best['Model']
                    row['lock_best_auc'] = best['AUC']
                    row['lock_best_auc_ci'] = best.get('AUC 95% CI', '')

        results.append(row)
        pd.DataFrame(results).to_csv(results_csv, index=False)
        print(f"  -> {row}")

    print(f"\n{'='*100}\nCUSTOM COMBO STABILITY SWEEP COMPLETE\n{'='*100}")
    print(pd.DataFrame(results).to_string(index=False))
    print(f"\nFull results: {results_csv}")


if __name__ == "__main__":
    main()
