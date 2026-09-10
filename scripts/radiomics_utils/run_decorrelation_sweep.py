"""
Tunes --decorrelation-threshold (the Spearman-correlation cutoff for dropping near-duplicate
features, previously hardcoded at 0.85) across the 4 combos that cleared the standard stability
threshold in the scaling study: hbp, pvp_hbp, tp_dwi_b2, tp_hbp. Each threshold value gets its own
binwidth-label root (thresholds change what survives at every stage, including the nested-CV OOF
diagnostic, so results must not share a folder/cache with other threshold values).

Motivation: lower thresholds prune near-duplicate columns (e.g. an original-filter feature and its
own correlated cousins) more aggressively, which should reduce the vote-splitting across stability-
selection resamples discussed in project history -- directly testing that theory, not hunting for
a better-performing combo. 0.85 is the existing baseline (already run in the scaling study, not
rerun here).

Usage:
    caffeinate -i python3 run_decorrelation_sweep.py --dry-run
    caffeinate -i python3 run_decorrelation_sweep.py
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_seven_combo_stability import combo_base_dir, summary_results_dir, run_stage, LOG_DIR

COMBOS = [["hbp"], ["pvp", "hbp"], ["tp", "dwi_b2"], ["tp", "hbp"]]
THRESHOLDS = [0.70, 0.75, 0.80, 0.90, 0.95]  # 0.85 is the existing baseline, not rerun


def binwidth_label_for(threshold):
    return f"binwidth_perseq_nowavelet_nologsigma_decorr{int(round(threshold * 100)):03d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", type=int, default=2, choices=[2, 3])
    ap.add_argument("--stability-threshold", type=float, default=0.7)
    ap.add_argument("--stability-n-resamples", type=int, default=200)
    ap.add_argument("--exclude-feature-families", nargs="*", default=["wavelet", "log-sigma"])
    ap.add_argument("--skip-gridsearch-train", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    total = len(COMBOS) * len(THRESHOLDS)
    print(f"Decorrelation sweep: {len(COMBOS)} combos x {len(THRESHOLDS)} thresholds = {total} runs "
          f"(0.85 baseline already exists, not rerun).")
    for t in THRESHOLDS:
        for c in COMBOS:
            print(f"  decorr={t}  {c}  -> binwidth-label={binwidth_label_for(t)}")
    if args.dry_run:
        return

    LOG_DIR.mkdir(exist_ok=True)
    all_results = []
    combined_csv = Path(__file__).resolve().parent / "decorrelation_sweep_results.csv"

    for threshold in THRESHOLDS:
        binwidth_label = binwidth_label_for(threshold)
        results_csv = summary_results_dir(args.cutoff, binwidth_label) / "decorrelation_sweep_results.csv"
        results_csv.parent.mkdir(parents=True, exist_ok=True)

        extra = ['--feature-selection-method', 'stability',
                 '--stability-threshold', str(args.stability_threshold),
                 '--stability-n-resamples', str(args.stability_n_resamples),
                 '--decorrelation-threshold', str(threshold),
                 '--exclude-feature-families'] + args.exclude_feature_families

        for sequences in COMBOS:
            seq_str = "_".join(sequences)
            print(f"\n[decorr={threshold}] {sequences}", flush=True)
            row = {'decorrelation_threshold': threshold, 'sequences': seq_str}

            rc, elapsed, log = run_stage('partition', args.cutoff, sequences, binwidth_label, extra)
            row['partition_status'] = 'success' if rc == 0 else 'failed'
            row['partition_elapsed_sec'] = round(elapsed, 1)
            print(f"  partition {'OK' if rc == 0 else 'FAILED'} in {elapsed:.0f}s")

            base_dir = combo_base_dir(args.cutoff, sequences, binwidth_label)
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

            oof_path = base_dir / 'clean_partition_select' / 'nested_cv_diagnostics.csv'
            if oof_path.exists():
                odf = pd.read_csv(oof_path)
                best_oof = odf.loc[odf['oof_auc'].idxmax()]
                row['oof_best_model'] = best_oof['model']
                row['oof_best_auc'] = round(float(best_oof['oof_auc']), 4)
                row['oof_best_auc_ci_lower'] = round(float(best_oof['auc_ci_lower']), 4)
                row['oof_best_auc_ci_upper'] = round(float(best_oof['auc_ci_upper']), 4)

            if rc == 0 and not args.skip_gridsearch_train:
                rc_gs, t_gs, _ = run_stage('gridsearch', args.cutoff, sequences, binwidth_label)
                row['gridsearch_status'] = 'success' if rc_gs == 0 else 'failed'
                if rc_gs == 0:
                    rc_tr, t_tr, _ = run_stage('train', args.cutoff, sequences, binwidth_label)
                    row['train_status'] = 'success' if rc_tr == 0 else 'failed'
                    if rc_tr == 0:
                        mc_path = base_dir / 'trained_model_outputs' / 'model_comparison.csv'
                        if mc_path.exists():
                            comp = pd.read_csv(mc_path)
                            comp['_auc'] = pd.to_numeric(comp['AUC'], errors='coerce')
                            best = comp.loc[comp['_auc'].idxmax()]
                            row['lock_best_model'] = best['Model']
                            row['lock_best_auc'] = best['AUC']

            all_results.append(row)
            pd.DataFrame(all_results).to_csv(combined_csv, index=False)
            pd.DataFrame([r for r in all_results if r['decorrelation_threshold'] == threshold]).to_csv(results_csv, index=False)
            print(f"  -> {row}")

    df = pd.DataFrame(all_results).sort_values(['sequences', 'decorrelation_threshold'])
    print(f"\n{'='*100}\nDECORRELATION SWEEP COMPLETE\n{'='*100}")
    print(df.to_string(index=False))
    print(f"\nCombined results: {combined_csv}")


if __name__ == "__main__":
    main()
