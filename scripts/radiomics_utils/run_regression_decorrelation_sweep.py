"""
Regression-pipeline counterpart of run_decorrelation_sweep.py / run_scaling_combo_stability.py:
tunes --decorrelation-threshold (0.65-0.90 in 0.05 steps) across the full 15-combo scaling set
(5 single-sequence baselines + all 10 pairs of {ap, pvp, tp, dwi_b2, hbp}) -- 90 runs total --
using tils_regression_pipeline.py (TILs_resection as continuous, not binarized).

Per project discussion: this sequence set followed from a pattern observed in the classification
pipeline on the same dev set; running it through the regression pipeline too is a legitimate,
different outcome-modeling test of the same sequences, not further dredging on the same target.

Usage:
    caffeinate -i python3 run_regression_decorrelation_sweep.py --dry-run
    caffeinate -i python3 run_regression_decorrelation_sweep.py
"""
import argparse
import itertools
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_seven_combo_stability import LOG_DIR  # reuse the shared log dir only

PYTHON = "/usr/bin/python3"
PIPELINE = str(Path(__file__).resolve().parent / "tils_regression_pipeline.py")

_RESULTS_DIR = (Path.home() / "Library" / "CloudStorage"
                / "OneDrive-TheMountSinaiHospital"
                / "Regeneron retrospective radiomics"
                / "Results" / "main_pipeline" / "immunophenotype")

SEQUENCES = ["ap", "pvp", "tp", "dwi_b2", "hbp"]
COMBOS = [[s] for s in SEQUENCES] + [list(pair) for pair in itertools.combinations(SEQUENCES, 2)]
THRESHOLDS = [0.65, 0.70, 0.75, 0.80, 0.85, 0.90]


def binwidth_label_for(threshold):
    return f"binwidth_perseq_nowavelet_nologsigma_decorr{int(round(threshold * 100)):03d}"


def combo_base_dir(sequences, binwidth_label):
    return _RESULTS_DIR / "all_tils" / "radiomics" / binwidth_label / "combinations" / "_".join(sequences)


def summary_results_dir(binwidth_label):
    return _RESULTS_DIR / "all_tils" / "radiomics" / binwidth_label / "summary_results"


def run_stage(stage, sequences, binwidth_label, extra_args=None):
    base = combo_base_dir(sequences, binwidth_label)
    if stage == 'partition':
        io_args = ['--output-dir', str(base / 'clean_partition_select'),
                   '--clinical-output-dir', str(summary_results_dir(binwidth_label))]
    elif stage == 'gridsearch':
        io_args = ['--output-dir', str(base / 'optimized_params'),
                   '--input-dir', str(base / 'clean_partition_select')]
    elif stage == 'train':
        io_args = ['--output-dir', str(base / 'trained_model_outputs'),
                   '--input-data-dir', str(base / 'clean_partition_select'),
                   '--input-params-dir', str(base / 'optimized_params')]
    else:
        io_args = []
    cmd = [PYTHON, PIPELINE, '--stage', stage, '--sequences'] + sequences + io_args
    if extra_args:
        cmd += extra_args
    label = f"{stage}_{'_'.join(sequences)}"
    log_path = LOG_DIR / f"regression_{label}.log"
    start = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start
    log_path.write_text(proc.stdout + "\n--- STDERR ---\n" + proc.stderr)
    return proc.returncode, elapsed, log_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stability-threshold", type=float, default=0.7)
    ap.add_argument("--stability-n-resamples", type=int, default=200)
    ap.add_argument("--exclude-feature-families", nargs="*", default=["wavelet", "log-sigma"])
    ap.add_argument("--skip-gridsearch-train", action="store_true")
    ap.add_argument("--thresholds", nargs="*", type=float, default=THRESHOLDS,
        help="Override the threshold list -- e.g. pass a single value for an initial scaling-only "
             "pass (matching main_pipeline.py's default 0.85) before committing to the full sweep.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    THRESHOLDS[:] = args.thresholds

    total = len(COMBOS) * len(THRESHOLDS)
    print(f"Regression decorrelation sweep: {len(COMBOS)} combos x {len(THRESHOLDS)} thresholds "
          f"= {total} runs.")
    for t in THRESHOLDS:
        for c in COMBOS:
            print(f"  decorr={t}  {c}  -> binwidth-label={binwidth_label_for(t)}")
    if args.dry_run:
        return

    LOG_DIR.mkdir(exist_ok=True)
    all_results = []
    combined_csv = Path(__file__).resolve().parent / "regression_decorrelation_sweep_results.csv"

    for threshold in THRESHOLDS:
        binwidth_label = binwidth_label_for(threshold)
        extra = ['--feature-selection-method', 'stability',
                 '--stability-threshold', str(args.stability_threshold),
                 '--stability-n-resamples', str(args.stability_n_resamples),
                 '--decorrelation-threshold', str(threshold),
                 '--exclude-feature-families'] + args.exclude_feature_families

        for sequences in COMBOS:
            seq_str = "_".join(sequences)
            print(f"\n[decorr={threshold}] {sequences}", flush=True)
            row = {'decorrelation_threshold': threshold, 'sequences': seq_str}

            rc, elapsed, log = run_stage('partition', sequences, binwidth_label, extra)
            row['partition_status'] = 'success' if rc == 0 else 'failed'
            row['partition_elapsed_sec'] = round(elapsed, 1)
            print(f"  partition {'OK' if rc == 0 else 'FAILED'} in {elapsed:.0f}s")

            base_dir = combo_base_dir(sequences, binwidth_label)
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
                best_oof = odf.loc[odf['oof_spearman_rho'].idxmax()]
                row['oof_best_model'] = best_oof['model']
                row['oof_best_rho'] = round(float(best_oof['oof_spearman_rho']), 4)
                row['oof_best_rho_ci_lower'] = round(float(best_oof['rho_ci_lower']), 4)
                row['oof_best_rho_ci_upper'] = round(float(best_oof['rho_ci_upper']), 4)

            if rc == 0 and not args.skip_gridsearch_train:
                rc_gs, t_gs, _ = run_stage('gridsearch', sequences, binwidth_label)
                row['gridsearch_status'] = 'success' if rc_gs == 0 else 'failed'
                if rc_gs == 0:
                    rc_tr, t_tr, _ = run_stage('train', sequences, binwidth_label)
                    row['train_status'] = 'success' if rc_tr == 0 else 'failed'
                    if rc_tr == 0:
                        mc_path = base_dir / 'trained_model_outputs' / 'model_comparison.csv'
                        if mc_path.exists():
                            comp = pd.read_csv(mc_path)
                            comp['_rho'] = pd.to_numeric(comp['Spearman_rho'], errors='coerce')
                            best = comp.loc[comp['_rho'].idxmax()]
                            row['lock_best_model'] = best['Model']
                            row['lock_best_rho'] = best['Spearman_rho']

            all_results.append(row)
            pd.DataFrame(all_results).to_csv(combined_csv, index=False)
            print(f"  -> {row}")

    df = pd.DataFrame(all_results).sort_values(['sequences', 'decorrelation_threshold'])
    print(f"\n{'='*100}\nREGRESSION DECORRELATION SWEEP COMPLETE\n{'='*100}")
    print(df.to_string(index=False))
    print(f"\nCombined results: {combined_csv}")


if __name__ == "__main__":
    main()
