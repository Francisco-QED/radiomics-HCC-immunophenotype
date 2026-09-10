"""
Runs main_pipeline.py's --stage partition with --feature-selection-method stability (Meinshausen-
Buhlmann stability selection, see main_pipeline.py's run_stability_selection) for all 7 pre-specified
LI-RADS combos, then gridsearch+train for whichever combos produce a non-empty consensus set.

Usage:
    caffeinate -i python3 run_seven_combo_stability.py --cutoff 2 --binwidth-label binwidth_perseq_nowavelet2
    python3 run_seven_combo_stability.py --cutoff 2 --binwidth-label binwidth_perseq_nowavelet2 --dry-run
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

PYTHON = "/usr/bin/python3"
PIPELINE = str(Path(__file__).resolve().parent / "main_pipeline.py")

_RESULTS_DIR = (Path.home() / "Library" / "CloudStorage"
                / "OneDrive-TheMountSinaiHospital"
                / "Regeneron retrospective radiomics"
                / "Results" / "main_pipeline" / "immunophenotype")

LOG_DIR = Path(__file__).resolve().parent / "stability_sweep_logs"

SEVEN_COMBOS = [
    ["ap", "pvp", "hbp", "dwi_b1", "adc"],
    ["ap", "pvp", "hbp", "dwi_b1", "adc", "t2"],
    ["ap", "tp", "hbp", "dwi_b1", "adc"],
    ["ap", "pvp", "tp", "dwi_b1", "adc"],
    ["pre", "ap", "pvp", "hbp", "dwi_b1", "adc"],
    ["ap", "pvp", "hbp", "dwi_b2", "adc"],
    ["pre", "ap", "pvp", "tp", "hbp"],
]


def combo_base_dir(cutoff, sequences, binwidth_label):
    cutoff_label = f"tils_{cutoff}_vs_0-{cutoff-1}"
    return _RESULTS_DIR / cutoff_label / "radiomics" / binwidth_label / "combinations" / "_".join(sequences)


def summary_results_dir(cutoff, binwidth_label):
    cutoff_label = f"tils_{cutoff}_vs_0-{cutoff-1}"
    return _RESULTS_DIR / cutoff_label / "radiomics" / binwidth_label / "summary_results"


def run_stage(stage, cutoff, sequences, binwidth_label, extra_args=None):
    base = combo_base_dir(cutoff, sequences, binwidth_label)
    if stage == 'partition':
        io_args = ['--output-dir', str(base / 'clean_partition_select'),
                   '--clinical-output-dir', str(summary_results_dir(cutoff, binwidth_label))]
    elif stage == 'gridsearch':
        io_args = ['--output-dir', str(base / 'optimized_params'),
                   '--input-dir', str(base / 'clean_partition_select')]
    elif stage == 'train':
        io_args = ['--output-dir', str(base / 'trained_model_outputs'),
                   '--input-data-dir', str(base / 'clean_partition_select'),
                   '--input-params-dir', str(base / 'optimized_params')]
    else:
        io_args = []
    cmd = [PYTHON, PIPELINE, '--stage', stage, '--cutoff', str(cutoff), '--sequences'] + sequences + io_args
    if extra_args:
        cmd += extra_args
    label = f"{stage}_{'_'.join(sequences)}"
    log_path = LOG_DIR / f"{label}.log"
    start = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start
    log_path.write_text(proc.stdout + "\n--- STDERR ---\n" + proc.stderr)
    return proc.returncode, elapsed, log_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", type=int, default=2, choices=[2, 3])
    ap.add_argument("--binwidth-label", type=str, default="binwidth_perseq_nowavelet2")
    ap.add_argument("--stability-threshold", type=float, default=0.7)
    ap.add_argument("--stability-n-resamples", type=int, default=200)
    ap.add_argument("--exclude-feature-families", nargs="*", default=["wavelet"],
        help="Passed through to main_pipeline.py's --stage partition. Default matches "
             "main_pipeline.py's own default (wavelet only). Pass e.g. 'wavelet log-sigma' to also "
             "drop LoG-filtered features and shrink the candidate pool further.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print(f"Will run stability-selection partition (+gridsearch/train where non-empty) for "
          f"{len(SEVEN_COMBOS)} combos (cutoff={args.cutoff}, binwidth-label={args.binwidth_label}, "
          f"threshold={args.stability_threshold}, n_resamples={args.stability_n_resamples}, "
          f"exclude_feature_families={args.exclude_feature_families}):")
    for s in SEVEN_COMBOS:
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

    for idx, sequences in enumerate(SEVEN_COMBOS, 1):
        seq_str = "_".join(sequences)
        print(f"\n[{idx}/{len(SEVEN_COMBOS)}] {sequences}", flush=True)
        row = {'sequences': seq_str}

        print("  partition (stability selection)...", flush=True)
        rc, elapsed, log = run_stage('partition', args.cutoff, sequences, args.binwidth_label, extra)
        row['partition_status'] = 'success' if rc == 0 else 'failed'
        row['partition_elapsed_sec'] = round(elapsed, 1)
        print(f"  partition {'OK' if rc == 0 else 'FAILED'} in {elapsed:.0f}s")
        if rc != 0:
            row['error'] = log.read_text().strip().splitlines()[-1] if log.exists() else 'unknown'
            results.append(row)
            pd.DataFrame(results).to_csv(results_csv, index=False)
            continue

        n_features = None
        cf_path = combo_base_dir(args.cutoff, sequences, args.binwidth_label) / 'clean_partition_select' / 'consensus_features.csv'
        if cf_path.exists():
            n_features = len(pd.read_csv(cf_path))
        row['n_consensus_features'] = n_features

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

    print(f"\n{'='*100}\nSTABILITY SWEEP COMPLETE\n{'='*100}")
    print(pd.DataFrame(results).to_string(index=False))
    print(f"\nFull results: {results_csv}")


if __name__ == "__main__":
    main()
