"""
Exploratory follow-up: relax the cluster-based stability-selection threshold, per combo, to reach
~15 sign-stable features instead of the standard 0.7-threshold's 0/1-feature result, for the
wavelet+log-sigma-restored 7-combo sweep (binwidth_perseq_allfamilies_clustered). Reuses each
combo's already-cached stability_selection_clustered.csv (same --output-dir as the original run) so
only the ICC/nested-CV-diagnostic step reruns, not the 200-resample stability computation itself.

NOTE: this OVERWRITES each combo's clean_partition_select/consensus_features.csv and downstream
gridsearch/train artifacts in place -- the standard-threshold (0.7) results have already been
recorded in project memory before running this, per established practice for this kind of
exploratory widening (see project memory's "10-feature exploration" precedent).

These are EXPLORATORY results at very low selection frequency (6.5%-13.5%, i.e. selected in only
~13-27 of 200 resamples) -- report as such, not as validated stability results.

Usage:
    caffeinate -i python3 run_seven_combo_top15.py
"""
import subprocess
import time
from pathlib import Path

import pandas as pd

PYTHON = "/usr/bin/python3"
PIPELINE = str(Path(__file__).resolve().parent / "main_pipeline.py")

_RESULTS_DIR = (Path.home() / "Library" / "CloudStorage"
                / "OneDrive-TheMountSinaiHospital"
                / "Regeneron retrospective radiomics"
                / "Results" / "main_pipeline" / "immunophenotype")

BINWIDTH_LABEL = "binwidth_perseq_allfamilies_clustered"
CUTOFF = 2
LOG_DIR = Path(__file__).resolve().parent / "stability_sweep_logs"

# Per-combo threshold = the 15th-highest sign-stable cluster's Selection_Frequency (from the
# already-cached stability_selection_clustered.csv), minus a small epsilon for float-inclusive head(15).
COMBO_THRESHOLDS = {
    ("ap", "pvp", "hbp", "dwi_b1", "adc"): 0.09,
    ("ap", "pvp", "hbp", "dwi_b1", "adc", "t2"): 0.13,
    ("ap", "tp", "hbp", "dwi_b1", "adc"): 0.06,
    ("ap", "pvp", "tp", "dwi_b1", "adc"): 0.10,
    ("pre", "ap", "pvp", "hbp", "dwi_b1", "adc"): 0.06,
    ("ap", "pvp", "hbp", "dwi_b2", "adc"): 0.07,
    ("pre", "ap", "pvp", "tp", "hbp"): 0.10,
}

TARGET_FEATURE_COUNT = 15


def combo_base_dir(sequences):
    cutoff_label = f"tils_{CUTOFF}_vs_0-{CUTOFF-1}"
    return _RESULTS_DIR / cutoff_label / "radiomics" / BINWIDTH_LABEL / "combinations" / "_".join(sequences)


def summary_results_dir():
    cutoff_label = f"tils_{CUTOFF}_vs_0-{CUTOFF-1}"
    return _RESULTS_DIR / cutoff_label / "radiomics" / BINWIDTH_LABEL / "summary_results"


def run_stage(stage, sequences, extra_args=None):
    base = combo_base_dir(sequences)
    if stage == 'partition':
        io_args = ['--output-dir', str(base / 'clean_partition_select'),
                   '--clinical-output-dir', str(summary_results_dir())]
    elif stage == 'gridsearch':
        io_args = ['--output-dir', str(base / 'optimized_params'),
                   '--input-dir', str(base / 'clean_partition_select')]
    elif stage == 'train':
        io_args = ['--output-dir', str(base / 'trained_model_outputs'),
                   '--input-data-dir', str(base / 'clean_partition_select'),
                   '--input-params-dir', str(base / 'optimized_params')]
    else:
        io_args = []
    cmd = [PYTHON, PIPELINE, '--stage', stage, '--cutoff', str(CUTOFF), '--sequences'] + list(sequences) + io_args
    if extra_args:
        cmd += extra_args
    label = f"top15_{stage}_{'_'.join(sequences)}"
    log_path = LOG_DIR / f"{label}.log"
    start = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start
    log_path.write_text(proc.stdout + "\n--- STDERR ---\n" + proc.stderr)
    return proc.returncode, elapsed, log_path


def main():
    LOG_DIR.mkdir(exist_ok=True)
    results = []
    results_csv = summary_results_dir() / "top15_sweep_results.csv"
    results_csv.parent.mkdir(parents=True, exist_ok=True)

    for idx, (sequences, threshold) in enumerate(COMBO_THRESHOLDS.items(), 1):
        seq_str = "_".join(sequences)
        print(f"\n[{idx}/{len(COMBO_THRESHOLDS)}] {sequences} (threshold={threshold})", flush=True)
        row = {'sequences': seq_str, 'stability_threshold': threshold}

        extra = ['--feature-selection-method', 'stability',
                 '--stability-threshold', str(threshold),
                 '--stability-n-resamples', '200',
                 '--target-feature-count', str(TARGET_FEATURE_COUNT),
                 '--exclude-feature-families']

        print("  partition (relaxed threshold, reusing cached resamples)...", flush=True)
        rc, elapsed, log = run_stage('partition', sequences, extra)
        row['partition_status'] = 'success' if rc == 0 else 'failed'
        row['partition_elapsed_sec'] = round(elapsed, 1)
        print(f"  partition {'OK' if rc == 0 else 'FAILED'} in {elapsed:.0f}s")
        if rc != 0:
            row['error'] = log.read_text().strip().splitlines()[-1] if log.exists() else 'unknown'
            results.append(row)
            pd.DataFrame(results).to_csv(results_csv, index=False)
            continue

        cf_path = combo_base_dir(sequences) / 'clean_partition_select' / 'consensus_features.csv'
        n_features = len(pd.read_csv(cf_path)) if cf_path.exists() else None
        row['n_consensus_features'] = n_features

        print("  gridsearch...", flush=True)
        rc_gs, t_gs, _ = run_stage('gridsearch', sequences)
        row['gridsearch_status'] = 'success' if rc_gs == 0 else 'failed'
        print(f"  gridsearch {'OK' if rc_gs == 0 else 'FAILED'} in {t_gs:.0f}s")

        if rc_gs == 0:
            print("  train...", flush=True)
            rc_tr, t_tr, _ = run_stage('train', sequences)
            row['train_status'] = 'success' if rc_tr == 0 else 'failed'
            print(f"  train {'OK' if rc_tr == 0 else 'FAILED'} in {t_tr:.0f}s")
            if rc_tr == 0:
                mc_path = combo_base_dir(sequences) / 'trained_model_outputs' / 'model_comparison.csv'
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

    print(f"\n{'='*100}\nTOP-15 EXPLORATORY SWEEP COMPLETE\n{'='*100}")
    print(pd.DataFrame(results).to_string(index=False))
    print(f"\nFull results: {results_csv}")


if __name__ == "__main__":
    main()
