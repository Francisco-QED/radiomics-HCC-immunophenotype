"""
Runs tils_regression_pipeline.py (partition -> gridsearch -> train) for the 7 pre-specified,
LI-RADS-motivated sequence combinations (same combo list as run_seven_combo_partition.py /
sequence_sweep_findings_report.docx), no cutoff involved (TILs_resection modeled as continuous).

Output layout mirrors the classification sweep's convention, just without a tils_{C}_vs_0-{M}
cutoff segment:
    <RESULTS_ROOT>/all_tils/radiomics/<binwidth-label>/combinations/<seq_str>/{clean_partition_select,optimized_params,trained_model_outputs}
    <RESULTS_ROOT>/all_tils/radiomics/<binwidth-label>/summary_results/{sweep_results.csv, gridsearch_train_results.csv, ...}

Usage:
    caffeinate -i python3 regression_sweep.py --binwidth-label binwidth_perseq_nowavelet
    python3 regression_sweep.py --binwidth-label binwidth_perseq_nowavelet --dry-run
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

PYTHON = "/usr/bin/python3"
PIPELINE = str(Path(__file__).resolve().parent / "tils_regression_pipeline.py")

_RESULTS_DIR = (Path.home() / "Library" / "CloudStorage"
                / "OneDrive-TheMountSinaiHospital"
                / "Regeneron retrospective radiomics"
                / "Results" / "main_pipeline" / "immunophenotype")

LOG_DIR = Path(__file__).resolve().parent / "regression_sweep_logs"
DEFAULT_BINWIDTH_LABEL = "binwidth_perseq_nowavelet"

SEVEN_COMBOS = [
    ["ap", "pvp", "hbp", "dwi_b1", "adc"],
    ["ap", "pvp", "hbp", "dwi_b1", "adc", "t2"],
    ["ap", "tp", "hbp", "dwi_b1", "adc"],
    ["ap", "pvp", "tp", "dwi_b1", "adc"],
    ["pre", "ap", "pvp", "hbp", "dwi_b1", "adc"],
    ["ap", "pvp", "hbp", "dwi_b2", "adc"],
    ["pre", "ap", "pvp", "tp", "hbp"],
]


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
    log_path = LOG_DIR / f"{label}.log"
    start = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start
    log_path.write_text(proc.stdout + "\n--- STDERR ---\n" + proc.stderr)
    return proc.returncode, elapsed, log_path


def parse_train_results(sequences, binwidth_label):
    csv_path = combo_base_dir(sequences, binwidth_label) / "trained_model_outputs" / "model_comparison.csv"
    if not csv_path.exists():
        return None
    return pd.read_csv(csv_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binwidth-label", type=str, default=DEFAULT_BINWIDTH_LABEL)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--feature-selection-method", choices=['fold-consensus', 'stability'],
        default='fold-consensus',
        help="Passed through to tils_regression_pipeline.py's --stage partition. 'stability' also "
             "uses --stability-threshold/--stability-n-resamples below. Point --binwidth-label at a "
             "distinct folder when using 'stability' -- it does NOT namespace itself, and reusing "
             "the fold-consensus sweep's folder will overwrite that method's saved artifacts.")
    ap.add_argument("--stability-threshold", type=float, default=0.7)
    ap.add_argument("--stability-n-resamples", type=int, default=200)
    args = ap.parse_args()
    binwidth_label = args.binwidth_label

    partition_extra_args = []
    if args.feature_selection_method == 'stability':
        partition_extra_args = [
            '--feature-selection-method', 'stability',
            '--stability-threshold', str(args.stability_threshold),
            '--stability-n-resamples', str(args.stability_n_resamples),
        ]

    print(f"Will run partition+gridsearch+train for {len(SEVEN_COMBOS)} combinations "
          f"(binwidth-label={binwidth_label}, feature-selection-method={args.feature_selection_method}):")
    for s in SEVEN_COMBOS:
        print(f"  {s}")
    if args.dry_run:
        return

    LOG_DIR.mkdir(exist_ok=True)
    results_csv = summary_results_dir(binwidth_label) / "sweep_results.csv"
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    gs_train_csv = summary_results_dir(binwidth_label) / "gridsearch_train_results.csv"

    partition_results = []
    gs_train_results = []
    if results_csv.exists():
        existing = pd.read_csv(results_csv)
        partition_results = existing.to_dict('records')
    done_partition = {r['sequences'] for r in partition_results if r.get('status') == 'success'}
    if gs_train_csv.exists():
        existing_gt = pd.read_csv(gs_train_csv)
        gs_train_results = existing_gt.to_dict('records')
    done_train = {r['sequences_str'] for r in gs_train_results if r.get('train_status') == 'success'}

    for idx, sequences in enumerate(SEVEN_COMBOS, 1):
        seq_str = "_".join(sequences)
        print(f"\n[{idx}/{len(SEVEN_COMBOS)}] {sequences}", flush=True)

        # --- partition ---
        if seq_str in done_partition:
            print("  partition already done, skipping")
        else:
            print("  partition...", flush=True)
            rc, elapsed, log = run_stage('partition', sequences, binwidth_label, partition_extra_args)
            row = {'sequences': seq_str, 'n_sequences': len(sequences),
                   'status': 'success' if rc == 0 else 'failed', 'elapsed_sec': round(elapsed, 1),
                   'log_path': str(log)}
            if rc != 0:
                row['error'] = log.read_text().strip().splitlines()[-1] if log.exists() else 'unknown'
            partition_results.append(row)
            pd.DataFrame(partition_results).to_csv(results_csv, index=False)
            print(f"  partition {'OK' if rc == 0 else 'FAILED'} in {elapsed:.0f}s")
            if rc != 0:
                continue

        # --- gridsearch ---
        print("  gridsearch...", flush=True)
        rc_gs, t_gs, log_gs = run_stage('gridsearch', sequences, binwidth_label)
        print(f"  gridsearch {'OK' if rc_gs == 0 else 'FAILED'} in {t_gs:.0f}s")
        if rc_gs != 0:
            gs_train_results.append({'sequences_str': seq_str, 'gridsearch_status': 'failed', 'train_status': 'skipped'})
            pd.DataFrame(gs_train_results).to_csv(gs_train_csv, index=False)
            continue

        # --- train ---
        print("  train...", flush=True)
        rc_tr, t_tr, log_tr = run_stage('train', sequences, binwidth_label)
        print(f"  train {'OK' if rc_tr == 0 else 'FAILED'} in {t_tr:.0f}s")
        row = {'sequences_str': seq_str, 'gridsearch_status': 'success', 'train_status': 'success' if rc_tr == 0 else 'failed',
               'gridsearch_sec': round(t_gs), 'train_sec': round(t_tr)}
        if rc_tr == 0:
            comp = parse_train_results(sequences, binwidth_label)
            if comp is not None:
                comp['_rho_float'] = pd.to_numeric(comp['Spearman_rho'], errors='coerce')
                best = comp.loc[comp['_rho_float'].idxmax()]
                row.update({
                    'lock_best_model': best['Model'], 'lock_best_rho': best['Spearman_rho'],
                    'lock_best_rho_ci': best.get('rho 95% CI', ''), 'lock_best_r2': best.get('R2', ''),
                    'lock_best_mae': best.get('MAE', ''),
                    'lock_all_models': comp[['Model', 'Spearman_rho', 'rho 95% CI', 'R2', 'MAE']].to_dict('records'),
                })
        gs_train_results.append(row)
        pd.DataFrame(gs_train_results).to_csv(gs_train_csv, index=False)
        print(f"  -> lock-set best: {row.get('lock_best_model', '?')} rho={row.get('lock_best_rho', '?')}")

    df = pd.DataFrame(gs_train_results)
    print(f"\n{'='*100}\nFINAL LOCK-SET COMPARISON (ranked by best-model Spearman rho)\n{'='*100}")
    if len(df) == 0 or 'train_status' not in df.columns:
        print("No combos were attempted.")
    else:
        success = df[df['train_status'] == 'success'].copy()
        success['_rho'] = pd.to_numeric(success['lock_best_rho'], errors='coerce')
        success = success.sort_values('_rho', ascending=False)
        cols = ['sequences_str', 'lock_best_model', 'lock_best_rho', 'lock_best_rho_ci', 'lock_best_r2', 'lock_best_mae']
        print(success[cols].to_string(index=False))
    print(f"\nFull results: {gs_train_csv}")


if __name__ == "__main__":
    main()
