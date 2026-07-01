##### ------------ Gridsearch + train for the top-N combinations from sequence_sweep.py -------------- #####
# Reads sweep_results.csv, picks the top N by best_oof_auc per cutoff, and runs --stage gridsearch
# then --stage train for each, collecting lock-set AUC results into a ranked comparison table.
#
# Usage:
#   caffeinate -i python3 gridsearch_train_sweep.py
#   python3 gridsearch_train_sweep.py --dry-run          # list the combos, don't run
#   python3 gridsearch_train_sweep.py --top-n 5          # only top 5 per cutoff (default 10)

import argparse
import csv
import subprocess
import time
from pathlib import Path

import pandas as pd

PYTHON = "/usr/bin/python3"
PIPELINE = str(Path(__file__).resolve().parent / "main_pipeline.py")
SWEEP_RESULTS = Path(__file__).resolve().parent / "sweep_results.csv"
OUTPUT_CSV = Path(__file__).resolve().parent / "gridsearch_train_results.csv"
LOG_DIR = Path(__file__).resolve().parent / "sweep_logs"

# Full vocabulary of known sequences -- must stay in sync with sequence_sweep.py's combination rules.
KNOWN_SEQS = sorted(['dwi_b1', 'dwi_b2', 'ap', 'pvp', 'tp', 'hbp', 'adc', 't2', 'pre'],
                    key=len, reverse=True)  # longest first for greedy matching


def parse_sequences(seq_str):
    """Recover the original sequence list from the underscore-joined column in sweep_results.csv.
    Simple longest-match greedy parse so 'dwi_b1' is never split into ['dwi', 'b1']."""
    parts = []
    rem = seq_str
    while rem:
        matched = False
        for seq in KNOWN_SEQS:
            if rem.startswith(seq):
                parts.append(seq)
                rem = rem[len(seq):]
                if rem.startswith('_'):
                    rem = rem[1:]
                matched = True
                break
        if not matched:
            raise ValueError(f"Cannot parse sequence string {seq_str!r}, stuck at {rem!r}")
    return parts


def load_top_combos(top_n):
    with open(SWEEP_RESULTS) as f:
        rows = list(csv.DictReader(f))
    succeeded = [r for r in rows if r['status'] == 'success']
    succeeded.sort(key=lambda r: float(r['best_oof_auc']), reverse=True)
    combos = []
    for cutoff in ['2', '3']:
        top = [r for r in succeeded if r['cutoff'] == cutoff][:top_n]
        for r in top:
            combos.append({
                'cutoff': int(r['cutoff']),
                'sequences': parse_sequences(r['sequences']),
                'sequences_str': r['sequences'],
                'partition_best_model': r['best_model'],
                'partition_best_oof_auc': float(r['best_oof_auc']),
                'partition_auc_ci_lower': float(r['best_auc_ci_lower']),
                'partition_auc_ci_upper': float(r['best_auc_ci_upper']),
                'partition_n_consensus': r.get('n_consensus_features', '?'),
            })
    return combos


def run_stage(stage, cutoff, sequences, extra_args=None):
    cmd = [PYTHON, PIPELINE, '--stage', stage,
           '--cutoff', str(cutoff),
           '--sequences'] + sequences
    if extra_args:
        cmd += extra_args
    label = f"{stage}_cutoff{cutoff}_{'_'.join(sequences)}"
    log_path = LOG_DIR / f"{label}.log"
    start = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start
    log_path.write_text(proc.stdout + "\n--- STDERR ---\n" + proc.stderr)
    return proc.returncode, elapsed, log_path


def parse_train_results(cutoff, sequences):
    """Read the model_comparison.csv written by the train stage."""
    cutoff_val = int(cutoff)
    neg_max = cutoff_val - 1
    cutoff_label = f"test_{cutoff_val}_vs_0-{neg_max}"
    base = (f"/Volumes/T7 Shield/Restrepo/HCC_Retrospective_radiomics/Results/main_pipeline/"
            f"{cutoff_label}/dev110_test30/binwidth5/{'_'.join(sequences)}/trained_model_outputs")
    csv_path = Path(base) / "model_comparison.csv"
    if not csv_path.exists():
        return None
    return pd.read_csv(csv_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--top-n', type=int, default=10)
    args = parser.parse_args()

    LOG_DIR.mkdir(exist_ok=True)
    combos = load_top_combos(args.top_n)
    total = len(combos)

    print(f"Will run gridsearch + train for {total} combinations "
          f"(top {args.top_n} per cutoff from sweep_results.csv):")
    for i, c in enumerate(combos, 1):
        print(f"  {i:2d}. cutoff={c['cutoff']}  {c['sequences_str']:<45}  "
              f"partition AUC={c['partition_best_oof_auc']:.3f}")
    if args.dry_run:
        return

    # Load any partial results from a previous interrupted run so we skip already-done combos.
    done_keys = set()
    if OUTPUT_CSV.exists():
        existing = pd.read_csv(OUTPUT_CSV)
        done = existing[existing['train_status'] == 'success']
        done_keys = set(zip(done['cutoff'].astype(str), done['sequences_str']))
        results = existing.to_dict('records')
        print(f"Resuming: {len(done_keys)} combos already done, skipping them.")
    else:
        results = []

    for idx, c in enumerate(combos, 1):
        cutoff = c['cutoff']
        seqs = c['sequences']
        label = f"cutoff{cutoff}_{'_'.join(seqs)}"

        if (str(cutoff), c['sequences_str']) in done_keys:
            print(f"\n[{idx}/{total}] {label} -- already done, skipping")
            continue

        print(f"\n[{idx}/{total}] {label}", flush=True)

        # --- gridsearch ---
        print(f"  gridsearch...", flush=True)
        rc_gs, t_gs, log_gs = run_stage('gridsearch', cutoff, seqs)
        print(f"  gridsearch {'OK' if rc_gs == 0 else 'FAILED'} in {t_gs:.0f}s")

        if rc_gs != 0:
            results.append({**c, 'gridsearch_status': 'failed', 'train_status': 'skipped'})
            pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
            continue

        # SHAP intentionally excluded from the sweep: KernelExplainer (used for GP and TabPFN)
        # runs thousands of model evaluations per sample and takes hours per combo, making a
        # 20-combo sweep impractical. Run SHAP separately on the winning combination after the
        # sweep identifies it: python main_pipeline.py --stage train --shap --cutoff X --sequences ...
        print(f"  train...", flush=True)
        rc_tr, t_tr, log_tr = run_stage('train', cutoff, seqs)
        print(f"  train {'OK' if rc_tr == 0 else 'FAILED'} in {t_tr:.0f}s")

        row = {**c, 'gridsearch_status': 'success' if rc_gs == 0 else 'failed',
               'train_status': 'success' if rc_tr == 0 else 'failed',
               'gridsearch_sec': round(t_gs), 'train_sec': round(t_tr)}

        if rc_tr == 0:
            comp = parse_train_results(cutoff, seqs)
            if comp is not None:
                # Pull AUC column -- stored as string "0.XX" in the comparison CSV.
                comp['_auc_float'] = pd.to_numeric(comp['AUC'], errors='coerce')
                best = comp.loc[comp['_auc_float'].idxmax()]
                row.update({
                    'lock_best_model': best['Model'],
                    'lock_best_auc': best['AUC'],
                    'lock_best_auc_ci': best.get('AUC 95% CI', ''),
                    'lock_best_sensitivity': best.get('Sensitivity', ''),
                    'lock_best_specificity': best.get('Specificity', ''),
                    # Keep the full per-model summary as a compact JSON blob for reference.
                    'lock_all_models': comp[['Model', 'AUC', 'AUC 95% CI']].to_dict('records'),
                })

        results.append(row)
        pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
        print(f"  -> lock-set best: {row.get('lock_best_model','?')} AUC={row.get('lock_best_auc','?')}")

    # Final ranked table
    df = pd.DataFrame(results)
    success = df[df['train_status'] == 'success'].copy()
    success['_auc'] = pd.to_numeric(success['lock_best_auc'], errors='coerce')
    success = success.sort_values('_auc', ascending=False)

    print(f"\n{'='*110}")
    print("FINAL LOCK-SET COMPARISON (ranked by best-model AUC)")
    print(f"{'='*110}")
    cols = ['cutoff', 'sequences_str', 'lock_best_model', 'lock_best_auc', 'lock_best_auc_ci',
            'lock_best_sensitivity', 'lock_best_specificity', 'partition_best_oof_auc']
    print(success[cols].to_string(index=False))
    print(f"\nFull results: {OUTPUT_CSV}")


if __name__ == '__main__':
    main()
