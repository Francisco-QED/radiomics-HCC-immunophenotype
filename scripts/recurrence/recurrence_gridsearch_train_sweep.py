##### ------------ Gridsearch + train for the top-N combinations from recurrence_sweep.py -------------- #####
# Reads recurrence_sweep_results.csv, picks the top N by oof_c_index, and runs --stage gridsearch
# then --stage train for each, collecting lock-set C-index results into a ranked comparison table.
# Mirrors gridsearch_train_sweep.py (the classification-pipeline equivalent) as closely as the
# survival pipeline's stage structure allows.
#
# Usage:
#   caffeinate -i python3 recurrence_gridsearch_train_sweep.py
#   python3 recurrence_gridsearch_train_sweep.py --dry-run       # list the combos, don't run
#   python3 recurrence_gridsearch_train_sweep.py --top-n 5       # only top 5 (default 10)
#
# --binwidth-label must match whatever recurrence_sweep.py used to produce the sweep_results.csv
# being read here.

import argparse
import csv
import subprocess
import time
from pathlib import Path

import pandas as pd

PYTHON   = "/usr/bin/python3"
PIPELINE = str(Path(__file__).resolve().parent / "recurrence_pipeline.py")
_RESULTS_DIR = (Path.home() / "Library" / "CloudStorage"
                / "OneDrive-TheMountSinaiHospital"
                / "Regeneron retrospective radiomics"
                / "Results" / "main_pipeline" / "recurrence_survival" / "radiomics")

LOG_DIR = Path(__file__).resolve().parent / "recurrence_sweep_logs"

DEFAULT_BINWIDTH_LABEL = "binwidth_perseq"  # must match recurrence_sweep.py's own default


def combo_base_dir(sequences, binwidth_label=DEFAULT_BINWIDTH_LABEL):
    return _RESULTS_DIR / binwidth_label / "combinations" / "_".join(sequences)


def summary_results_dir(binwidth_label=DEFAULT_BINWIDTH_LABEL):
    return _RESULTS_DIR / binwidth_label / "summary_results"


# Full vocabulary of known sequences -- must stay in sync with recurrence_sweep.py's combination rules.
KNOWN_SEQS = sorted(['dwi_b1', 'dwi_b2', 'ap', 'pvp', 'tp', 'hbp', 'adc', 't2', 'pre'],
                    key=len, reverse=True)  # longest first for greedy matching


def parse_sequences(seq_str):
    """Recover the original sequence list from the underscore-joined column in
    recurrence_sweep_results.csv. Simple longest-match greedy parse so 'dwi_b1' is never split
    into ['dwi', 'b1']."""
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


def load_top_combos(top_n, binwidth_label=DEFAULT_BINWIDTH_LABEL):
    sweep_results = summary_results_dir(binwidth_label) / "recurrence_sweep_results.csv"
    with open(sweep_results) as f:
        rows = list(csv.DictReader(f))
    succeeded = [r for r in rows if r['status'] == 'success']
    succeeded.sort(key=lambda r: float(r['oof_c_index']), reverse=True)
    combos = []
    for r in succeeded[:top_n]:
        combos.append({
            'sequences': parse_sequences(r['sequences']),
            'sequences_str': r['sequences'],
            'partition_oof_c_index': float(r['oof_c_index']),
            'partition_ci_lower': float(r['ci_lower']),
            'partition_ci_upper': float(r['ci_upper']),
            'partition_n_consensus': r.get('n_consensus_features', '?'),
        })
    return combos


def run_stage(stage, sequences, binwidth_label=DEFAULT_BINWIDTH_LABEL, event_col=None, time_col=None):
    base = combo_base_dir(sequences, binwidth_label)
    if stage == 'gridsearch':
        io_args = ['--output-dir', str(base / 'optimized_params'),
                   '--input-dir', str(base / 'partition_select')]
    elif stage == 'train':
        io_args = ['--output-dir', str(base / 'trained_model_outputs'),
                   '--input-dir', str(base / 'partition_select'),
                   '--input-params-dir', str(base / 'optimized_params')]
    else:
        io_args = []

    cmd = [PYTHON, PIPELINE, '--stage', stage, '--binwidth-label', binwidth_label,
           '--sequences'] + sequences + io_args
    # Must match whatever --event-col/--time-col recurrence_sweep.py's partition stage used --
    # gridsearch/train stages index y_dev.csv/y_lock.csv (already saved under those column names)
    # by these same names, not by re-reading --outcomes-path.
    if event_col:
        cmd += ['--event-col', event_col]
    if time_col:
        cmd += ['--time-col', time_col]
    label = f"{stage}_{binwidth_label}_{'_'.join(sequences)}"
    log_path = LOG_DIR / f"{label}.log"
    start = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start
    log_path.write_text(proc.stdout + "\n--- STDERR ---\n" + proc.stderr)
    return proc.returncode, elapsed, log_path


def parse_train_results(sequences, binwidth_label=DEFAULT_BINWIDTH_LABEL):
    """Read the model_comparison.csv written by the train stage."""
    csv_path = combo_base_dir(sequences, binwidth_label) / "trained_model_outputs" / "model_comparison.csv"
    if not csv_path.exists():
        return None
    return pd.read_csv(csv_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binwidth-label', type=str, default=DEFAULT_BINWIDTH_LABEL,
                        help="Must match the --binwidth-label used for recurrence_sweep.py's run.")
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--top-n', type=int, default=10)
    parser.add_argument('--event-col', type=str, default=None,
                        help="Must match the --event-col recurrence_sweep.py's partition stage "
                             "used for this binwidth_label (e.g. 'event_2yr'). Defaults to "
                             "recurrence_pipeline.py's own default ('event') when omitted.")
    parser.add_argument('--time-col', type=str, default=None,
                        help="Must match the --time-col recurrence_sweep.py's partition stage "
                             "used for this binwidth_label (e.g. 'time_days_2yr'). Defaults to "
                             "recurrence_pipeline.py's own default ('time_days') when omitted.")
    args = parser.parse_args()
    binwidth_label = args.binwidth_label

    output_csv = summary_results_dir(binwidth_label) / "recurrence_gridsearch_train_results.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    LOG_DIR.mkdir(exist_ok=True)
    combos = load_top_combos(args.top_n, binwidth_label)
    total = len(combos)

    print(f"Will run gridsearch + train for {total} combinations "
          f"(top {args.top_n} for binwidth_label={binwidth_label} from recurrence_sweep_results.csv):")
    for i, c in enumerate(combos, 1):
        print(f"  {i:2d}. {c['sequences_str']:<45}  partition C-index={c['partition_oof_c_index']:.3f}")
    if args.dry_run:
        return

    # Resume support: skip combos already marked success in a previous (interrupted) run.
    done_keys = set()
    if output_csv.exists():
        existing = pd.read_csv(output_csv)
        done = existing[existing['train_status'] == 'success']
        done_keys = set(done['sequences_str'])
        results = existing[existing['sequences_str'].isin(done_keys)].to_dict('records')
        print(f"Resuming: {len(done_keys)} combos already done, skipping them.")
    else:
        results = []

    for idx, c in enumerate(combos, 1):
        seqs = c['sequences']
        label = "_".join(seqs)

        if c['sequences_str'] in done_keys:
            print(f"\n[{idx}/{total}] {label} -- already done, skipping")
            continue

        print(f"\n[{idx}/{total}] {label}", flush=True)

        # --- gridsearch (RSF only -- LASSO-Cox's alpha is tuned by inner CV in the train stage) ---
        print(f"  gridsearch...", flush=True)
        rc_gs, t_gs, log_gs = run_stage('gridsearch', seqs, binwidth_label, args.event_col, args.time_col)
        print(f"  gridsearch {'OK' if rc_gs == 0 else 'FAILED'} in {t_gs:.0f}s")

        if rc_gs != 0:
            results.append({**c, 'gridsearch_status': 'failed', 'train_status': 'skipped'})
            pd.DataFrame(results).to_csv(output_csv, index=False)
            continue

        print(f"  train...", flush=True)
        rc_tr, t_tr, log_tr = run_stage('train', seqs, binwidth_label, args.event_col, args.time_col)
        print(f"  train {'OK' if rc_tr == 0 else 'FAILED'} in {t_tr:.0f}s")

        row = {**c, 'gridsearch_status': 'success' if rc_gs == 0 else 'failed',
               'train_status': 'success' if rc_tr == 0 else 'failed',
               'gridsearch_sec': round(t_gs), 'train_sec': round(t_tr)}

        if rc_tr == 0:
            comp = parse_train_results(seqs, binwidth_label)
            if comp is not None:
                comp['_c_float'] = pd.to_numeric(comp['C-index'], errors='coerce')
                valid = comp.dropna(subset=['_c_float'])
                if len(valid) > 0:
                    best = valid.loc[valid['_c_float'].idxmax()]
                    row.update({
                        'lock_best_model': best['Model'],
                        'lock_best_c_index': best['C-index'],
                        'lock_best_c_index_ci': best.get('95% CI', ''),
                        'lock_best_logrank_p': best.get('Logrank p', ''),
                        'lock_all_models': comp[['Model', 'C-index', '95% CI', 'Logrank p']].to_dict('records'),
                    })

        results.append(row)
        pd.DataFrame(results).to_csv(output_csv, index=False)
        print(f"  -> lock-set best: {row.get('lock_best_model','?')} "
              f"C-index={row.get('lock_best_c_index','?')} "
              f"logrank p={row.get('lock_best_logrank_p','?')}")

    # Final ranked table
    df = pd.DataFrame(results)
    print(f"\n{'='*110}")
    print("FINAL LOCK-SET COMPARISON (ranked by best-model C-index)")
    print(f"{'='*110}")
    if len(df) == 0 or 'train_status' not in df.columns:
        # No combos were even attempted (e.g. Stage 1 produced zero successful/consensus-feature
        # combos, so load_top_combos had nothing to pick) -- an empty DataFrame has no
        # 'train_status' column at all, so indexing it directly would KeyError instead of just
        # reporting nothing to rank.
        print("No combos were attempted -- nothing to rank.")
    else:
        success = df[df['train_status'] == 'success'].copy()
        success['_c'] = pd.to_numeric(success['lock_best_c_index'], errors='coerce')
        success = success.sort_values('_c', ascending=False)
        cols = ['sequences_str', 'lock_best_model', 'lock_best_c_index', 'lock_best_c_index_ci',
                'lock_best_logrank_p', 'partition_oof_c_index']
        print(success[cols].to_string(index=False))
    print(f"\nFull results: {output_csv}")


if __name__ == '__main__':
    main()
