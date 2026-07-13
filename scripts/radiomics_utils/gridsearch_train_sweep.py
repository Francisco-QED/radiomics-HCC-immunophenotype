##### ------------ Gridsearch + train for the top-N combinations from sequence_sweep.py -------------- #####
# Reads sweep_results.csv, picks the top N by best_oof_auc per cutoff, and runs --stage gridsearch
# then --stage train for each, collecting lock-set AUC results into a ranked comparison table.
#
# Usage (run once per TILs cutoff; reads/writes that cutoff's own <binwidth-label>/summary_results/):
#   caffeinate -i python3 gridsearch_train_sweep.py --cutoff 2
#   caffeinate -i python3 gridsearch_train_sweep.py --cutoff 3
#   python3 gridsearch_train_sweep.py --cutoff 2 --dry-run       # list the combos, don't run
#   python3 gridsearch_train_sweep.py --cutoff 2 --top-n 5       # only top 5 (default 10)
#
# --binwidth-label targets a different binwidth setting's combos (must match whatever label
# sequence_sweep.py used for the partition stage, e.g. --binwidth-label binwidth5). gridsearch/train
# don't take --features-path themselves -- they read already-processed X_dev/X_lock from the
# partition stage's output, which already baked in whichever features were used.

import argparse
import csv
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

LOG_DIR = Path(__file__).resolve().parent / "sweep_logs"

# Must match sequence_sweep.py's DEFAULT_BINWIDTH_LABEL.
DEFAULT_BINWIDTH_LABEL = "binwidth_perseq"


# Must match sequence_sweep.py's combo_base_dir exactly -- that's where sequence_sweep.py's
# partition-stage output for this combo already lives, and where this script's gridsearch/train
# stages need to read from and write to.
def combo_base_dir(cutoff, sequences, binwidth_label=DEFAULT_BINWIDTH_LABEL):
    cutoff_val = int(cutoff)
    neg_max = cutoff_val - 1
    cutoff_label = f"tils_{cutoff_val}_vs_0-{neg_max}"
    return _RESULTS_DIR / cutoff_label / "radiomics" / binwidth_label / "combinations" / "_".join(sequences)


# Must match sequence_sweep.py's summary_results_dir: the per-cutoff/per-binwidth home for this
# cutoff's sweep_results.csv (read here) and gridsearch_train_results.csv (written here).
def summary_results_dir(cutoff, binwidth_label=DEFAULT_BINWIDTH_LABEL):
    cutoff_val = int(cutoff)
    neg_max = cutoff_val - 1
    cutoff_label = f"tils_{cutoff_val}_vs_0-{neg_max}"
    return _RESULTS_DIR / cutoff_label / "radiomics" / binwidth_label / "summary_results"


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


def load_top_combos(cutoff, top_n, binwidth_label=DEFAULT_BINWIDTH_LABEL):
    sweep_results = summary_results_dir(cutoff, binwidth_label) / "sweep_results.csv"
    with open(sweep_results) as f:
        rows = list(csv.DictReader(f))
    succeeded = [r for r in rows if r['status'] == 'success' and int(r['cutoff']) == cutoff]
    succeeded.sort(key=lambda r: float(r['best_oof_auc']), reverse=True)
    combos = []
    for r in succeeded[:top_n]:
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


def run_stage(stage, cutoff, sequences, binwidth_label=DEFAULT_BINWIDTH_LABEL, extra_args=None):
    base = combo_base_dir(cutoff, sequences, binwidth_label)
    if stage == 'gridsearch':
        io_args = ['--output-dir', str(base / 'optimized_params'),
                   '--input-dir', str(base / 'clean_partition_select')]
    elif stage == 'train':
        io_args = ['--output-dir', str(base / 'trained_model_outputs'),
                   '--input-data-dir', str(base / 'clean_partition_select'),
                   '--input-params-dir', str(base / 'optimized_params')]
    else:
        io_args = []

    cmd = [PYTHON, PIPELINE, '--stage', stage,
           '--cutoff', str(cutoff),
           '--sequences'] + sequences + io_args
    if extra_args:
        cmd += extra_args
    label = f"{stage}_cutoff{cutoff}_{'_'.join(sequences)}"
    log_path = LOG_DIR / f"{label}.log"
    start = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start
    log_path.write_text(proc.stdout + "\n--- STDERR ---\n" + proc.stderr)
    return proc.returncode, elapsed, log_path


def parse_train_results(cutoff, sequences, binwidth_label=DEFAULT_BINWIDTH_LABEL):
    """Read the model_comparison.csv written by the train stage."""
    csv_path = combo_base_dir(cutoff, sequences, binwidth_label) / "trained_model_outputs" / "model_comparison.csv"
    if not csv_path.exists():
        return None
    return pd.read_csv(csv_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cutoff', type=int, required=True, choices=[2, 3],
                        help='TILs cutoff to run. Run once per cutoff.')
    parser.add_argument('--binwidth-label', type=str, default=DEFAULT_BINWIDTH_LABEL,
                        help='Must match the --binwidth-label used for this cutoff\'s sequence_sweep.py run.')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--top-n', type=int, default=10)
    args = parser.parse_args()
    binwidth_label = args.binwidth_label

    # Per-cutoff output: lives beside this cutoff's own combinations, not at the shared root.
    output_csv = summary_results_dir(args.cutoff, binwidth_label) / "gridsearch_train_results.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    LOG_DIR.mkdir(exist_ok=True)
    combos = load_top_combos(args.cutoff, args.top_n, binwidth_label)
    total = len(combos)

    print(f"Will run gridsearch + train for {total} combinations "
          f"(top {args.top_n} for cutoff {args.cutoff} from sweep_results.csv):")
    for i, c in enumerate(combos, 1):
        print(f"  {i:2d}. cutoff={c['cutoff']}  {c['sequences_str']:<45}  "
              f"partition AUC={c['partition_best_oof_auc']:.3f}")
    if args.dry_run:
        return

    # Load any partial results from a previous interrupted run so we skip already-done combos.
    done_keys = set()
    if output_csv.exists():
        existing = pd.read_csv(output_csv)
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
        rc_gs, t_gs, log_gs = run_stage('gridsearch', cutoff, seqs, binwidth_label)
        print(f"  gridsearch {'OK' if rc_gs == 0 else 'FAILED'} in {t_gs:.0f}s")

        if rc_gs != 0:
            results.append({**c, 'gridsearch_status': 'failed', 'train_status': 'skipped'})
            pd.DataFrame(results).to_csv(output_csv, index=False)
            continue

        # SHAP intentionally excluded from the sweep: KernelExplainer (used for GP and TabPFN)
        # runs thousands of model evaluations per sample and takes hours per combo, making a
        # 20-combo sweep impractical. Run SHAP separately on the winning combination after the
        # sweep identifies it: python main_pipeline.py --stage train --shap --cutoff X --sequences ...
        print(f"  train...", flush=True)
        rc_tr, t_tr, log_tr = run_stage('train', cutoff, seqs, binwidth_label)
        print(f"  train {'OK' if rc_tr == 0 else 'FAILED'} in {t_tr:.0f}s")

        row = {**c, 'gridsearch_status': 'success' if rc_gs == 0 else 'failed',
               'train_status': 'success' if rc_tr == 0 else 'failed',
               'gridsearch_sec': round(t_gs), 'train_sec': round(t_tr)}

        if rc_tr == 0:
            comp = parse_train_results(cutoff, seqs, binwidth_label)
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
        pd.DataFrame(results).to_csv(output_csv, index=False)
        print(f"  -> lock-set best: {row.get('lock_best_model','?')} AUC={row.get('lock_best_auc','?')}")

    # Final ranked table
    df = pd.DataFrame(results)
    print(f"\n{'='*110}")
    print("FINAL LOCK-SET COMPARISON (ranked by best-model AUC)")
    print(f"{'='*110}")
    if len(df) == 0 or 'train_status' not in df.columns:
        print("No combos were attempted -- nothing to rank.")
    else:
        success = df[df['train_status'] == 'success'].copy()
        success['_auc'] = pd.to_numeric(success['lock_best_auc'], errors='coerce')
        success = success.sort_values('_auc', ascending=False)
        cols = ['cutoff', 'sequences_str', 'lock_best_model', 'lock_best_auc', 'lock_best_auc_ci',
                'lock_best_sensitivity', 'lock_best_specificity', 'partition_best_oof_auc']
        print(success[cols].to_string(index=False))
    print(f"\nFull results: {output_csv}")


if __name__ == '__main__':
    main()
