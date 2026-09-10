"""
Publishes per-rank deliverables into a binwidth folder's summary_results/, reconstructing the
pieces whose original generating scripts are not present anywhere in the repo or on disk (they
predate this file and were never committed). See case_overlay_qc.py's docstring for the same
situation with the per-patient tumor overlay.

Reconstructed, not recovered — the formulas below were reverse-engineered by matching against the
existing binwidth_perseq_nowavelet/summary_results/rank1_..._lock_set_feature_alignment.csv values
(P085, P150, P090, P080, P111 all matched to 4+ decimal places), not derived from any surviving
code:

  direction_i   = sign of the mean nonzero LASSO coefficient for consensus feature i across the
                  partition stage's 5 folds (clean_partition_select/lasso_coeffs_fold.csv)
  oriented_i    = X_processed[feature_i] (already dev-fit-scaled) * direction_i
  n_features_aligned_of_N = count(oriented_i > 0)   -- NOT conditioned on the patient's true class
  mean_aligned_z           = mean(oriented_i)        -- also unconditional; despite the name this
                              is "how far this patient's profile sits in the LASSO-selected
                              features' high-TILs direction", not "agreement with ground truth"
  model_correct            = (model_proba_high_TILs >= 0.5) == true_class

What this script does, per combo ranked by build_sweep_diagnostics.py:
  1. Copies model_roc_comparison.png / _no_svm_linear.png from trained_model_outputs/ into
     summary_results/ with a rank{i}_{combo}_ prefix (these ARE native main_pipeline.py output,
     no reconstruction needed).
  2. Loads trained_model_outputs/best_model.pkl and reproduces SHAP bar + beeswarm plots + a raw
     shap-values CSV against X_dev_processed.csv, using main_pipeline.py's own
     create_shap_explainer() (imported directly, not reimplemented) so tree/linear/kernel explainer
     selection matches exactly. Applies the torch.set_num_threads(1) fix (see project memory) before
     touching any model, since one combo's best model is tabpfn and SHAP's KernelExplainer hangs on
     it otherwise.
  3. rank1 only: writes lock_set_feature_alignment.csv and lock_set_separation_plot.png using the
     formulas above.

Usage:
    python publish_rank_deliverables.py --radiomics-dir ".../binwidth_perseq_nowavelet2" --top-n 7
"""
import argparse
import glob
import os
import re
import shutil
import sys

import numpy as np
import pandas as pd
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import shap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from main_pipeline import create_shap_explainer  # reuse the exact explainer-selection logic


def _load_model_then_configure_torch(model_path):
    """Unpickle the model BEFORE torch is ever imported, then import torch and pin it to a
    single thread. Order matters: importing torch first and only then unpickling an xgboost
    model reliably segfaults here (Intel libiomp vs LLVM libomp both loaded -- the
    threadpoolctl warning that precedes the crash). This is a different manifestation of the
    same torch/OpenMP conflict noted in project memory for TabPFN (there it hangs instead of
    segfaulting, because TabPFN's own torch import happens inside SHAP's KernelExplainer call,
    after torch is already loaded either way -- unpickling xgboost first sidesteps both."""
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    import torch
    torch.set_num_threads(1)
    return model


def find_rank_combos(summary_dir):
    """Recover rank->combo mapping from results_<rank_label>_rank{i}_{combo}.csv filenames
    written by build_sweep_diagnostics.py (the authoritative ranking)."""
    pattern = os.path.join(summary_dir, "results_*_rank*_*.csv")
    rank_combo = {}
    for path in glob.glob(pattern):
        m = re.search(r"_rank(\d+)_(.+)\.csv$", os.path.basename(path))
        if m:
            rank_combo[int(m.group(1))] = m.group(2)
    return dict(sorted(rank_combo.items()))


def copy_roc_plots(combo_dir, summary_dir, rank, combo):
    for fname in ["model_roc_comparison.png", "model_roc_comparison_no_svm_linear.png"]:
        src = os.path.join(combo_dir, "trained_model_outputs", fname)
        if os.path.exists(src):
            dst = os.path.join(summary_dir, f"rank{rank}_{combo}_{fname}")
            shutil.copyfile(src, dst)
            print(f"  copied {fname} -> {os.path.basename(dst)}")
        else:
            print(f"  WARNING: missing {src}")


def get_best_model_name(summary_dir, combo):
    gs_path = os.path.join(summary_dir, "gridsearch_train_results.csv")
    df = pd.read_csv(gs_path)
    row = df[df["sequences_str"] == combo]
    if row.empty:
        return None
    return row.iloc[0]["lock_best_model"]


def publish_shap(combo_dir, summary_dir, rank, combo, model_name):
    model_path = os.path.join(combo_dir, "trained_model_outputs", "best_model.pkl")
    xdev_path = os.path.join(combo_dir, "clean_partition_select", "X_dev_processed.csv")
    if not (os.path.exists(model_path) and os.path.exists(xdev_path)):
        print(f"  WARNING: missing best_model.pkl or X_dev_processed.csv for {combo} -- skipping SHAP")
        return

    model = _load_model_then_configure_torch(model_path)
    X_dev = pd.read_csv(xdev_path, index_col=0)

    explainer = create_shap_explainer(model, X_dev, model_name=model_name)
    shap_values = explainer(X_dev)

    values = shap_values.values
    if values.ndim == 3:  # some explainers return (n, features, n_classes)
        values = values[:, :, -1]

    raw_df = pd.DataFrame(values, columns=X_dev.columns, index=X_dev.index)
    raw_path = os.path.join(summary_dir, f"rank{rank}_{combo}_shap_values_raw.csv")
    raw_df.to_csv(raw_path)
    print(f"  wrote {os.path.basename(raw_path)}  (model={model_name})")

    for plot_type, suffix in [("bar", "shap_bar.png"), ("dot", "shap_beeswarm.png")]:
        fig = plt.figure()
        shap.summary_plot(values, X_dev, plot_type=plot_type, show=False)
        out_path = os.path.join(summary_dir, f"rank{rank}_{combo}_{suffix}")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {os.path.basename(out_path)}")


def publish_lock_set_alignment(combo_dir, summary_dir, rank, combo, model_name, clinical_lock_path):
    consensus_path = os.path.join(combo_dir, "clean_partition_select", "consensus_features.csv")
    coefs_path = os.path.join(combo_dir, "clean_partition_select", "lasso_coeffs_fold.csv")
    xlock_path = os.path.join(combo_dir, "clean_partition_select", "X_lock_processed.csv")
    ylock_path = os.path.join(combo_dir, "clean_partition_select", "y_lock.csv")
    model_path = os.path.join(combo_dir, "trained_model_outputs", "best_model.pkl")

    feats = pd.read_csv(consensus_path)["feature"].tolist()
    n_feats = len(feats)

    coefs = pd.read_csv(coefs_path, index_col=0)
    fold_cols = [c for c in coefs.columns if c.startswith("Fold_")]
    sub = coefs.loc[feats, fold_cols]
    mean_nonzero = sub.apply(lambda r: r[r != 0].mean() if (r != 0).any() else 0.0, axis=1)
    direction = np.sign(mean_nonzero)

    X_lock = pd.read_csv(xlock_path, index_col=0)
    y_lock = pd.read_csv(ylock_path, index_col=0).iloc[:, 0]

    model = _load_model_then_configure_torch(model_path)
    proba = model.predict_proba(X_lock[feats])[:, -1]
    proba = pd.Series(proba, index=X_lock.index)

    clinical = pd.read_csv(clinical_lock_path)
    tils_col = clinical.set_index("PatientID")["TILs_resection"] if "PatientID" in clinical.columns \
        else clinical.set_index(clinical.columns[0])["TILs_resection"]

    rows = []
    for pid in X_lock.index:
        oriented = X_lock.loc[pid, feats].to_numpy(dtype=float) * direction.to_numpy(dtype=float)
        true_class = int(y_lock.loc[pid])
        p = float(proba.loc[pid])
        rows.append({
            "PatientID": pid,
            "TILs_resection": tils_col.get(pid, np.nan),
            "true_class": true_class,
            "model_proba_high_TILs": round(p, 3),
            "model_correct": (p >= 0.5) == bool(true_class),
            f"n_features_aligned_of_{n_feats}": int((oriented > 0).sum()),
            "mean_aligned_z": round(float(oriented.mean()), 3),
        })

    df = pd.DataFrame(rows).sort_values("model_proba_high_TILs", ascending=False)
    out_path = os.path.join(summary_dir, f"rank{rank}_{combo}_lock_set_feature_alignment.csv")
    df.to_csv(out_path, index=False)
    print(f"  wrote {os.path.basename(out_path)}")

    # --- separation plot ---
    df_plot = df.sort_values("model_proba_high_TILs", ascending=True)
    tils_levels = sorted(df_plot["TILs_resection"].dropna().unique())
    cmap = plt.get_cmap("RdBu_r")
    color_for = {lvl: cmap(i / max(len(tils_levels) - 1, 1)) for i, lvl in enumerate(tils_levels)}

    fig, ax = plt.subplots(figsize=(9, 0.28 * len(df_plot) + 1.5))
    for _, row in df_plot.iterrows():
        ax.barh(f"{row['PatientID']}  (TILs={row['TILs_resection']:g})",
                row["model_proba_high_TILs"], color=color_for[row["TILs_resection"]],
                edgecolor="black", linewidth=0.5)
    ax.axvline(0.5, color="black", linestyle="--", label="Decision threshold (0.5)")
    for i, (_, row) in enumerate(df_plot.iterrows()):
        if not row["model_correct"]:
            ax.text(row["model_proba_high_TILs"] + 0.02, i, "✗ misclassified",
                    color="red", fontweight="bold", va="center", fontsize=9)
    handles = [plt.Rectangle((0, 0), 1, 1, color=color_for[lvl]) for lvl in tils_levels]
    labels = [f"TILs={lvl:g}" for lvl in tils_levels]
    handles.append(plt.Line2D([0], [0], color="black", linestyle="--"))
    labels.append("Decision threshold (0.5)")
    ax.legend(handles, labels, loc="lower right", fontsize=9)
    ax.set_xlabel(f"Model-predicted probability of high TILs")
    ax.set_xlim(0, 1.05)
    ax.set_title(f"Lock-set separation: predicted probability vs. true TILs group (reconstructed)\n"
                 f"(rank{rank} model: {combo} / {model_name})")
    plt.tight_layout()
    plot_path = os.path.join(summary_dir, f"rank{rank}_{combo}_lock_set_separation_plot.png")
    plt.savefig(plot_path, dpi=200)
    plt.close(fig)
    print(f"  wrote {os.path.basename(plot_path)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--radiomics-dir", required=True, help="e.g. .../binwidth_perseq_nowavelet2")
    ap.add_argument("--top-n", type=int, default=7)
    args = ap.parse_args()

    combos_dir = os.path.join(args.radiomics_dir, "combinations")
    summary_dir = os.path.join(args.radiomics_dir, "summary_results")
    clinical_lock_path = os.path.join(summary_dir, "clinical_metadata_lock.csv")

    rank_combo = find_rank_combos(summary_dir)
    if not rank_combo:
        raise SystemExit(f"No results_*_rank*_*.csv found in {summary_dir} -- run build_sweep_diagnostics.py first")

    for rank, combo in list(rank_combo.items())[:args.top_n]:
        print(f"\n=== rank{rank}: {combo} ===")
        combo_dir = os.path.join(combos_dir, combo)
        copy_roc_plots(combo_dir, summary_dir, rank, combo)
        model_name = get_best_model_name(summary_dir, combo)
        if model_name is None:
            print(f"  WARNING: no lock_best_model found for {combo} in gridsearch_train_results.csv -- skipping SHAP/alignment")
            continue
        publish_shap(combo_dir, summary_dir, rank, combo, model_name)
        if rank == 1:
            publish_lock_set_alignment(combo_dir, summary_dir, rank, combo, model_name, clinical_lock_path)


if __name__ == "__main__":
    main()
