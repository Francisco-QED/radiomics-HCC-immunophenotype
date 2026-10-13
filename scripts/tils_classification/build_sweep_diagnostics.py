##### ------------ Post-sweep diagnostic summaries for a binwidth folder -------------- #####
# Rebuilds the two summary_results diagnostics that accompany a sequence/train sweep, matching the
# format first produced for the binwidth5 run:
#
#   1. feature_consistency_cutoff{C}.csv / _summary.csv
#        How often each consensus radiomics feature is re-selected across every combination that has
#        a clean_partition_select/consensus_features.csv. The _summary breaks the pooled selections
#        (weighted by how many combos picked each feature) down by sequence, feature class, filter
#        type, and the top-15 class_feature types.
#
#   2. results_cutoffge{C}_vs_0-{C-1}_rank{i}_{combo}.csv
#        For every combination that has a trained trained_model_outputs/model_comparison.csv, the
#        per-classifier lock-set table (AUC + Youden operating point), one file per combo ranked by
#        that combo's best single-model AUC. Top --top-n combos only (default 5).
#
# Usage:
#   /usr/bin/python3 build_sweep_diagnostics.py \
#       --radiomics-dir ".../tils_2_vs_0-1/radiomics/binwidth_perseq"
#   # defaults to the binwidth_perseq folder if --radiomics-dir is omitted.

import argparse
import re
from pathlib import Path

import pandas as pd

# Longest first so 'dwi_b1'/'dwi_b2' win over any shorter prefix during the greedy sequence match.
KNOWN_SEQS = sorted(
    ["dwi_b1", "dwi_b2", "ap", "pvp", "tp", "hbp", "adc", "t2", "pre"],
    key=len, reverse=True,
)

# Only these columns from model_comparison.csv go into the ranked results files, and 'Model' is
# renamed to 'Classifier' -- matching the binwidth5 results_cutoffge2_vs_0-1_rank*.csv layout.
RANK_COLUMNS = {
    "Model": "Classifier",
    "AUC": "AUC",
    "AUC 95% CI": "AUC 95% CI",
    "Youden Threshold": "Youden Threshold",
    "Sensitivity (Youden)": "Sensitivity (Youden)",
    "Sens (Youden) 95% CI": "Sens (Youden) 95% CI",
    "Specificity (Youden)": "Specificity (Youden)",
    "Spec (Youden) 95% CI": "Spec (Youden) 95% CI",
}

_DEFAULT_RADIOMICS_DIR = (
    Path.home() / "Library" / "CloudStorage" / "OneDrive-TheMountSinaiHospital"
    / "Regeneron retrospective radiomics" / "Results" / "main_pipeline"
    / "immunophenotype" / "tils_2_vs_0-1" / "radiomics" / "binwidth_perseq"
)


def parse_feature(feature):
    """Split a consensus feature id into (sequence, filter, filter_type, class, name).

    Example: 'dwi_b2_wavelet-HLH_ngtdm_Contrast'
      -> sequence='dwi_b2', filter='wavelet-HLH', filter_type='wavelet',
         class='ngtdm', name='Contrast'
    The sequence prefix is matched greedily (longest known sequence first); the remainder is
    underscore-delimited as filter / class / name, since pyradiomics filter tags and CamelCase
    feature names never contain underscores themselves.
    """
    seq = next((s for s in KNOWN_SEQS if feature.startswith(s + "_")), None)
    if seq is None:
        raise ValueError(f"Unrecognized sequence prefix in feature {feature!r}")
    rest = feature[len(seq) + 1:]
    parts = rest.split("_")
    if len(parts) < 3:
        raise ValueError(f"Cannot split filter/class/name from {feature!r}")
    filt = parts[0]
    fclass = parts[1]
    fname = "_".join(parts[2:])
    if filt.startswith("wavelet"):
        ftype = "wavelet"
    elif filt.startswith("log-sigma"):
        ftype = "log-sigma"
    elif filt == "original":
        ftype = "original"
    else:
        ftype = filt
    return seq, filt, ftype, fclass, fname


def collect_selections(combos_dir):
    """Return an ordered list of (combo_name, feature) for every consensus feature in every combo."""
    selections = []
    combo_names = []
    for combo in sorted(p for p in combos_dir.iterdir() if p.is_dir()):
        consensus = combo / "clean_partition_select" / "consensus_features.csv"
        if not consensus.exists():
            continue
        combo_names.append(combo.name)
        feats = pd.read_csv(consensus)["feature"].tolist()
        for f in feats:
            selections.append((combo.name, f))
    return selections, combo_names


def build_feature_consistency(selections, n_total_combos):
    """Detail table: one row per unique feature with appearance count and the combos it came from."""
    order = []  # preserve first-seen order of features for a stable combination_list
    per_feature = {}
    for combo, feature in selections:
        if feature not in per_feature:
            per_feature[feature] = []
            order.append(feature)
        per_feature[feature].append(combo)

    rows = []
    for feature in order:
        combos = per_feature[feature]
        seq, filt, ftype, fclass, fname = parse_feature(feature)
        rows.append({
            "feature": feature,
            "sequence": seq,
            "filter": filt,
            "filter_type": ftype,
            "feature_class": fclass,
            "feature_name": fname,
            "n_appearances": len(combos),
            "n_total_combinations": n_total_combos,
            "pct_combinations": round(100.0 * len(combos) / n_total_combos, 1),
            "combination_list": "; ".join(combos),
        })
    df = pd.DataFrame(rows).sort_values(
        ["n_appearances", "feature"], ascending=[False, True]
    ).reset_index(drop=True)
    return df


def _pct(n, total):
    return f"{100.0 * n / total:.1f}%"


def write_summary(detail, out_path):
    """Reproduce the multi-section _summary.csv layout (breakdowns weighted by n_appearances)."""
    total = int(detail["n_appearances"].sum())

    def breakdown(col):
        s = detail.groupby(col)["n_appearances"].sum().sort_values(ascending=False)
        return [(idx, int(v), _pct(int(v), total)) for idx, v in s.items()]

    # class_feature type pooled over every selection (weighted by appearances), top 15
    detail = detail.copy()
    detail["class_feature"] = detail["feature_class"] + "_" + detail["feature_name"]
    top_types = (detail.groupby("class_feature")["n_appearances"].sum()
                 .sort_values(ascending=False).head(15))

    lines = []
    lines.append("BREAKDOWN BY SEQUENCE")
    lines.append("Sequence,Count,Pct")
    for name, cnt, pct in breakdown("sequence"):
        lines.append(f"{name},{cnt},{pct}")
    lines.append("")
    lines.append("BREAKDOWN BY FEATURE CLASS")
    lines.append("Class,Count,Pct")
    for name, cnt, pct in breakdown("feature_class"):
        lines.append(f"{name},{cnt},{pct}")
    lines.append("")
    lines.append("BREAKDOWN BY FILTER TYPE")
    lines.append("Filter,Count,Pct")
    for name, cnt, pct in breakdown("filter_type"):
        lines.append(f"{name},{cnt},{pct}")
    lines.append("")
    lines.append('"TOP 15 FEATURE TYPES (class_name, any sequence/filter)"')
    lines.append("ClassFeatureName,Count")
    for name, cnt in top_types.items():
        lines.append(f"{name},{int(cnt)}")

    out_path.write_text("\n".join(lines) + "\n")


def _best_non_degenerate_auc(df):
    """Best AUC among models whose Youden operating point isn't degenerate. AUC is a
    threshold-independent ranking metric, so a model can score well on it while its Youden
    threshold still collapses to predicting a single class for every lock patient (Sensitivity
    (Youden) == 0 or Specificity (Youden) == 0) -- not a real classifier at any usable cutoff.
    Returns None if every model for this combo is degenerate."""
    df = df.copy()
    df["_auc"] = pd.to_numeric(df["AUC"], errors="coerce")
    non_degenerate = df[(df["Sensitivity (Youden)"] > 0) & (df["Specificity (Youden)"] > 0)]
    if non_degenerate.empty:
        return None
    return non_degenerate["_auc"].max()


def build_rank_files(combos_dir, out_dir, cutoff, rank_label, top_n):
    """Write per-combo ranked lock-set result tables for combos that have a trained model_comparison.
    Ranks by each combo's best AUC among non-degenerate models only (see _best_non_degenerate_auc).
    The written table still lists all 8 classifiers, degenerate ones included, for context."""
    scored = []
    skipped_all_degenerate = []
    for combo in sorted(p for p in combos_dir.iterdir() if p.is_dir()):
        mc = combo / "trained_model_outputs" / "model_comparison.csv"
        if not mc.exists():
            continue
        df = pd.read_csv(mc)
        best_auc = _best_non_degenerate_auc(df)
        if best_auc is None:
            skipped_all_degenerate.append(combo.name)
            continue
        scored.append((best_auc, combo.name, df))

    scored.sort(key=lambda t: (-t[0], t[1]))
    written = []
    for rank, (best_auc, name, df) in enumerate(scored[:top_n], start=1):
        out = df[list(RANK_COLUMNS)].rename(columns=RANK_COLUMNS)
        path = out_dir / f"results_{rank_label}_rank{rank}_{name}.csv"
        out.to_csv(path, index=False)
        written.append((rank, name, round(float(best_auc), 3), path.name))
    return written, len(scored), skipped_all_degenerate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--radiomics-dir", type=Path, default=_DEFAULT_RADIOMICS_DIR,
                        help="Folder containing combinations/ and summary_results/ (a binwidth dir).")
    parser.add_argument("--top-n", type=int, default=5,
                        help="How many top-AUC combos get a ranked results file (default 5).")
    args = parser.parse_args()

    radiomics_dir = args.radiomics_dir
    combos_dir = radiomics_dir / "combinations"
    out_dir = radiomics_dir / "summary_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Recover the TILs cutoff from the '.../tils_{C}_vs_0-{C-1}/...' path segment.
    m = re.search(r"tils_(\d+)_vs_0-(\d+)", str(radiomics_dir))
    if not m:
        raise SystemExit(f"Could not find a tils_C_vs_0-M segment in {radiomics_dir}")
    cutoff = int(m.group(1))
    tils_segment = m.group(0)                       # e.g. 'tils_2_vs_0-1'
    rank_label = tils_segment.replace("tils_", "cutoffge")  # e.g. 'cutoffge2_vs_0-1'

    # --- feature consistency ---
    selections, combo_names = collect_selections(combos_dir)
    if not combo_names:
        raise SystemExit(f"No consensus_features.csv found under {combos_dir}")
    detail = build_feature_consistency(selections, len(combo_names))
    detail_path = out_dir / f"feature_consistency_cutoff{cutoff}.csv"
    summary_path = out_dir / f"feature_consistency_cutoff{cutoff}_summary.csv"
    detail.to_csv(detail_path, index=False)
    write_summary(detail, summary_path)
    print(f"Feature consistency over {len(combo_names)} combos "
          f"({int(detail['n_appearances'].sum())} selections, {len(detail)} unique features)")
    print(f"  wrote {detail_path.name}")
    print(f"  wrote {summary_path.name}")

    # --- ranked lock-set results ---
    written, n_ranked, skipped = build_rank_files(combos_dir, out_dir, cutoff, rank_label, args.top_n)
    print(f"\nRanked results over {n_ranked} combos with a non-degenerate model (top {args.top_n}):")
    for rank, name, best_auc, fname in written:
        print(f"  rank{rank}: {name:<30} best AUC={best_auc}  -> {fname}")
    if not written:
        print("  (none -- no trained_model_outputs/model_comparison.csv yet)")
    if skipped:
        print(f"\nExcluded ({len(skipped)}) -- every model's Youden operating point was degenerate "
              f"(Sensitivity or Specificity == 0):")
        for name in skipped:
            print(f"  {name}")


if __name__ == "__main__":
    main()
