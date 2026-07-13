##### ------------ Post-sweep diagnostic summaries for the recurrence survival sweep -------------- #####
# Survival-pipeline counterpart to build_sweep_diagnostics.py (the classification version). Builds:
#
#   1. feature_consistency_summary.csv / feature_consistency.csv
#        Same idea as the classification version: how often each consensus radiomics feature is
#        re-selected across every combination that has a partition_select/consensus_features.csv.
#
#   2. results_rank{i}_{combo}.csv
#        For every combination with a trained trained_model_outputs/model_comparison.csv, the
#        per-model lock-set table, ranked by combo. Top --top-n combos only (default 5).
#
#        Ranking uses each combo's best C-index among models with a SIGNIFICANT logrank split
#        (KM high-vs-low-risk median split, p < 0.05) -- the survival analog of the classification
#        pipeline's non-degenerate-model filter. C-index itself is a ranking statistic and doesn't
#        have classification's "good AUC, degenerate Youden threshold" blind spot, but a model can
#        still have a mediocre-but-nonzero C-index whose median-risk split isn't actually a real
#        separation between groups -- this excludes those before crowning a combo's "best" model.
#
# Usage:
#   /usr/bin/python3 build_recurrence_sweep_diagnostics.py \
#       --recurrence-dir ".../recurrence_survival/binwidth_perseq"

import argparse
from pathlib import Path

import pandas as pd

KNOWN_SEQS = sorted(
    ["dwi_b1", "dwi_b2", "ap", "pvp", "tp", "hbp", "adc", "t2", "pre"],
    key=len, reverse=True,
)

LOGRANK_SIG_CUTOFF = 0.05

_DEFAULT_RECURRENCE_DIR = (
    Path.home() / "Library" / "CloudStorage" / "OneDrive-TheMountSinaiHospital"
    / "Regeneron retrospective radiomics" / "Results" / "main_pipeline"
    / "recurrence_survival" / "radiomics" / "binwidth_perseq"
)


def parse_feature(feature):
    """Split a consensus feature id into (sequence, filter, filter_type, class, name).
    Same convention as build_sweep_diagnostics.py's parse_feature."""
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
    selections = []
    combo_names = []
    for combo in sorted(p for p in combos_dir.iterdir() if p.is_dir()):
        consensus = combo / "partition_select" / "consensus_features.csv"
        if not consensus.exists():
            continue
        feats = pd.read_csv(consensus)["feature"].tolist()
        if not feats:
            continue  # zero-consensus-feature combos carry no signal to aggregate
        combo_names.append(combo.name)
        for f in feats:
            selections.append((combo.name, f))
    return selections, combo_names


def build_feature_consistency(selections, n_total_combos):
    order = []
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
    total = int(detail["n_appearances"].sum())

    def breakdown(col):
        s = detail.groupby(col)["n_appearances"].sum().sort_values(ascending=False)
        return [(idx, int(v), _pct(int(v), total)) for idx, v in s.items()]

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


def collect_alpha_diagnostics(combos_dir):
    """Concatenate every combo's per-fold alpha_diagnostics.csv (written by
    recurrence_pipeline.py's run_partition_stage) into one sweep-wide table, so the alpha
    selection can be audited across all combos at once instead of opening each one individually."""
    rows = []
    for combo in sorted(p for p in combos_dir.iterdir() if p.is_dir()):
        diag_path = combo / "partition_select" / "alpha_diagnostics.csv"
        if not diag_path.exists():
            continue
        df = pd.read_csv(diag_path)
        df.insert(0, "combo", combo.name)
        rows.append(df)
    if not rows:
        return None
    return pd.concat(rows, ignore_index=True)


def _best_significant_c_index(df):
    """Best C-index among models whose logrank p-value (KM high-vs-low-risk median split) is
    significant (< 0.05). Returns None if every model for this combo is non-significant."""
    df = df.copy()
    df["_c"] = pd.to_numeric(df["C-index"], errors="coerce")
    df["_p"] = pd.to_numeric(df["Logrank p"], errors="coerce")
    significant = df[(df["_p"] < LOGRANK_SIG_CUTOFF) & df["_c"].notna()]
    if significant.empty:
        return None
    return significant["_c"].max()


def build_rank_files(combos_dir, out_dir, top_n):
    scored = []
    skipped_no_significant_split = []
    for combo in sorted(p for p in combos_dir.iterdir() if p.is_dir()):
        mc = combo / "trained_model_outputs" / "model_comparison.csv"
        if not mc.exists():
            continue
        df = pd.read_csv(mc)
        best_c = _best_significant_c_index(df)
        if best_c is None:
            skipped_no_significant_split.append(combo.name)
            continue
        scored.append((best_c, combo.name, df))

    scored.sort(key=lambda t: (-t[0], t[1]))
    written = []
    for rank, (best_c, name, df) in enumerate(scored[:top_n], start=1):
        path = out_dir / f"results_rank{rank}_{name}.csv"
        df.drop(columns=[c for c in ("_c", "_p") if c in df.columns]).to_csv(path, index=False)
        written.append((rank, name, round(float(best_c), 3), path.name))
    return written, len(scored), skipped_no_significant_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recurrence-dir", type=Path, default=_DEFAULT_RECURRENCE_DIR,
                        help="Folder containing per-combo subfolders and summary_results/ "
                             "(a recurrence_survival/<binwidth-label> dir).")
    parser.add_argument("--top-n", type=int, default=5,
                        help="How many top combos get a ranked results file (default 5).")
    args = parser.parse_args()

    recurrence_dir = args.recurrence_dir
    combos_dir = recurrence_dir / "combinations"
    out_dir = recurrence_dir / "summary_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- feature consistency ---
    selections, combo_names = collect_selections(combos_dir)
    if not combo_names:
        raise SystemExit(f"No consensus_features.csv with features found under {combos_dir}")
    detail = build_feature_consistency(selections, len(combo_names))
    detail_path = out_dir / "feature_consistency.csv"
    summary_path = out_dir / "feature_consistency_summary.csv"
    detail.to_csv(detail_path, index=False)
    write_summary(detail, summary_path)
    print(f"Feature consistency over {len(combo_names)} combos "
          f"({int(detail['n_appearances'].sum())} selections, {len(detail)} unique features)")
    print(f"  wrote {detail_path.name}")
    print(f"  wrote {summary_path.name}")

    # --- alpha selection audit, across every combo/fold ---
    alpha_diag = collect_alpha_diagnostics(combos_dir)
    if alpha_diag is not None:
        alpha_path = out_dir / "alpha_diagnostics_summary.csv"
        alpha_diag.to_csv(alpha_path, index=False)
        n_folds = len(alpha_diag)
        at_ceiling = alpha_diag["alpha_max_ratio"] >= 0.99
        print(f"\nAlpha selection audit over {n_folds} fold-fits ({alpha_diag['combo'].nunique()} combos):")
        print(f"  best_alpha/alpha_max ratio: min={alpha_diag['alpha_max_ratio'].min():.3f}, "
              f"median={alpha_diag['alpha_max_ratio'].median():.3f}, "
              f"max={alpha_diag['alpha_max_ratio'].max():.3f}")
        print(f"  fold-fits at the grid ceiling (ratio >= 0.99): {at_ceiling.sum()}/{n_folds} "
              f"({100*at_ceiling.mean():.1f}%)")
        if at_ceiling.mean() > 0.5:
            print("  NOTE: majority of fold-fits are landing at the grid ceiling -- the 5-order-of-"
                  "magnitude span below alpha_max may still be too narrow for this sweep; consider "
                  "widening it (see _compute_alpha_max / run_fold_pruning_survival).")
        print(f"  wrote {alpha_path.name}")

    # --- ranked lock-set results ---
    written, n_ranked, skipped = build_rank_files(combos_dir, out_dir, args.top_n)
    print(f"\nRanked results over {n_ranked} combos with a significant logrank split "
          f"(top {args.top_n}):")
    for rank, name, best_c, fname in written:
        print(f"  rank{rank}: {name:<30} best C-index={best_c}  -> {fname}")
    if not written:
        print("  (none -- no trained_model_outputs/model_comparison.csv yet)")
    if skipped:
        print(f"\nExcluded ({len(skipped)}) -- no model's KM high-vs-low-risk split was "
              f"significant (logrank p >= {LOGRANK_SIG_CUTOFF}):")
        for name in skipped:
            print(f"  {name}")


if __name__ == "__main__":
    main()
