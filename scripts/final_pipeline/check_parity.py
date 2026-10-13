"""Fails (exit 1) if the two selection pipelines drift apart.

The classification (main_pipeline.py) and ordinal (tils_regression_pipeline.py) pipelines are meant to be
mirror images: only the things that depend on the outcome type may differ (univariate metric, the penalized
model, outcome-specific metrics/plots). Everything in SHARED must be textually identical (comments ignored).
Run after editing either file:   /usr/bin/python3 check_parity.py
"""
import ast
import re
import sys
from pathlib import Path

S = Path(__file__).resolve().parent.parent
FILES = {"classification": S / "tils_classification/main_pipeline.py",
         "ordinal": S / "tils_regression/tils_regression_pipeline.py"}
SHARED = ["benjamini_hochberg", "global_cleanup", "interobserver_stability", "spearman_filter",
          "preselect_sequences", "filter_feature_families", "compute_correlation_clusters",
          "load_preprocessed_data"]
# Intentionally different (outcome-type specific): calculate_feature_metrics, run_fold_pruning (screen +
# penalized model), stratified_resample / CV splitting (strata), summarize_nested_cv, model zoo/tuning grids,
# threshold/ROC/sensitivity code (classification only), scatter plots (ordinal only).
EXPECTED_FLAG_DIFF = {"--cutoff", "--scoring", "--threshold", "--univariate-metric"}


def bodies(path):
    s = Path(path).read_text()
    L = s.splitlines()
    out = {}
    for n in ast.parse(s).body:
        if isinstance(n, ast.FunctionDef):
            txt = re.sub(r"#.*", "", "\n".join(L[n.lineno - 1:n.end_lineno]))
            out[n.name] = [l.strip() for l in txt.splitlines() if l.strip()]
    return out


def flags(path):
    out = {}
    for n in ast.walk(ast.parse(Path(path).read_text())):
        if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "add_argument" and n.args:
            d = None
            for k in n.keywords:
                if k.arg == "default":
                    try:
                        d = ast.literal_eval(k.value)
                    except Exception:
                        d = ast.unparse(k.value)
            out[n.args[0].value] = d
    return out


def main():
    bad = []
    c, r = bodies(FILES["classification"]), bodies(FILES["ordinal"])
    for name in SHARED:
        if c.get(name) != r.get(name):
            bad.append(f"function differs: {name}")
    fc, fr = flags(FILES["classification"]), flags(FILES["ordinal"])
    for k in sorted(set(fc) ^ set(fr)):
        if k not in EXPECTED_FLAG_DIFF:
            bad.append(f"flag only in {'classification' if k in fc else 'ordinal'}: {k}")
    for k in sorted(set(fc) & set(fr)):
        if fc[k] != fr[k] and k != "--metric-cutoff":      # AUC 0.65 vs |rho| 0.25 are different scales
            bad.append(f"default differs: {k}: {fc[k]!r} vs {fr[k]!r}")
    if bad:
        print("PARITY CHECK FAILED:\n  - " + "\n  - ".join(bad))
        sys.exit(1)
    print(f"parity OK: {len(SHARED)} shared functions identical; CLI flags match "
          f"(expected differences: {sorted(EXPECTED_FLAG_DIFF)} + --metric-cutoff default).")


if __name__ == "__main__":
    main()
