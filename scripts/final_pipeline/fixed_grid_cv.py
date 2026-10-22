"""Step 1 of the simplified scheme: single-layer repeated CV over a FIXED (k, C) grid -- nothing is tuned.

For every (sequence set, k, C) the same outer folds are used (10 repeats x 5-fold, repeat r -> seed 42+r) and
the pooled OOF AUC is recorded per repeat. All cells are reported; none is selected. Feature selection (QC,
imputation, clustering, representative, top-k) and ICC are refit inside every training fold; there is NO inner CV.
If AUC is flat across the grid, the inner tuning layer of the nested design adds only noise.
Pipeline = the 'gpt' compact ridge pipeline (step S0 of ablation_ladder.py). Dev patients only; lock never read.
"""
import argparse, json, os, sys, time, warnings
from datetime import datetime
from pathlib import Path
sys.dont_write_bytecode = True
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ablation_ladder as al


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-parent", required=True)
    ap.add_argument("--core-dir", default=al.DEFAULT_CORE_DIR)
    ap.add_argument("--sets", nargs="+", default=list(al.SEQUENCE_SETS))
    ap.add_argument("--step", default="S0_theirs", choices=list(al.STEPS))
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--label", default="fixed_grid_cv")
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    core = al.load_core(a.core_dir)
    Selector = al.make_selector_class(core)
    X, y, dev_ids, lock_ids, split_path, feature_path, readers = core.load_development(Path(core.ROOT))
    run = Path(a.output_parent) / f"{a.label}_{datetime.now():%Y%m%d_%H%M%S}"
    run.mkdir(parents=True, exist_ok=False)
    grid = [(k, C) for k in core.K_GRID for C in core.C_GRID]
    (run / "manifest.json").write_text(json.dumps({
        "started": datetime.now().isoformat(), "endpoint": "TILs>=2", "n_development": len(dev_ids), "lock_evaluated": False,
        "step": a.step, "cfg": al.STEPS[a.step], "grid_k": list(core.K_GRID), "grid_C": list(core.C_GRID),
        "repeats": a.repeats, "outer": "5-fold stratified, seed 42+repeat", "inner_cv": "NONE (no tuning)",
        "sha256": {"features": al.sha256(feature_path), "split": al.sha256(split_path)}}, indent=2, default=str))
    print("RUN_DIRECTORY=" + str(run), flush=True)
    base = al.build_pipeline(core, Selector, al.STEPS[a.step])
    rows = []
    for set_name in a.sets:
        cols = sorted(c for c in X.columns if any(c.startswith(s + "_original_") for s in al.SEQUENCE_SETS[set_name]))
        Xs = X[cols]
        for rep in range(a.repeats):
            t0 = time.time()
            preds = {g: np.full(len(y), np.nan) for g in grid}
            for fold, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42 + rep).split(X, y), 1):
                icc = core.TechnicalICCFilter(*readers).fit(Xs.iloc[tr])
                Xtr, Xva, ytr = icc.transform(Xs.iloc[tr]), icc.transform(Xs.iloc[va]), y.iloc[tr]

                def one(g):
                    m = clone(base).set_params(select__k=g[0], model__C=g[1]).fit(Xtr, ytr)
                    return g, m.predict_proba(Xva)[:, 1]
                for g, p in Parallel(n_jobs=a.jobs, backend="threading")(delayed(one)(g) for g in grid):
                    preds[g][va] = p
            for (k, C), p in preds.items():
                rows.append(dict(sequence_set=set_name, k=k, C=C, repeat=rep, auc=core.classification_metrics(y, p)["auc"]))
            pd.DataFrame(rows).to_csv(run / "per_repeat_auc.csv", index=False)
            print(f"[{set_name}] repeat {rep}: {len(grid)} cells, {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    summ = df.groupby(["sequence_set", "k", "C"]).auc.agg(mean_auc="mean", sd_auc="std", min_auc="min", max_auc="max").reset_index()
    summ.round(4).to_csv(run / "grid_summary.csv", index=False)
    print("COMPLETED=" + str(run), flush=True)


if __name__ == "__main__":
    main()
