"""Guard for reading result files: only COMPLETED runs that came from this pipeline are used.
complete   : the run folder has summary.csv (ladders / survival runs write it last) or a manifest.json with a 'finished' key (clinical-only runs write it last).
mine       : if the manifest records the imported 'gpt' library path, it must be this project's own copy (vendored_functions/ in the repo, or this session's scratchpad snapshot).
Anything else (unfinished runs, runs from another agent or process) is skipped and reported; nothing is ever deleted or modified."""
import glob, json
from pathlib import Path


def _consistent(d):
    """summary.csv (written once at the end from the in-memory results) must agree with the saved OOF file: for every (sequence_set, family) the number of distinct
    repeats with predictions equals n_ok in the summary. A run folder whose files were reverted or overwritten by another writer fails this check."""
    import pandas as pd
    try:
        sm = pd.read_csv(d / "summary.csv")
        f = d / "oof_predictions.csv" if (d / "oof_predictions.csv").exists() else d / "oof_risk.csv"
        o = pd.read_csv(f, usecols=lambda c: c in ("sequence_set", "family", "repeat"))
        if "n_ok" not in sm.columns: return True
        keys = [k for k in ("sequence_set", "family") if k in sm.columns and k in o.columns]      # clinical-only runs have no sequence_set
        got = o.groupby(keys).repeat.nunique()
        for _, r in sm.iterrows():
            key = tuple(r[k] for k in keys) if len(keys) > 1 else r[keys[0]]
            if int(got.get(key, 0)) != int(r.n_ok): return False
        return True
    except Exception:
        return False


def usable_files(pattern, verbose=True):
    keep, skipped = [], []
    for f in sorted(glob.glob(str(pattern))):
        d = Path(f).parent; man = d / "manifest.json"; why = None
        try: m = json.loads(man.read_text()) if man.exists() else {}
        except Exception: m = {}
        if not ((d / "summary.csv").exists() or "finished" in m): why = "unfinished"
        elif (d / "summary.csv").exists() and not _consistent(d): why = "inconsistent (summary does not match the saved predictions: folder was altered)"
        elif m.get("gpt_sparse_script") and not any(k in m["gpt_sparse_script"] for k in ("/private/tmp/claude-503/", "final_pipeline/vendored_functions/")): why = "not from this pipeline"
        (skipped.append((f, why)) if why else keep.append(f))
    if verbose:
        print(f"run_guard: {len(keep)} complete runs used, {len(skipped)} skipped" + ("".join(f"\n   skipped ({w}): {Path(f).parent.name}" for f, w in skipped[:6])), flush=True)
    return keep
