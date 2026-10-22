"""Guard for reading result files: only COMPLETED runs that came from this pipeline are used.
complete   : the run folder has summary.csv (ladders / survival runs write it last) or a manifest.json with a 'finished' key (clinical-only runs write it last).
mine       : if the manifest records the imported 'gpt' library path, it must be this project's own copy (vendored_functions/ in the repo, or this session's scratchpad snapshot).
Anything else (unfinished runs, runs from another agent or process) is skipped and reported; nothing is ever deleted or modified."""
import glob, json
from pathlib import Path


def usable_files(pattern, verbose=True):
    keep, skipped = [], []
    for f in sorted(glob.glob(str(pattern))):
        d = Path(f).parent; man = d / "manifest.json"; why = None
        try: m = json.loads(man.read_text()) if man.exists() else {}
        except Exception: m = {}
        if not ((d / "summary.csv").exists() or "finished" in m): why = "unfinished"
        elif m.get("gpt_sparse_script") and not any(k in m["gpt_sparse_script"] for k in ("/private/tmp/claude-503/", "final_pipeline/vendored_functions/")): why = "not from this pipeline"
        (skipped.append((f, why)) if why else keep.append(f))
    if verbose:
        print(f"run_guard: {len(keep)} complete runs used, {len(skipped)} skipped" + ("".join(f"\n   skipped ({w}): {Path(f).parent.name}" for f, w in skipped[:6])), flush=True)
    return keep
