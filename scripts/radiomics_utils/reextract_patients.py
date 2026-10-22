"""Re-extract radiomics for a few patients and splice their rows into an existing sheet.

Use after fixing images/segmentations/N4 for specific patients, instead of re-running the whole cohort.
Uses the same settings, per-sequence bin widths, N4 rule and extract_features_single() as extraction.py,
so a re-extracted row is identical in method to the rest of the sheet.

    python reextract_patients.py --cohort SA1 --pids P002 P013 P066 P101 --binwidth perseq \
        --sheet "<.../Radiomics_sheets/SA1/R-SA1_I-sitkBSpline_S-1mm_W-perseq.csv>" [--dry-run]

The patient's whole row is REPLACED (sequences with no image/mask become NaN, so a removed or relabeled
sequence cannot leave stale values behind). A timestamped .bak copy of the sheet is written first.
"""
import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from extraction import extract_features_single  # noqa: E402


def extract_patient(pid, cohort_dir, config, bw):
    exec_cfg = config["execution"]
    no_n4 = set(exec_cfg.get("sequences_without_n4", []))
    interp, space = exec_cfg["interpolators"][0], exec_cfg["spacings"][0]
    settings = config.get("extraction_settings", {}).copy()
    settings.update({"interpolator": interp, "resampledPixelSpacing": space, "binWidth": bw})
    perseq = exec_cfg.get("per_sequence_bin_widths", {}) if bw == "perseq" else {}
    row = {}
    for seq in exec_cfg["sequences"]:
        s = settings.copy()
        if perseq:
            s["binWidth"] = perseq.get(seq, 25)
        suffix = "" if seq in no_n4 else "_N4"
        img = os.path.join(cohort_dir, pid, f"{pid}_{seq}{suffix}.nrrd")
        msk = os.path.join(cohort_dir, pid, f"{pid}_{seq}.seg.nrrd")
        if not (os.path.exists(img) and os.path.exists(msk)):
            print(f"    {pid} {seq}: image or mask missing -> NaN", flush=True)
            continue
        feats = extract_features_single(img, msk, {"imageType": config.get("imageType", {}), "setting": s})
        if feats:
            row.update({f"{seq}_{k}": v for k, v in feats.items()})
            print(f"    {pid} {seq}: {len(feats)} features", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, help="folder under Total_segmentations (SA1, SA2, SA2-post)")
    ap.add_argument("--pids", nargs="*", default=[], help="patients to (re-)extract")
    ap.add_argument("--allow-new", action="store_true", help="permit --pids that are not yet in the sheet (adds rows)")
    ap.add_argument("--drop", nargs="*", default=[], help="patients to REMOVE from the sheet (e.g. moved to another cohort)")
    ap.add_argument("--binwidth", required=True, help="perseq | 5 | 20")
    ap.add_argument("--sheet", required=True, help="existing CSV to update")
    ap.add_argument("--base-dir", default="/Volumes/T7 Shield/Restrepo/HCC_Retrospective_radiomics/Total_segmentations")
    ap.add_argument("--params", default=str(HERE.parent / "radiomics_params.yaml"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    config = yaml.safe_load(open(a.params))
    bw = a.binwidth if a.binwidth == "perseq" else int(a.binwidth)
    sheet = pd.read_csv(a.sheet, index_col=0)
    unknown = [p for p in a.pids if p not in sheet.index]
    if unknown and not a.allow_new:
        raise SystemExit(f"Patients not in sheet (pass --allow-new to add rows): {unknown}")
    missing_drop = [p for p in a.drop if p not in sheet.index]
    if missing_drop:
        raise SystemExit(f"--drop patients not in sheet: {missing_drop}")
    print(f"sheet {a.sheet}\n  {sheet.shape}, extracting {a.pids} (new: {unknown}), dropping {a.drop} (binWidth={bw})")
    if a.dry_run:
        return
    new_rows = {}
    for pid in a.pids:
        t = time.time()
        print(f"  {pid}", flush=True)
        new_rows[pid] = extract_patient(pid, os.path.join(a.base_dir, a.cohort), config, bw)
        print(f"  {pid} done in {time.time() - t:.0f}s", flush=True)
    unexpected = {c for r in new_rows.values() for c in r} - set(sheet.columns)
    if unexpected:
        raise SystemExit(f"{len(unexpected)} extracted columns are not in the sheet (e.g. {sorted(unexpected)[:3]}); aborting")
    bak = f"{a.sheet}.bak_{time.strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(a.sheet, bak)
    sheet = sheet.drop(index=a.drop)
    for pid, r in new_rows.items():
        sheet.loc[pid] = pd.Series(r).reindex(sheet.columns)
    sheet = sheet.sort_index()
    sheet.index.name = "PatientID"
    sheet.to_csv(a.sheet)
    print(f"updated {len(new_rows)} rows, dropped {len(a.drop)}; now {sheet.shape}; backup: {bak}")


if __name__ == "__main__":
    main()
