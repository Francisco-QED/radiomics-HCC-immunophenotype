"""Audit v3/v4 geometry from the Slicer QC logs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask-root", required=True)
    ap.add_argument("--variant", choices=["v3", "v4"], required=True)
    args = ap.parse_args()
    root = Path(args.mask_root).expanduser().absolute()
    files = sorted(root.glob("qc_peritumor_*.csv"))
    if len(files) != 3:
        raise RuntimeError(f"Expected three QC logs (main, Reader1, Reader2), found {files}")
    frames = []
    for path in files:
        d = pd.read_csv(path)
        d.insert(0, "qc_file", path.name)
        frames.append(d)
    data = pd.concat(frames, ignore_index=True)
    ok = data[data.status == "OK"].copy()
    if ok.empty:
        raise RuntimeError("No successful masks found")
    numeric = ["tumor_ml", "inner_ml", "outer_ml", "annulus_ml", "inner_tumor_frac", "inner_is_entire_tumor", "annulus_tumor_overlap_voxels"]
    for col in numeric:
        ok[col] = pd.to_numeric(ok[col], errors="coerce")
    if ok[numeric].isna().any().any():
        raise RuntimeError("Missing geometry fields in successful QC rows")
    if args.variant == "v3":
        invalid = ok[ok.outer_ml.abs() > 1e-9]
        ok["geometry_flag"] = ""
        ok.loc[ok.inner_tumor_frac >= 0.95, "geometry_flag"] = "inner_rim_ge95pct_tumor"
        ok.loc[ok.inner_is_entire_tumor == 1, "geometry_flag"] = "inner_rim_is_entire_tumor"
    else:
        invalid = ok[(ok.inner_ml.abs() > 1e-9) | (ok.annulus_tumor_overlap_voxels != 0)]
        ok["geometry_flag"] = ""
    if not invalid.empty:
        invalid.to_csv(root / "INVALID_GEOMETRY.csv", index=False)
        raise RuntimeError(f"{len(invalid)} masks violate {args.variant} geometry; see INVALID_GEOMETRY.csv")
    ok.to_csv(root / "geometry_audit.csv", index=False)
    summary = {
        "variant": args.variant,
        "successful_masks": int(len(ok)),
        "skipped_masks": int((data.status == "SKIPPED").sum()),
        "error_masks": int((data.status == "ERROR").sum()),
        "entire_tumor_inner_masks": int((ok.inner_is_entire_tumor == 1).sum()),
        "inner_ge95pct_tumor_masks": int((ok.inner_tumor_frac >= 0.95).sum()),
        "tumor_overlap_violations": int((ok.annulus_tumor_overlap_voxels != 0).sum()) if args.variant == "v4" else None,
    }
    (root / "geometry_audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if summary["error_masks"]:
        raise RuntimeError("Slicer reported mask-generation errors")


if __name__ == "__main__":
    main()
