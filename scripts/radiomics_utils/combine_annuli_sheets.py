"""Validate and combine SA1 and SA2-post annulus sheets without altering inputs."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheets-root", required=True)
    ap.add_argument("--binwidth", default="perseq")
    args = ap.parse_args()
    root = Path(args.sheets_root).expanduser().absolute()
    sa1_path = root / "SA1" / f"R-SA1_I-sitkBSpline_S-1mm_W-{args.binwidth}.csv"
    sa2_path = root / "SA2_post" / f"R-SA2-post_I-sitkBSpline_S-1mm_W-{args.binwidth}.csv"
    sa1 = pd.read_csv(sa1_path)
    sa2 = pd.read_csv(sa2_path)
    if list(sa1.columns) != list(sa2.columns):
        raise RuntimeError("SA1 and SA2-post sheet schemas differ")
    if sa1.PatientID.duplicated().any() or sa2.PatientID.duplicated().any():
        raise RuntimeError("Duplicate PatientID within an input sheet")
    overlap = set(sa1.PatientID.astype(str)) & set(sa2.PatientID.astype(str))
    if overlap:
        raise RuntimeError(f"Patient overlap between cohorts: {sorted(overlap)}")
    if (len(sa1), len(sa2)) != (111, 30):
        raise RuntimeError(f"Expected 111 SA1 and 30 SA2-post rows, got {len(sa1)} and {len(sa2)}")
    combined = pd.concat([sa1, sa2], ignore_index=True)
    dest_dir = root / "SA1_2_post"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"R-SA1_2-post_I-sitkBSpline_S-1mm_W-{args.binwidth}.csv"
    if dest.exists():
        raise FileExistsError(f"Refusing to overwrite {dest}")
    combined.to_csv(dest, index=False)
    print(f"saved {dest} {combined.shape}")


if __name__ == "__main__":
    main()
