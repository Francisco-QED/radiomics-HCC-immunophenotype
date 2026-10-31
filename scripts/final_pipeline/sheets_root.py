"""Where the radiomics sheets (feature tables + inter-reader sheets for the ICC) are read from.
Default = <project>/Results/Radiomics_sheets (intratumoral features, per-sequence bin widths). Environment variables:
  RADIOMICS_SHEETS_DIR  another folder with the same layout (e.g. .../Radiomics_sheets/Annuli_v2)
  RADIOMICS_BINWIDTH    perseq (default) | 5 | 20: which extraction variant (sheet suffix W-<value>) the pipeline reads, for features AND the inter-reader ICC
The choice is recorded in each run manifest."""
import os
from pathlib import Path


def sheets_dir(root):
    return Path(os.environ.get("RADIOMICS_SHEETS_DIR") or Path(root) / "Results/Radiomics_sheets")


def binwidth():
    return os.environ.get("RADIOMICS_BINWIDTH") or "perseq"


def combined_sheet(root, sheet):
    return sheets_dir(root) / (f"SA1_2_post/R-SA1_2-post_I-sitkBSpline_S-1mm_W-{binwidth()}.csv" if sheet == "post" else f"SA1_2_pre/R-SA1_2_I-sitkBSpline_S-1mm_W-{binwidth()}.csv")


def reader_sheets(root):
    return [str(sheets_dir(root) / f"inter-reader/Reader{i}/R-Reader{i}_I-sitkBSpline_S-1mm_W-{binwidth()}.csv") for i in (1, 2)]
