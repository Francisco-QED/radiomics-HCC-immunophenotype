"""Peritumoral-annulus radiomics: build annuli from the tumor ROIs and extract features exactly as the intratumoral sheets.

Annulus (per axial slice, in-plane physical distances): tumor-ROI voxels within `--in-mm` of the ROI edge (inner rim) plus the voxels outside the ROI within
`--out-mm` of its edge (outer rim); slices without tumor stay empty. No liver clipping. Label 1 of the .seg.nrrd is the tumor (same label the intratumoral
extraction uses). Mask geometry (origin, spacing, direction) is copied from the ROI so PyRadiomics sees the same grid as before.

Extraction = radiomics_utils/extraction.py settings (radiomics_params.yaml): all image types, default feature classes, sitkBSpline, 1 mm, per-sequence bin widths,
N4 images except adc / dwi_b1 / dwi_b2. Output columns '<seq>_<pyradiomics key>' like the existing sheets; rows = PatientID.

    python annuli_extraction.py --target SA1 --out-root <.../Radiomics_sheets/Annuli> --jobs 4 [--pids P001 ...] [--limit-seq pvp]
Targets: SA1, SA2-post (Total_segmentations/<target>), Reader1, Reader2 (Total_segmentations/Inter-reader/<reader>, images in Inter-reader/images).
"""
import argparse, logging, os, sys, time
from multiprocessing import Pool
from pathlib import Path
sys.dont_write_bytecode = True
import numpy as np, pandas as pd, yaml
import SimpleITK as sitk
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent
SEG = Path(os.environ.get("RADIOMICS_SEG_ROOT", "/Volumes/T7 Shield/Restrepo/HCC_Retrospective_radiomics/Total_segmentations"))
SUPPLIED = Path(os.environ.get("RADIOMICS_ANNULUS_MASK_ROOT", "/Volumes/LaCie2/peritumor_all_2026-10-26"))
logging.getLogger("radiomics").setLevel(logging.ERROR)


def annulus_array(tumor, spacing_xyz, out_mm, in_mm):
    """tumor: bool array (z, y, x). Slice-wise ring: inner rim (<= in_mm inside the edge) + outer rim (<= out_mm outside)."""
    sy, sx = spacing_xyz[1], spacing_xyz[0]; ring = np.zeros_like(tumor, dtype=np.uint8)
    for k in np.flatnonzero(tumor.any((1, 2))):
        s = tumor[k]; d_out = ndi.distance_transform_edt(~s, sampling=(sy, sx)); d_in = ndi.distance_transform_edt(s, sampling=(sy, sx))
        ring[k] = ((~s) & (d_out <= out_mm)) | (s & (d_in <= in_mm))
    return ring


def build_annulus(seg_path, out_mm, in_mm):
    img = sitk.ReadImage(str(seg_path)); arr = sitk.GetArrayFromImage(img)
    if arr.ndim != 3: raise ValueError(f"unexpected mask dimension {arr.shape}")
    ring = annulus_array(arr == 1, img.GetSpacing(), out_mm, in_mm); out = sitk.GetImageFromArray(ring); out.CopyInformation(img)
    return out, int((arr == 1).sum()), int(ring.sum())


def paths(target, pid, seq, no_n4):
    suffix = "" if seq in no_n4 else "_N4"
    if target.startswith("Reader"): return SEG / "Inter-reader/images" / pid / f"{pid}_{seq}{suffix}.nrrd", SEG / "Inter-reader" / target / pid / f"{pid}_{seq}.seg.nrrd"
    return SEG / target / pid / f"{pid}_{seq}{suffix}.nrrd", SEG / target / pid / f"{pid}_{seq}.seg.nrrd"


def work(task):
    target, pid, seq, cfg, out_mm, in_mm, mask_dir, source, mask_root, bw = task
    from radiomics import featureextractor
    ex, es = cfg["execution"], cfg["extraction_settings"].copy()
    img_p, seg_p = paths(target, pid, seq, set(ex["sequences_without_n4"]))
    if not (img_p.exists() and seg_p.exists()): return pid, seq, None, "missing image or mask"
    try:
        if source == "supplied":      # prefer target-specific generated masks; retain the legacy Reader1->SA1 fallback only when needed
            preferred = Path(mask_root) / target.replace("-", "_")
            legacy = Path(mask_root) / "SA1"
            base = preferred if preferred.is_dir() else legacy if target == "Reader1" and legacy.is_dir() else preferred
            ring = sitk.ReadImage(str(base / pid / f"{pid}_{seq}_annulus.seg.nrrd")); n_tum, n_ring = 0, int((sitk.GetArrayFromImage(ring) > 0).sum())
            ring = sitk.Cast(ring > 0, sitk.sitkUInt8)
        else: ring, n_tum, n_ring = build_annulus(seg_p, out_mm, in_mm)
        if n_ring == 0: return pid, seq, None, "empty annulus (no label-1 tumor)"
        if mask_dir is not None:
            Path(mask_dir, pid).mkdir(parents=True, exist_ok=True); sitk.WriteImage(ring, str(Path(mask_dir, pid, f"{pid}_{seq}.seg.nrrd")), useCompression=True)
        es.update({"interpolator": ex["interpolators"][0], "resampledPixelSpacing": ex["spacings"][0], "binWidth": (ex["per_sequence_bin_widths"].get(seq, 25) if bw == "perseq" else int(bw))})
        extractor = featureextractor.RadiomicsFeatureExtractor({"imageType": cfg.get("imageType", {}), "setting": es})
        res = extractor.execute(sitk.ReadImage(str(img_p)), ring)
        return pid, seq, {f"{seq}_{k}": v for k, v in res.items() if not k.startswith("diagnostics_")}, f"tumor {n_tum} -> annulus {n_ring} voxels"
    except Exception as e:
        return pid, seq, None, f"error: {e}"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--target", required=True, choices=["SA1", "SA2", "SA2-post", "Reader1", "Reader2"]); ap.add_argument("--out-root", required=True)
    ap.add_argument("--jobs", type=int, default=4); ap.add_argument("--pids", nargs="*", default=None); ap.add_argument("--limit-seq", nargs="*", default=None)
    ap.add_argument("--out-mm", type=float, default=5.0); ap.add_argument("--in-mm", type=float, default=2.0); ap.add_argument("--save-masks", action="store_true"); ap.add_argument("--binwidth", default="perseq", help="perseq | 5 | 20 (fixed bin width in all sequences)"); ap.add_argument("--mask-source", choices=["own", "supplied"], default="own"); ap.add_argument("--mask-root", default=str(SUPPLIED), help="folder with <cohort or reader>/<pid>/<pid>_<seq>_annulus.seg.nrrd for --mask-source supplied"); ap.add_argument("--out-subdir", default=None, help="override output sub-folder under --out-root")
    ap.add_argument("--params", default=str(HERE.parent / "radiomics_params.yaml")); a = ap.parse_args()
    cfg = yaml.safe_load(open(a.params)); seqs = a.limit_seq or cfg["execution"]["sequences"]
    root = SEG / "Inter-reader" / a.target if a.target.startswith("Reader") else SEG / a.target
    pids = a.pids or sorted(p for p in os.listdir(root) if p.startswith("P") and (root / p).is_dir())
    name = {"SA1": "R-SA1", "SA2": "R-SA2", "SA2-post": "R-SA2-post"}.get(a.target, f"R-{a.target}") + f"_I-sitkBSpline_S-1mm_W-{a.binwidth}.csv"
    out_dir = Path(a.out_root) / (a.out_subdir or ("inter-reader/" + a.target if a.target.startswith("Reader") else a.target.replace("-", "_") if a.target != "SA2" else "SA2_pre")); out_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = (Path(a.out_root) / "masks" / a.target) if a.save_masks else None
    tasks = [(a.target, p, s, cfg, a.out_mm, a.in_mm, mask_dir, a.mask_source, a.mask_root, a.binwidth) for p in pids for s in seqs]; print(f"{a.target}: {len(pids)} patients x {len(seqs)} sequences = {len(tasks)} extractions, annulus {a.out_mm} mm out / {a.in_mm} mm in", flush=True)
    rows, log, t0 = {p: {"PatientID": p} for p in pids}, [], time.time()
    with Pool(a.jobs, maxtasksperchild=25) as pool:
        for n, (pid, seq, feats, msg) in enumerate(pool.imap_unordered(work, tasks, chunksize=1), 1):
            if feats: rows[pid].update(feats)
            else: log.append((pid, seq, msg))
            if n % 20 == 0 or n == len(tasks): print(f"  {n}/{len(tasks)} done, {time.time() - t0:.0f}s, {len(log)} skipped", flush=True)
    df = pd.DataFrame([r for r in rows.values() if len(r) > 1]); order = ["PatientID"] + sorted([c for c in df.columns if c != "PatientID"], key=lambda c: (seqs.index(c.split("_")[0]) if c.split("_")[0] in seqs else 99, c))
    df = df[order].sort_values("PatientID"); dest = out_dir / name
    if a.pids or a.limit_seq: dest = out_dir / ("PARTIAL_" + name)
    df.to_csv(dest, index=False); pd.DataFrame(log, columns=["PatientID", "sequence", "reason"]).to_csv(out_dir / (dest.stem + "_skipped.csv"), index=False)
    print(f"saved {dest} {df.shape}; skipped {len(log)}; {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
