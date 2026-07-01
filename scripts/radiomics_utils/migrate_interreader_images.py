"""
Restructures Inter-reader folder to separate shared images from reader segmentations.

Before:
  Inter-reader/Reader1/PID/  -> raw .nrrd + _N4.nrrd + .seg.nrrd
  Inter-reader/Reader2/PID/  -> _N4.nrrd + .seg.nrrd  (no raw)

After:
  Inter-reader/images/PID/   -> raw .nrrd + _N4.nrrd  (moved from Reader1)
  Inter-reader/Reader1/PID/  -> .seg.nrrd only
  Inter-reader/Reader2/PID/  -> .seg.nrrd only  (_N4.nrrd deleted as duplicates)

Usage:
  python3 migrate_interreader_images.py           # dry run (no changes)
  python3 migrate_interreader_images.py --execute  # actually move/delete files
"""

import argparse
import os
import shutil

INTER_READER = "/Volumes/T7 Shield/Restrepo/HCC_Retrospective_radiomics/Total_segmentations/Inter-reader"
READER1 = os.path.join(INTER_READER, "Reader1")
READER2 = os.path.join(INTER_READER, "Reader2")
IMAGES  = os.path.join(INTER_READER, "images")


def main(execute):
    mode = "EXECUTE" if execute else "DRY RUN"
    print(f"=== migrate_interreader_images.py [{mode}] ===\n")

    patients = sorted(p for p in os.listdir(READER1) if os.path.isdir(os.path.join(READER1, p)))
    print(f"Found {len(patients)} patients in Reader1: {patients}\n")

    moves, deletes = 0, 0

    for pid in patients:
        r1_dir = os.path.join(READER1, pid)
        r2_dir = os.path.join(READER2, pid)
        img_dir = os.path.join(IMAGES, pid)

        files_r1 = sorted(os.listdir(r1_dir))
        # Skip macOS resource-fork shadow files (._*) — they're filesystem artefacts, not images.
        image_files = [f for f in files_r1
                       if not f.endswith('.seg.nrrd') and not f.startswith('._')]
        seg_files_r1 = [f for f in files_r1 if f.endswith('.seg.nrrd')]

        print(f"[{pid}]")
        print(f"  Move {len(image_files)} image files from Reader1 → images/")
        for f in image_files:
            src = os.path.join(r1_dir, f)
            dst = os.path.join(img_dir, f)
            print(f"    MOVE  {f}")
            if execute:
                os.makedirs(img_dir, exist_ok=True)
                shutil.move(src, dst)
            moves += 1

        # Delete _N4.nrrd duplicates from Reader2 (raw .nrrd not present there anyway)
        if os.path.isdir(r2_dir):
            files_r2 = sorted(os.listdir(r2_dir))
            r2_to_delete = [f for f in files_r2
                            if not f.endswith('.seg.nrrd') and not f.startswith('._')]
            print(f"  Delete {len(r2_to_delete)} duplicate image files from Reader2/")
            for f in r2_to_delete:
                path = os.path.join(r2_dir, f)
                print(f"    DELETE  {f}")
                if execute:
                    os.remove(path)
                deletes += 1
        else:
            print(f"  ⚠️  Reader2/{pid} not found — skipping Reader2 cleanup")

        print()

    print(f"Summary: {moves} moves, {deletes} deletes")
    if not execute:
        print("\nThis was a dry run. Re-run with --execute to apply changes.")
    else:
        print("\nDone.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true',
                        help='Actually move/delete files (default is dry run)')
    args = parser.parse_args()
    main(args.execute)
