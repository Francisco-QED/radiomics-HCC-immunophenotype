import os
import yaml
import logging
import itertools
from tqdm import tqdm
import pandas as pd
from radiomics import featureextractor
import SimpleITK as sitk


img_path = "/Volumes/T7 Shield/Restrepo/HCC_Retrospective_radiomics/Total_segmentations/SA1/P100/P100_ap_N4.nrrd"
mask_path = "/Volumes/T7 Shield/Restrepo/HCC_Retrospective_radiomics/Total_segmentations/SA1/P100/P100_ap.seg.nrrd"
img = sitk.ReadImage(img_path)
mask = sitk.ReadImage(mask_path)
print(img.GetSpacing() == mask.GetSpacing())
print(img.GetOrigin() == mask.GetOrigin())
print(img.GetDirection() == mask.GetDirection())

