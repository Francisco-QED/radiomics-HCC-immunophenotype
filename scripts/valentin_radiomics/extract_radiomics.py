## System modules
import os, sys
import numpy as np
import pandas as pd
# import radiomics 
from radiomics import featureextractor, getTestCase
import pdb
import matplotlib.pyplot as plt

sys.path.append('/home/fauvev01/BMEII-Functions/')

## Custom modules
import bmeii_io as bio

## Inputs
pat = 'Perez-Corrales.65'
session = 'post treatment'
series = 'DWI peri-lesion'
file = '1 No series description_1'

idir_im = '/home/fauvev01/Bachir/Radiomics/data/liver_seg/'+pat+'/'+session+'/'+series+'/'+file+'.nrrd'
idir_seg = '/home/fauvev01/Bachir/Radiomics/data/liver_seg/'+pat+'/'+session+'/'+series+'/NewSegmentation.nrrd'
# idir_seg = '/home/fauvev01/Bachir/Radiomics/data/bowel_seg/'+pat+'/'+series+'/NewSegmentation.nrrd'

# idir_im = '/home/fauvev01/Bachir/Radiomics/data/liver_seg/Barbato.57/pre-treatment/HBP peri-lesion/47 T1 VIBE POST-CAIPI 20 min_SUB.nrrd'
# idir_seg = '/home/fauvev01/Bachir/Radiomics/data/liver_seg/Barbato.57/pre-treatment/HBP peri-lesion/Segmentation.seg.nrrd'

pdb.set_trace()


## Functions
def imroisave(im,roi,odir):
	im_temp = im
	seg_temp = roi

	im_out = np.stack([im_temp]*3, axis=2)

	im_out[:,:,1] = im_out[:,:,1] * (seg_temp == 0).astype(int)
	im_out[:,:,2] = im_out[:,:,2] * (seg_temp == 0).astype(int)
	im_out[:,:,0] = im_out[:,:,0] * (seg_temp == 0).astype(int) + seg_temp * 255

	im_out = (im_out - np.min(im_out))/(np.max(im_out)-np.min(im_out))

	plt.imsave(odir, im_out, cmap='gray')

## Main Code

##############
#### Radiomics 
##############

# Instantiate the extractor
extractor = featureextractor.RadiomicsFeatureExtractor()

result = extractor.execute(idir_im, idir_seg)


out = []
headers = []

for i in result.keys():
	out.append([result[i]])
	headers.append(i)


# odir = '/home/fauvev01/Bachir/Radiomics/output/radiomics/liver/'+pat+'__'+session+'__'+series+'__'+file+'.csv'

odir = '/home/fauvev01/Bachir/Radiomics/output/radiomics/liver/'+pat+'/'+session+'/'+series+'/'+pat+'__'+session+'__'+series+'__'+file+'.csv'

# pdb.set_trace()

bio.write_csv(odir, np.array(out).T.tolist(), headers)

##############
# Slice by slice inspection
##############

# im = bio.read_file(idir_im)
# seg = bio.read_file(idir_seg)

# for i in range(np.shape(im)[2]):

# 	im_temp = im[:,:,i]
# 	seg_temp = seg[:,:,i]

# 	odir = '/home/fauvev01/Bachir/Radiomics/output/image_inspection/test_'+str(i).zfill(3)+'.png'
	
# 	imroisave(im_temp, seg_temp, odir)

# 	# if len(np.unique(seg_temp)) != 1:
# 	# 	pdb.set_trace()


print('>>> DONE <<<')

# pdb.set_trace()

