import sys
sys.path.insert(1, '/trinity/home/xwan/MPNST_DL/working')
from dataset import MPNSTDataMoule, MPNSTDataset
import yaml
import argparse
from lightning.pytorch.loggers import NeptuneLogger
import pandas as pd
import numpy as np


mri_type = 'T1'
data_csv = pd.read_csv(f"/trinity/home/xwan/MPNST_DL/input/{mri_type}/train_{mri_type}.csv")
img_paths = [f'/trinity/home/xwan/data/MPNST/{pid}/{mri_type}.nii.gz' for pid in data_csv['Patient']]
seg_paths = [f'/trinity/home/xwan/data/MPNST/{pid}/segmentations.nii.gz' for pid in data_csv['Patient']]
labels = data_csv['MPNST'].tolist()
ids = data_csv['Patient'].tolist()
set = [{"image": img, "seg": seg, "label": label, "case_id": pid} for img, seg, label, pid in zip(img_paths, seg_paths,labels,ids)]
pixdim = [0.6, 0.6, 5]
spatial_size = ()
data = MPNSTDataset(data=set, mri_type=mri_type, pixdim=pixdim, spatial_size=spatial_size, transform=None)
roi_coodinates = data.get_roi()

v_m = 0
v_id = ''

# find biggest volume
bbxs = []
for case_id in roi_coodinates.keys():
    a, b, c, d, e, f = roi_coodinates[case_id]
    if case_id != 'MPNSTRad-012_1':
        l, w, h = b-a, d-c, f-e
    else:
        l, w, h = b-a, f-e, d-c
    bbxs.append(np.array([l,w,h]))
    print(f"bbx size = {l, w, h}")
    v = (b-a)*(d-c)*(f-e)
    if v > v_m:
        v_m = v      
        v_id = case_id   
print(f"biggest volume is from {v_id}")
bbxs = np.array(bbxs)
# print(bbxs)
print(f"medium values of bbxs: {np.median(bbxs,-2)}")
print(f"max values of bbxs: {np.max(bbxs,-2)}")
print(f"min values of bbxs: {np.min(bbxs,-2)}")
print(f"mean values of bbxs: {np.mean(bbxs,-2)}")


# find spacing 

pixdims = data.get_pixdim()
reso = []
for case_id in pixdims.keys():
    x, y, z =  pixdims[case_id]
    if case_id != 'MPNSTRad-012_1':
        l, w, h = x, y, z
    else:
        l, w, h = x, z, y
    reso.append(np.array([l,w,h]))
reso = np.array(reso)
print(f"medium values of reso: {np.median(reso,-2)}")
print(f"max values of reso: {np.max(reso,-2)}")
print(f"mean values of reso: {np.mean(reso,-2)}")





