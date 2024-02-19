from working.dataset import MPNSTDataMoule, MPNSTDataset
import yaml
import argparse
from lightning.pytorch.loggers import NeptuneLogger
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--fold', type=int)
    return parser.parse_args()

args = parse_args()
with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
data = MPNSTDataMoule(
        batch_size = int(cfg['dataset']['batch_size']),
        pixdim = tuple(cfg['transform']['pixdim']),
        spatial_size = tuple(cfg['transform']['spatial_size']),
        fold = 0,
        mri_type =  cfg['dataset']['task_name'],
    )
data.prepare_data()
data.setup()



# Test MPNSTDataset
# mri_type = 'T1'
# fold = 0
# data_csv = pd.read_csv(f"/trinity/home/xwan/MPNST_DL/input/{mri_type}/train_{mri_type}.csv")
# train_df = data_csv[data_csv.fold != fold].reset_index(drop=False)
# train_img_paths = [f'/trinity/home/xwan/data/MPNST/{pid}/{mri_type}.nii.gz' for pid in train_df['Patient']]
# train_seg_paths = [f'/trinity/home/xwan/data/MPNST/{pid}/segmentations.nii.gz' for pid in train_df['Patient']]
# train_labels = train_df['MPNST'].tolist()
# train_ids = train_df['Patient'].tolist()
# train_set = [{"image": img, "seg": seg, "label": label, "case_id": pid} for img, seg, label, pid in zip(train_img_paths, train_seg_paths, train_labels, train_ids)]

mri_type = 'T1'
data_csv = pd.read_csv(f"/trinity/home/xwan/MPNST_DL/input/{mri_type}/train_{mri_type}.csv")
img_paths = [f'/trinity/home/xwan/data/MPNST/{pid}/{mri_type}.nii.gz' for pid in data_csv['Patient']]
seg_paths = [f'/trinity/home/xwan/data/MPNST/{pid}/segmentations.nii.gz' for pid in data_csv['Patient']]
labels = data_csv['MPNST'].tolist()
ids = data_csv['Patient'].tolist()
set = [{"image": img, "seg": seg, "label": label, "case_id": pid} for img, seg, label, pid in zip(img_paths, seg_paths,labels,ids)]
a = MPNSTDataset(data=set, mri_type=mri_type, transform=None)
a.prepare_pixdim()
array([0.6116878, 0.6129556, 4.8543463], dtype=float32)

# a = MPNSTDataset(data=train_set, mri_type=mri_type, transform=None)


with open('/trinity/home/xwan/MPNST_DL/input/configs/config015.yaml') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
data = MPNSTDataMoule(
        batch_size = int(4),
        pixdim = tuple(cfg['transform']['pixdim']),
        spatial_size = tuple(cfg['transform']['spatial_size']),
        fold = 0,
        mri_type =  cfg['dataset']['task_name'],
    )
data.prepare_data()
data.setup()
train = data.train_dataloader()
# img, label, id =  next(iter(train))


# v_m = 0
# v_id = ''
# for case_id in roi_coodinates.keys():
#     a, b, c, d, e, f = roi_coodinates[case_id]
#     v = (b-a)*(d-c)*(f-e)
#     if v > v_m:
#         v_m = v      
#         v_id = case_id   

# [27, 322, 88, 185, 77, 421]
# (295, 97, 344)

# for i in img_paths:
#     img_f = nib.load(i).get_fdata()
#     print(img_f.shape)


# train_trans = Compose([
#                 Spacingd(
#                     keys = keys,
#                     pixdim = [0.6,0.6,5],
#                     mode = ("bilinear", "nearest")
#                     ),
#                 Orientationd(keys = keys, axcodes="PLI"),
#                 NormalizeIntensityd(keys = 'image', nonzero = False, channel_wise = True),])

train_trans = Compose([
#                 Orientationd(keys = keys, axcodes="LIP"),
#                 ScaleIntensityd(keys = 'image', channel_wise = True),])

# a = MPNSTDataset(data=set, mri_type=mri_type, transform=train_trans, use_roi=False)
