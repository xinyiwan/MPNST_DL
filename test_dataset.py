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
# with open(args.config) as f:
#         cfg = yaml.load(f, Loader=yaml.FullLoader)
# data = MPNSTDataMoule(
#         batch_size = int(cfg['dataset']['batch_size']),
#         pixdim = tuple(cfg['transform']['pixdim']),
#         spatial_size = tuple(cfg['transform']['spatial_size']),
#         fold = 0,
#         mri_type =  cfg['dataset']['task_name'],
#     )
# data.prepare_data()
# data.setup()



# Test MPNSTDataset
mri_type = 'T1'
fold = 0
data_csv = pd.read_csv(f"/trinity/home/xwan/MPNST_DL/input/{mri_type}/train_{mri_type}.csv")
train_df = data_csv[data_csv.fold != fold].reset_index(drop=False)
train_img_paths = [f'/trinity/home/xwan/data/MPNST/{pid}/{mri_type}.nii.gz' for pid in train_df['Patient']]
train_seg_paths = [f'/trinity/home/xwan/data/MPNST/{pid}/segmentations.nii.gz' for pid in train_df['Patient']]
train_labels = train_df['MPNST'].tolist()
train_ids = train_df['Patient'].tolist()
train_set = [{"image": img, "seg": seg, "label": label, "case_id": pid} for img, seg, label, pid in zip(train_img_paths, train_seg_paths, train_labels, train_ids)]

a = MPNSTDataset(data=train_set, mri_type=mri_type, transform=None)
