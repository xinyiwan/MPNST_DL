import os, glob
import pandas as pd
import pytorch_lightning as pl
import torch
import joblib
from tqdm import tqdm
from monai.data import Dataset
from utilities import extract_3d_bbx
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from monai.transforms import (
    Compose,
    Resized,
    LoadImaged,
    Orientationd,
    RandGaussianNoised,
    ScaleIntensityd,
    Spacingd,
    ResizeWithPadOrCropd,
    RandSpatialCropSamplesd,
    RandAdjustContrastd,
    RandRotated,
    RandFlipd,
    RandZoomd,
    MapTransform,
    RandHistogramShiftd,
    RandRotate90d,
)
import nibabel as nib
import numpy as np



class MPNSTDataMoule(pl.LightningDataModule):
    def __init__(self, batch_size, pixdim, spatial_size, fold, mri_type, if_sampler=True, if_use_roi=False):
        super().__init__()
        self.batch_size = batch_size
        self.pixdim = pixdim
        self.spatial_size = spatial_size
        self.fold = fold
        self.mri_type = mri_type
        self.if_sampler = if_sampler
        self.if_use_roi = if_use_roi
        self.data_csv = pd.read_csv(f"/trinity/home/xwan/MPNST_DL/input/{self.mri_type}/train_{self.mri_type}.csv")
        self.train_trans = None
        self.val_trans = None
        self.train_set = None
        self.val_set = None
        self.sampler = None

    def prepare_data(self):
        # Load labels
        train_df = self.data_csv[self.data_csv.fold != self.fold].reset_index(drop=False)
        train_img_paths = [f'/trinity/home/xwan/data/MPNST/{pid}/{self.mri_type}.nii.gz' for pid in train_df['Patient']]
        train_seg_paths = [f'/trinity/home/xwan/data/MPNST/{pid}/segmentations.nii.gz' for pid in train_df['Patient']]
        train_labels = train_df['MPNST'].tolist()
        train_ids = train_df['Patient'].tolist()

        val_df = self.data_csv[self.data_csv.fold == self.fold].reset_index(drop=False)
        val_img_paths = [f'/trinity/home/xwan/data/MPNST/{pid}/{self.mri_type}.nii.gz' for pid in val_df['Patient']]
        val_seg_paths = [f'/trinity/home/xwan/data/MPNST/{pid}/segmentations.nii.gz' for pid in val_df['Patient']]
        val_labels = val_df['MPNST'].tolist()
        val_ids = train_df['Patient'].tolist()
    
        self.train_set = [{"image": img, "seg": seg, "label": label, "case_id": pid} for img, seg, label, pid in zip(train_img_paths, train_seg_paths, train_labels, train_ids)]
        self.val_set = [{"image": img, "seg": seg, "label": label, "case_id": pid} for img, seg, label, pid in zip(val_img_paths, val_seg_paths, val_labels, val_ids)]
        

    def build_transform(self, keys="image"):
            train_trans = Compose([
                Spacingd(
                    keys = keys,
                    pixdim = self.pixdim,
                    mode = ("bilinear")
                    ),
                Resized(
                    keys = keys,
                    spatial_size=self.spatial_size
                ),
                RandGaussianNoised(keys = keys,
                                   prob=0.3, 
                                   mean=0.0, 
                                   std=0.1),
                RandAdjustContrastd(keys = keys,
                                    prob=0.5),
                RandHistogramShiftd(keys = keys,
                                    prob=0.5),
                ScaleIntensityd(keys = keys),
                RandZoomd(keys = keys,
                          prob=0.3,
                          min_zoom=0.9, 
                          max_zoom=1.1),
                RandRotated(prob=0.5,
                            range_x = 0.4,
                            range_y = 0.4,
                            range_z = 0.4,
                            keys = keys),
                RandFlipd(keys = keys,
                          prob=0.1, spatial_axis=1),
            ])  
            val_trans = Compose([
                Spacingd(
                    keys = keys,
                    pixdim = self.pixdim,
                    mode = ("bilinear")
                    ),
                Resized(
                    keys = keys,
                    spatial_size=self.spatial_size
                ),
                ScaleIntensityd(keys = keys),
            ]) 
            return train_trans, val_trans
    
    def setup(self, stage = None):

        self.train_trans, self.val_trans = self.build_transform()
        self.train_set = MPNSTDataset(data=self.train_set, mri_type=self.mri_type, transform=self.train_trans, use_roi=self.if_use_roi)
        self.val_set = MPNSTDataset(data=self.val_set, mri_type=self.mri_type, transform=self.val_trans, use_roi=self.if_use_roi)

    def train_dataloader(self):
        if self.if_sampler == True:
            df = self.data_csv[self.data_csv.fold != self.fold].reset_index(drop=False)
            class_sample_count = np.array(
            [len(np.where(df.MPNST == t)[0]) for t in np.unique(df.MPNST)])
            weight = 1. / class_sample_count
            samples_weight = np.array([weight[t] for t in df.MPNST])
            samples_weight = torch.from_numpy(samples_weight)
            samples_weight = samples_weight.double()
            self.sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=2, sampler=self.sampler)
    
        
    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=2, shuffle=False)        


class MPNSTDataset(Dataset):
    def __init__(self, data, mri_type, transform=None, use_roi=True):
        self.data = data
        self.mri_type = mri_type
        self.transform = transform
        self.use_roi = use_roi
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        row = self.data[index]
        case_id = row['case_id']
        label = int(row['label'])
        _3d_images = self.load_images_3d(case_id, self.use_roi)
        _3d_images = torch.tensor(_3d_images).float()
        sample = {"image": _3d_images.unsqueeze(0), "label": label, "case_id": case_id}
        # If use transformation
        if self.transform != None:
            sample = self.transform(sample) 
        
        return sample    

    def prepare_roi(self):
        # use for the first time
        roi_coodinates = {}
        print("Caulculating the ROI from segmentation for every images...")
        for row in tqdm(self.data, total=len(self.data)):
            case_id = row['case_id']
            seg = f"/trinity/home/xwan/data/MPNST/{case_id}/segmentations.nii.gz"
            coodinates = extract_3d_bbx(seg)
            roi_coodinates[case_id] = coodinates
                            
        joblib.dump(roi_coodinates, f"/trinity/home/xwan/MPNST_DL/input/{self.mri_type}/3d_roi_{self.mri_type}.pkl")
        return roi_coodinates
    
    def __get_roi(self):
        if (f"3d_roi_{self.mri_type}.pkl" in os.listdir(f"/trinity/home/xwan/MPNST_DL/input/{self.mri_type}/"))\
            and (self.use_roi) :
            print("Loading the ROI from segmentations for all the images...")
            roi_coodinates = joblib.load(f"/trinity/home/xwan/MPNST_DL/input/{self.mri_type}/3d_roi_{self.mri_type}.pkl")
            return roi_coodinates
        else:
            print("Prepare dataset first to generate .pkl file.")

    def load_images_3d(self, case_id, use_roi):
        f_img = f"/trinity/home/xwan/data/MPNST/{case_id}/{self.mri_type}.nii.gz"
        img = nib.load(f_img).get_fdata()
        if use_roi == True:
            img_roi = self.__get_roi()
            x1, x2, y1, y2, z1, z2 = img_roi[case_id]
            return img[x1:x2, y1:y2:, z1:z2]
        else:
            return img
        