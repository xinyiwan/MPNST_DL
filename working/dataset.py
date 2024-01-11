import os, glob
import pandas as pd
import monai
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    ScaleIntensityd,
    Spacingd,
    ResizeWithPadOrCropd,
)


class MPNSTDataMoule(pl.LightningDataModule):
    def __init__(self, task, batch_size, train_val_ratio, pixdim, spatial_size, mri_type = 'T1'):
        super().__init__()
        self.batch_size = batch_size
        self.train_val_ratio = train_val_ratio
        self.pixdim = pixdim
        self.spatial_size = spatial_size
        self.mri_type = mri_type
        self.transform = None
        self.data_dicts = None
        self.preprocess = None
        self.train_set = None
        self.val_set = None
        self.test_set = None

    def prepare_data(self):
        # Load labels
        gt = pd.read_csv(f'/trinity/home/xwan/MPNST_DL/input/{self.mri_type}/train_{self.mri_type}.csv')
        train = pd.read_csv(f"input/{self.mri_type}/train_{self.mri_type}.csv")

        image_paths = [f'/trinity/home/xwan/data/MPNST/{pid}/{self.mri_type}.nii.gz' for pid in train['Patient']]
        labels = train['MPNST'].tolist()
        self.data_dicts = [{"image": img, "label": label} for img, label in zip(image_paths, labels)]

    def build_transform(self, keys="image"):
            trans = Compose([
                LoadImaged(keys = keys),
                EnsureChannelFirstd(keys = keys),
                Orientationd(keys = keys, axcodes="PLI"),
                Spacingd(
                    keys = keys,
                    pixdim = self.pixdim,
                    mode = ("bilinear")
                    ),
                ResizeWithPadOrCropd(
                    keys = keys,
                    spatial_size=self.spatial_size
                ),
                ScaleIntensityd(keys = keys)
            ])  
            return trans
    
    def setup(self, stage = None):
        num_subjects = len(self.data_dicts)
        num_train_subjects = int(round(num_subjects * self.train_val_ratio))
        num_val_subjects = num_subjects - num_train_subjects
        splits = num_train_subjects, num_val_subjects
        train_subjects, val_subjects = random_split(self.data_dicts, splits)

        self.transform = self.build_transform()
        self.train_set = monai.data.Dataset(data=train_subjects, transform=self.transform)
        self.val_set = monai.data.Dataset(data=val_subjects, transform=self.transform)
    
    def train_dataloader(self):
         return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=2)
        
    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=2)        
