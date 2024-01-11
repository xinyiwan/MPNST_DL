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
    def __init__(self, batch_size, pixdim, spatial_size, fold, mri_type):
        super().__init__()
        self.batch_size = batch_size
        self.pixdim = pixdim
        self.spatial_size = spatial_size
        self.fold = fold
        self.mri_type = mri_type
        self.transform = None
        self.train_set = None
        self.val_set = None

    def prepare_data(self):
        # Load labels
        data_csv = pd.read_csv(f"/trinity/home/xwan/MPNST_DL/input/{self.mri_type}/train_{self.mri_type}.csv")

        train_df = data_csv[data_csv.fold != self.fold].reset_index(drop=False)
        train_img_paths = [f'/trinity/home/xwan/data/MPNST/{pid}/{self.mri_type}.nii.gz' for pid in train_df['Patient']]
        train_labels = train_df['MPNST'].tolist()

        val_df = data_csv[data_csv.fold == self.fold].reset_index(drop=False)
        val_img_paths = [f'/trinity/home/xwan/data/MPNST/{pid}/{self.mri_type}.nii.gz' for pid in val_df['Patient']]
        val_labels = val_df['MPNST'].tolist()
    
        self.train_set = [{"image": img, "label": label} for img, label in zip(train_img_paths, train_labels)]
        self.val_set = [{"image": img, "label": label} for img, label in zip(val_img_paths, val_labels)]
        

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

        self.transform = self.build_transform()
        self.train_set = monai.data.Dataset(data=self.train_set, transform=self.transform)
        self.val_set = monai.data.Dataset(data=self.val_set, transform=self.transform)
    
    def train_dataloader(self):
         return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=2)
        
    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=2)        
