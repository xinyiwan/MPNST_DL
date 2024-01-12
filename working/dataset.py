import os, glob
import pandas as pd
import monai
import pytorch_lightning as pl
import torch
import joblib
from tqdm import tqdm
from monai.data import Dataset
from working.utilities import extract_3d_bbx
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
import nibabel as nib



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
        train_seg_paths = [f'/trinity/home/xwan/data/MPNST/{pid}/segmentations.nii.gz' for pid in train_df['Patient']]
        train_labels = train_df['MPNST'].tolist()
        train_ids = train_df['Patient'].tolist()

        val_df = data_csv[data_csv.fold == self.fold].reset_index(drop=False)
        val_img_paths = [f'/trinity/home/xwan/data/MPNST/{pid}/{self.mri_type}.nii.gz' for pid in val_df['Patient']]
        val_seg_paths = [f'/trinity/home/xwan/data/MPNST/{pid}/segmentations.nii.gz' for pid in val_df['Patient']]
        val_labels = val_df['MPNST'].tolist()
        val_ids = train_df['Patient'].tolist()
    
        self.train_set = [{"image": img, "seg": seg, "label": label, "case_id": pid} for img, seg, label, pid in zip(train_img_paths, train_seg_paths, train_labels, train_ids)]
        self.val_set = [{"image": img, "seg": seg, "label": label, "case_id": pid} for img, seg, label, pid in zip(val_img_paths, val_seg_paths, val_labels, val_ids)]
        

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
        # self.train_set = monai.data.Dataset(data=self.train_set, transform=self.transform)
        # self.val_set = monai.data.Dataset(data=self.val_set, transform=self.transform)
        self.train_set = MPNSTDataset(self.train_set, mri_type=self.mri_type, transform=self.transform)
        self.val_set = MPNSTDataset(self.val_set, mri_type=self.mri_type, transform=self.transform)        

    def train_dataloader(self):
         return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=2, shuffle=True)
        
    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=2, shuffle=False)        


class MPNSTDataset(Dataset):
    def __init__(self, data, mri_type, transform=None, is_train=True, use_roi=True):
        super().__init__()
        self.data = data
        self.mri_type = mri_type
        self.transform = transform
        self.use_roi = use_roi
        self.img_roi = self.__prepare_roi()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        row = self.data.loc[index]
        case_id = int(row.Patient)
        label = int(row[self.MPNST])
        _3d_images = self.load_roi_images_3d(case_id)
        _3d_images = torch.tensor(_3d_images).float()

        if self.is_train:
            return {"image": _3d_images, "label": label, "case_id": case_id}
        else:
            return {"image": _3d_images, "case_id": case_id}

    def __prepare_roi(self):
        roi_coodinates = {}
        if (f"3d_roi_{self.mri_type}.pkl" in os.listdir("/trinity/home/xwan/MPNST_DL/input/"))\
            and (self.use_roi) :
            print("Loading the ROI from segmentations for all the images...")
            roi_coodinates = joblib.load(f"/trinity/home/xwan/MPNST_DL/input/3d_roi_{self.mri_type}.pkl")
            return roi_coodinates
        else:
            print("Caulculating the ROI from segmentation for every images...")
            for row in tqdm(self.data.iterrows(), total=len(self.data)):
                case_id = row[1].Patient
                seg = f"/trinity/home/xwan/data/MPNST/{case_id}/segmentations.nii.gz"
                coodinates = utils.extract_3d_bbx(seg)
                roi_coodinates[case_id] = coodinates

            joblib.dump(roi_coodinates, f"/trinity/home/xwan/MPNST_DL/input/3d_roi_{self.mri_type}.pkl")
            return roi_coodinates

    def load_roi_images_3d(self, case_id):

        f_img = f"/trinity/home/xwan/data/MPNST/{case_id}/{self.type}.nii.gz"
        img = nib.load(f_img).get_fdata()
        x1, x2, y1, y2, z1, z2 = self.img_roi[case_id]
        return img[x1:x2, y1:y2:, z1:z2]
        
     
