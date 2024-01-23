import os, glob
import pandas as pd
import monai
import pytorch_lightning as pl
import torch
import joblib
from tqdm import tqdm
from monai.data import Dataset
from utilities import extract_3d_bbx
from torch.utils.data import DataLoader
from monai.transforms import (
    Compose,
    Resized,
    LoadImaged,
    Orientationd,
    ScaleIntensityd,
    Spacingd,
    ResizeWithPadOrCropd,
    MapTransform,
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
                Spacingd(
                    keys = keys,
                    pixdim = self.pixdim,
                    mode = ("bilinear")
                    ),
                Resized(
                    keys = keys,
                    spatial_size=self.spatial_size
                ),
                ScaleIntensityd(keys = keys)
            ])  
            return trans
    
    def setup(self, stage = None):

        self.transform = self.build_transform()
        train_df = self.train_set
        val_df = self.val_set
        mri_type = self.mri_type
        tf = self.transform 
        self.train_set = MPNSTDataset(data=train_df, mri_type=mri_type, transform=tf)
        self.val_set = MPNSTDataset(data=val_df, mri_type=mri_type, transform=tf)        

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=0, shuffle=True)
        
    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=0, shuffle=False)        


class MPNSTDataset(Dataset):
    def __init__(self, data, mri_type, transform=None, is_train=True, use_roi=True):
        self.data = data
        self.mri_type = mri_type
        self.transform = transform
        self.use_roi = use_roi
        self.is_train = is_train
        self.use_roi = use_roi
        self.img_roi = self.__get_roi()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
    
        row = self.data[index]
        case_id = row['case_id']
        label = int(row['label'])
        _3d_images = self.load_roi_images_3d(case_id)
        _3d_images = torch.tensor(_3d_images).float()
        sample = {"image": _3d_images.unsqueeze(0), "label": label, "case_id": case_id}
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

    def load_roi_images_3d(self, case_id):

        f_img = f"/trinity/home/xwan/data/MPNST/{case_id}/{self.mri_type}.nii.gz"
        img = nib.load(f_img).get_fdata()
        x1, x2, y1, y2, z1, z2 = self.img_roi[case_id]
        return img[x1:x2, y1:y2:, z1:z2]
        