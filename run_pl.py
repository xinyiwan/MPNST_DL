from datetime import datetime
import os, glob
import tempfile
import argparse
import yaml
import pandas as pd
import numpy as np
import json
import neptune

import torch
from torch.nn import Linear, CrossEntropyLoss, functional as F
import torchmetrics
from torch.utils.data import random_split, DataLoader
from torch.optim.lr_scheduler import LambdaLR

import pytorch_lightning as pl
import matplotlib.pyplot as plt
import seaborn as sns

import monai
from monai.data import CacheDataset, list_data_collate, decollate_batch, DataLoader
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    ScaleIntensityd,
    Spacingd,
    ResizeWithPadOrCropd,
    AsDiscrete,
    Activations,
    EnsureType,
)
from monai.metrics import ROCAUCMetric
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import NeptuneLogger

from torchmetrics.classification import BinaryAccuracy
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
# from scikitplot.metrics import plot_confusion_matrix


# Set up root 
directory = "/trinity/home/xwan/data"
root_dir = tempfile.mkdtemp() if directory is None else directory
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MPNSTDataMoule(pl.LightningDataModule):
    def __init__(self, task, batch_size, train_val_ratio, pixdim, spatial_size, mri_type = 'T1'):
        super().__init__()
        self.batch_size = batch_size
        self.train_val_ratio = train_val_ratio
        self.pixdim = pixdim
        self.spatial_size = spatial_size
        self.base_dir = root_dir
        self.dataset_dir = os.path.join(root_dir, task)
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



class Model(pl.LightningModule):
    def __init__(self, net, learning_rate, decay_factor):
        super().__init__()
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        # optimizer paramters
        self.lr = learning_rate
        self.decay_factor = decay_factor

        self.net = net

    def forward(self, x):
        return self.net(x)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = LambdaLR(optimizer, lambda epoch: self.decay_factor**epoch)
        return [optimizer], [scheduler]
    
    def training_step(self, batch, batch_idx):
        x, y = batch['image'], batch['label']
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train/batch/loss", loss, prog_bar=False)

        y_true = y.cpu().detach().numpy()
        y_pred = y_hat.argmax(axis=1).cpu().detach().numpy()
        acc = accuracy_score(y_true, y_pred)
        self.log("train/batch/acc", acc)
        self.training_step_outputs.append({"loss": loss, "y_true": y_true, "y_pred": y_pred})
        return {"loss": loss, "y_true": y_true, "y_pred": y_pred}

    def on_train_epoch_end(self):
        loss = np.array([])
        y_true = np.array([])
        y_pred = np.array([])
        for results_dict in self.training_step_outputs:
            loss = np.append(loss, results_dict["loss"].cpu().detach().numpy())
            y_true = np.append(y_true, results_dict["y_true"])
            y_pred = np.append(y_pred, results_dict["y_pred"])
        acc = accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred)
        self.log("train/epoch/loss", loss.mean())
        self.log("train/epoch/acc", acc)
        self.log("train/epoch/auc", auc)
        self.training_step_outputs.clear()  # free memory

    def validation_step(self, batch, batch_idx):
        x, y = batch['image'], batch['label']
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        y_true = y.cpu().detach().numpy()
        y_pred = y_hat.argmax(axis=1).cpu().detach().numpy()

        self.validation_step_outputs.append({"loss": loss, "y_true": y_true, "y_pred": y_pred})

        return {"loss": loss, "y_true": y_true, "y_pred": y_pred}
    
    def on_validation_epoch_end(self):
        loss = np.array([])
        y_true = np.array([])
        y_pred = np.array([])
        for results_dict in self.validation_step_outputs:
            loss = np.append(loss, results_dict["loss"].cpu().detach().numpy())
            y_true = np.append(y_true, results_dict["y_true"])
            y_pred = np.append(y_pred, results_dict["y_pred"])
        acc = accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred)
        self.log("val/loss", loss.mean())
        self.log("val/acc", acc)
        self.log("val/auc", auc)
        self.validation_step_outputs.clear()  # free memory

def log_confusion_matrix(lit_model, data_module, neptune_logger):
    lit_model.freeze()
    val_data = data_module.val_dataloader()
    y_true = np.array([])
    y_pred = np.array([])
    for batch in val_data:
        x, y = batch['image'], batch['label']
        y = y.cpu().detach().numpy()
        y_hat = lit_model.forward(x).argmax(axis=1).cpu().detach().numpy()
        y_true = np.append(y_true, y)
        y_pred = np.append(y_pred, y_hat)

    fig, ax = plt.subplots(figsize=(16, 12))
    # plot_confusion_matrix(y_true, y_pred, ax=ax)
    neptune_logger.experiment["confusion_matrix"].upload(fig)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--train-val-ratio', type=int)
    parser.add_argument('--trans-pixdim', nargs='+', type=float)
    parser.add_argument('--trans-spatial-size', nargs='+', type=float)

    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    if args.batch_size:
        cfg['dataset']['batch_size'] = args.batch_size
    if args.train_val_ratio:
        cfg['dataset']['train_val_ratio'] = args.train_val_ratio
    if args.trans_pixdim:
        cfg['transform']['pixdim'] = tuple(args.trans_pixdim)
    if args.trans_spatial_size:
        cfg['transform']['spatial_size'] = tuple(args.spatial_size)


    # Define hyper-parameters
    parameters = {
        "dense_init_features": cfg['model']['densenet']['init_features'],
        "dense_growth_rate": cfg['model']['densenet']['growth_rate'],
        "dropout": cfg['model']['densenet']['dropout_prob'],
        "learning_rate": cfg['optimizer']['learning_rate'],
        "decay_factor":cfg['optimizer']['decay_factor'],
        "batch_size": cfg['train']['batch_size'],
        "n_epochs": cfg['train']['epochs'],
        "cfg": cfg['dataset']['task_name']
        } 



    # create learning rate logger
    lr_logger = LearningRateMonitor(logging_interval="epoch")

    # Init checkpoints save direction
    time_label = datetime.now().strftime("%m%d-%H%M")
    cp_dir = "/trinity/home/xwan/MPNST_DL/output/checkpoints/{}_{}".format(parameters["cfg"],time_label)
    if not os.path.exists(cp_dir):
        os.mkdir(cp_dir)

    # create model checkpointing object
    model_checkpoint = ModelCheckpoint(
        dirpath=cp_dir,
        filename="{epoch:02d}",
        save_weights_only=True,
        save_top_k=3,
        save_last=True,
        monitor="val/loss",
        every_n_epochs=1,
    )

    # (neptune) create NeptuneLogger
    neptune_logger = NeptuneLogger(
    project="xinyiwan/MPNST",
    api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI0YjhhZDFkNi1lMWNhLTQ1MzktYjcxYS05MDNlMjkyYTEyNDEifQ==",
    log_model_checkpoints=False,
)



    # Initialize a datamodule
    data = MPNSTDataMoule(
        task = cfg['dataset']['task'],
        batch_size = cfg['dataset']['batch_size'],
        train_val_ratio = cfg['dataset']['train_val_ratio'],
        pixdim = tuple(cfg['transform']['pixdim']),
        spatial_size = tuple(cfg['transform']['spatial_size']),
    )
    data.prepare_data()
    data.setup()
    print("Training:  ", len(data.train_set))
    print("Validation: ", len(data.val_set))
    
    if cfg['model']['net'] == 'resnet':
        net = monai.networks.nets.resnet.resnet18(
                                            spatial_dims = 3, 
                                            n_input_channels = cfg['model']['resnet']['in_channels'] 
                                            )
    else:
        net = monai.networks.nets.DenseNet(spatial_dims = 3, in_channels = cfg['model']['densenet']['in_channels'], 
                                             out_channels = cfg['model']['densenet']['num_classes'],
                                             init_features = cfg['model']['densenet']['init_features'],
                                             growth_rate = cfg['model']['densenet']['growth_rate'],
                                             block_config = tuple(cfg['model']['densenet']['block_config']),
                                             dropout_prob = cfg['model']['densenet']['dropout_prob'])

    # Initialize a trainer
    trainer = pl.Trainer(
        logger = neptune_logger,
        devices = [0],
        callbacks = [lr_logger, model_checkpoint],
        log_every_n_steps = 8,
        max_epochs = parameters["n_epochs"],
        enable_progress_bar = False,
    )

    # Initialize model
    model = Model(
        net = net,
        learning_rate = parameters["learning_rate"],
        decay_factor = parameters["decay_factor"]
    ).to(device)

    # (neptune) log model summary
    neptune_logger.log_model_summary(model=model, max_depth=-1)

    # (neptune) log hyper-parameters
    neptune_logger.log_hyperparams(params=parameters)

    trainer.fit(model=model, datamodule=data)
    
    # (neptune) log confusion matrix
    log_confusion_matrix(model, data, neptune_logger)


if __name__ == '__main__':
    main()



