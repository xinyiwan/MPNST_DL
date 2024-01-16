import os
from datetime import datetime
import yaml
import tempfile
import torch
import argparse
from dataset import MPNSTDataMoule
from model import MyModel
from net import init_net
from get_cfg import get_parameters
from utilities import log_confusion_matrix
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import NeptuneLogger
import monai


# Set up device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set arguments 
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--fold', type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    if args.fold:
        cfg['dataset']['fold'] = int(args.fold)
    
    parameters = get_parameters(cfg)

    # create learning rate logger
    lr_logger = LearningRateMonitor(logging_interval="epoch")

    # Init checkpoints save direction
    time_label = datetime.now().strftime("%m%d")
    exp_dir = "/trinity/home/xwan/MPNST_DL/output/checkpoints/{}_{}_config-{}".format(parameters["task"],time_label, cfg['config']['file_idx'])
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)
    cp_dir = os.path.join(exp_dir, f"fold_{parameters['fold']}")
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
    name = "{}_{}_cfg-{}_fold{}".format(parameters["task"], time_label, cfg['config']['file_idx'], parameters['fold']),
    api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI0YjhhZDFkNi1lMWNhLTQ1MzktYjcxYS05MDNlMjkyYTEyNDEifQ==",
    log_model_checkpoints=False,
    tags = ['training', parameters['net']]
    )


    # Initialize a datamodule
    data = MPNSTDataMoule(
        batch_size = cfg['dataset']['batch_size'],
        pixdim = tuple(cfg['transform']['pixdim']),
        spatial_size = tuple(cfg['transform']['spatial_size']),
        fold = args.fold,
        mri_type =  cfg['dataset']['task_name'],
    )
    data.prepare_data()
    data.setup()
    print("Training:  ", len(data.train_set))
    print("Validation: ", len(data.val_set))
    
    
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
    net = init_net(cfg)
    model = MyModel(
        net = net,
        learning_rate = parameters["learning_rate"],
        decay_factor = parameters["decay_factor"]
    ).to(device)

    # (neptune) log model summary and hyper-parameters
    neptune_logger.log_model_summary(model=model, max_depth=-1)
    neptune_logger.log_hyperparams(params=parameters)

    trainer.fit(model=model, datamodule=data)

    # (neptune) log confusion matrix
    log_confusion_matrix(model, data, neptune_logger, cp_dir)


if __name__ == '__main__':
    main()


