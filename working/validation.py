import os
import monai
import numpy as np
import pandas as pd
import torch
import yaml
from dataset import MPNSTDataMoule
from train import parse_args
from get_cfg import get_parameters
from net import init_net
from model import MyModel
import argparse
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('date', type=str)
    return parser.parse_args()

def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    parameters = get_parameters(cfg)
    data = pd.read_csv(f"/trinity/home/xwan/MPNST_DL/input/{parameters['task']}/train_{parameters['task']}.csv")
    device = torch.device("cuda")

    # Init model
    net = init_net(cfg)
    model = MyModel(
        net = net,
        learning_rate = parameters["learning_rate"],
        decay_factor = parameters["decay_factor"]
    ).to(device)

    aucs = []
    accs = []
    sens = []
    spes = []

    for i in range(5):
        data = MPNSTDataMoule(
            batch_size = cfg['dataset']['batch_size'],
            pixdim = tuple(cfg['transform']['pixdim']),
            spatial_size = tuple(cfg['transform']['spatial_size']),
            fold = i,
            mri_type =  cfg['dataset']['task_name'],
        )
        data.prepare_data()
        data.setup()
        checkpoint = torch.load('/trinity/home/xwan/MPNST_DL/output/checkpoints/{}_{}_config-{}/fold_{}/last.ckpt'.format(parameters['task'], args.date, int(parameters['idx']), i))
        model.load_state_dict(checkpoint['state_dict'], strict=False)

        val_loader = data.val_dataloader()
        preds = []
        labels = []
        with torch.no_grad():
            for  step, batch in enumerate(val_loader):
                model.eval()
                images = batch["image"].to(device)
                outputs = model(images)
                preds = np.append(preds, outputs.argmax(axis=1).detach().cpu().numpy())
                labels = np.append(labels, batch["label"].cpu().detach().numpy())
        print(labels)
        print(preds)

        acc = accuracy_score(labels, preds)
        auc = roc_auc_score(labels, preds)
        tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
        specificity = tn / (tn+fp)
        sensitivity = tp / (tp+fn)
        accs.append(acc)
        aucs.append(auc)
        sens.append(sensitivity)
        spes.append(specificity)

    df = pd.DataFrame()
    df['acc'] = accs
    df['auc'] = aucs
    df['sensitivity'] = sens
    df['specificity'] = spes
    df = df.round(4)
    df.to_csv('/trinity/home/xwan/MPNST_DL/output/checkpoints/{}_{}_config-{}/cv5.csv'.format(parameters['task'], args.date, int(parameters['idx'])))
    print(df)
        
if __name__ == '__main__':
    main()





