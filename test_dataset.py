from working.dataset import MPNSTDataMoule
import yaml
import argparse
from lightning.pytorch.loggers import NeptuneLogger

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
