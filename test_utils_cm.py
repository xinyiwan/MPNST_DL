from working.utils import log_confusion_matrix
from working.dataset import MPNSTDataMoule
from working.model import MyModel
import monai
import torch
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = monai.networks.nets.DenseNet(spatial_dims = 3, in_channels = 1, 
                                             out_channels = 2,
                                             init_features = 16,
                                             growth_rate = 16,
                                             block_config = [4, 8, 8, 4],
                                             dropout_prob = 0.2)

model = MyModel(
        net = net,
        learning_rate = 0.0001,
        decay_factor = 0.9,
    ).to(device)
checkpoint = torch.load('/trinity/home/xwan/MPNST_DL/output/checkpoints/T1_0111_config-5/fold_0/epoch=49.ckpt')
model.load_state_dict(checkpoint['state_dict'], strict=False)
model.eval()


neptune_logger = NeptuneLogger(
    project="xinyiwan/MPNST",
    name='test',
    api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI0YjhhZDFkNi1lMWNhLTQ1MzktYjcxYS05MDNlMjkyYTEyNDEifQ==",
    log_model_checkpoints=False,
    tags = ['training', 'test']
)

cp_dir = '/trinity/home/xwan/MPNST_DL/output/checkpoints/T1_0111_config-5/fold_0'
log_confusion_matrix(model, data, neptune_logger, cp_dir)
