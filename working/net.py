import monai
from networks.classifiers import cnn3d
from networks.resnet3d_spp import ResNet_spp, generate_model
def init_net(cfg):
    if cfg['model']['net'] == 'resnet':
        net = monai.networks.nets.resnet.resnet18(
                                            spatial_dims = 3, 
                                            n_input_channels = cfg['model']['resnet']['in_channels'] 
                                            )
    if cfg['model']['net'] == 'densenet':
        net = monai.networks.nets.DenseNet(spatial_dims = 3, in_channels = cfg['model']['densenet']['in_channels'], 
                                             out_channels = cfg['model']['densenet']['num_classes'],
                                             init_features = cfg['model']['densenet']['init_features'],
                                             growth_rate = cfg['model']['densenet']['growth_rate'],
                                             block_config = tuple(cfg['model']['densenet']['block_config']),
                                             dropout_prob = cfg['model']['densenet']['dropout_prob'])
    if cfg['model']['net'] == 'resnet_spp':
        net = generate_model(10, 
                             init= cfg['model']['resnet_spp']['init_features'],
                             in_channel = cfg['model']['resnet_spp']['in_channels'])
    if cfg['model']['net'] == 'classifier':
        net = cnn3d()
    return net





