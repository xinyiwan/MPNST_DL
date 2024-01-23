
def get_parameters(cfg):

    # Define hyper-parameters
    parameters = {
            'net': cfg['model']['net'],
            'learning_rate': cfg['optimizer']['learning_rate'],
            'decay_factor':cfg['optimizer']['decay_factor'],
            'batch_size': cfg['dataset']['batch_size'],
            'n_epochs': cfg['train']['epochs'],
            'task': cfg['dataset']['task_name'],
            'fold': cfg['dataset']['fold'],
            'idx': cfg['config']['file_idx'],
            } 
    if cfg['model']['net'] == 'densenet':
        parameters['dense_init_features'] = cfg['model']['densenet']['init_features']
        parameters['dense_growth_rate'] = cfg['model']['densenet']['growth_rate']
        parameters['block_config'] = cfg['model']['densenet']['block_config']
        parameters['dropout'] = cfg['model']['densenet']['dropout_prob']

    if cfg['model']['net'] == 'resnet':
        parameters['layers'] = cfg['model']['resnet']['layers']

    return parameters
