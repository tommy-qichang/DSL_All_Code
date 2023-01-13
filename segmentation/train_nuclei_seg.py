import argparse
import collections
import json

import torch
import numpy as np
import data_loader.data_loaders as module_data
import loss.loss as module_loss
import metric.metric_nuclei_seg as module_metric
# import model.unet as module_arch
# import optimizer.optimizer as module_optimizer
# import optimizer.lr_scheduler as module_scheduler
from parse_config import ConfigParser
from trainer import NucleiSegTrainer


# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):

    print(f'Configs:{json.dumps(config._config,indent=4)}')

    logger = config.get_logger('train')

    # setup data_loader instances
    train_data_loader = config.init_obj('data_loader')
    valid_data_loader = train_data_loader.split_validation()
    # test_data_loader = config.init_obj('test_data_loader', module_data)

    # build model architecture, then print to console
    model = config.init_obj('model')
    # logger.info(model)

    #######
    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]
    val_metrics = [getattr(module_metric, met) for met in config['metrics']]
    # test_metrics = [getattr(module_metric, met) for met in config['test_metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)

    # lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = NucleiSegTrainer(model, criterion, metrics, val_metrics, optimizer,
                               config=config,
                               data_loader=train_data_loader,
                               valid_data_loader=valid_data_loader,
                               lr_scheduler=None)

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
