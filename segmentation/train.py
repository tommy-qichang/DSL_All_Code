import argparse
import collections
import importlib
import stringcase
import numpy as np
import torch
import json
from parse_config import ConfigParser

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):
    print(f'==============Configuration==============')
    print(f'{json.dumps(config._config, indent=4)}')
    print(f'==============End Configuration==============')
    logger = config.get_logger('train')
    # setup data_loader instances
    my_transform = config.init_transform()
    data_loader = config.init_obj('data_loader', transforms=my_transform, training=True)
    valid_data_loader = data_loader.split_validation()

    # build model architecture, then print to console
    model = config.init_obj('model')
    logger.info(model)

    # get function handles of loss and metrics
    criterion = config.init_ftn('loss')

    # criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(importlib.import_module(f"metric.{met}"), met) for met in config['metric']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)

    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    # Sould config trainer name.
    trainer_name = config["trainer"]["type"]
    module = importlib.import_module(f"trainer.{stringcase.snakecase(trainer_name)}")
    Trainer = getattr(module, trainer_name)

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)

    logger.info(f"Use Trainer: {trainer_name}")
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
