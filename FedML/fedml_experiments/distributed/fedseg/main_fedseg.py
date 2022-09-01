import argparse
import logging
import os
import random
import socket
import sys
import datetime

import numpy as np
import psutil
import setproctitle
import torch
import wandb
import yaml

# add the FedML root directory to the python path

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))

from fedml_api.distributed.utils.gpu_mapping import mapping_processes_to_gpu_device_from_yaml_file

from fedml_api.model.cv.deeplabV3_plus import DeepLabV3_plus
from fedml_api.model.cv.unet import Unet
from fedml_api.distributed.fedseg.FedSegAPI import FedML_init, FedML_FedSeg_distributed
from fedml_api.distributed.fedseg.utils import count_parameters


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--cfg', type=str, default='default.yml')

    parser.add_argument('--model', type=str, default=None, metavar='N',
                        help='neural network used in training')

    parser.add_argument('--backbone', type=str, default=None,
                        help='employ with backbone (default: xception)')

    parser.add_argument('--backbone_pretrained', type=str2bool, default=None,
                        help='pretrained backbone (default: False)')

    parser.add_argument('--backbone_freezed', type=str2bool, default=None,
                        help='Freeze backbone to extract features only once (default: False)')

    parser.add_argument('--extract_feat', type=str2bool, default=None,
                        help='Extract Feature Maps of (default: False) NOTE: --backbone_freezed has to be True for this argument to be considered')

    parser.add_argument('--outstride', type=int, default=8,
                        help='network output stride (default: 16)')

    # # TODO: Remove this argument
    # parser.add_argument('--categories', type=str, default='person,dog,cat',
    #                     help='segmentation categories (default: person, dog, cat)')

    parser.add_argument('--dataset', type=str, default=None, metavar='N',
                        choices=['brats', 'heart', 'path', 'brats_miss'],
                        help='dataset used for training')

    parser.add_argument('--data_dir', type=str, default=None,
                        help='data directory')
 
    parser.add_argument('--checkname', type=str, default=None, help='set the checkpoint name')

    parser.add_argument('--partition_method', type=str, default=None, metavar='N',
                        help='how to partition the dataset on local workers')

    parser.add_argument('--partition_alpha', type=float, default=None, metavar='PA',
                        help='partition alpha (default: 0.5)')

    parser.add_argument('--client_num_in_total', type=int, default=None, metavar='NN',
                        help='number of workers in a distributed cluster')

    parser.add_argument('--client_num_per_round', type=int, default=None, metavar='NN',
                        help='number of workers')

    parser.add_argument('--save_client_model', type=str2bool, default=None,
                        help='whether to save locally trained model by clients (default: True')

    parser.add_argument('--batch_size', type=int, default=None, metavar='N',
                        help='input batch size for training (default: 32)')

    parser.add_argument('--sync_bn', type=str2bool, default=None,
                        help='whether to use sync bn (default: auto)')

    parser.add_argument('--freeze_bn', type=str2bool, default=None,
                        help='whether to freeze bn parameters (default: False)')

    parser.add_argument('--client_optimizer', type=str, default=None,
                        help='adam')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (default: 0.001)')

    parser.add_argument('--lr_scheduler', type=str, default=None,
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')

    parser.add_argument('--lr_step', type=int, default=None, help='#epochs to update lr for step scheduler')

    parser.add_argument('--beta1', type=float, default=None, help='momentum term of adam')

    parser.add_argument('--momentum', type=float, default=None,
                        metavar='M', help='momentum (default: 0.9)')

    parser.add_argument('--weight_decay', type=float, default=None,
                        metavar='M', help='w-decay (default: 5e-4)')

    parser.add_argument('--nesterov', type=str2bool, default=None,
                        help='whether use nesterov (default: False)')

    parser.add_argument('--loss_type', type=str, default=None,
                        choices=['ce', 'focal', 'dice', 'cedice', 'focaldice'],
                        help='loss func type (default: ce)')

    parser.add_argument('--epochs', type=int, default=None, metavar='EP',
                        help='how many epochs will be trained locally')

    parser.add_argument('--comm_round', type=int, default=None,
                        help='how many round of communications we shoud use')

    # parser.add_argument('--is_mobile', type=int, default=0,
    #                     help='whether the program is running on the FedML-Mobile server side')

    parser.add_argument('--evaluation_frequency', type=int, default=None,
                        help='Frequency of model evaluation on training dataset (Default: every 5th round)')

    parser.add_argument('--gpu_mapping_file', type=str, default=None,
                        help='the gpu utilization file for servers and clients. If there is no \
                        gpu_util_file, gpu will not be used.')

    parser.add_argument('--gpu_mapping_key', type=str, default=None,
                        help='the key in gpu utilization file')

    # parser.add_argument('--ci', type=int, default=0,
    #                     help='CI')

    args = parser.parse_args()

    return args


def cfg_parse(args, cfg):
    args_dict = args.__dict__
    args_key_list = [key for key in args_dict]
    cfg_key_list = [key for key in cfg]

    # init None args with cfg values
    undefined_arg_key = filter(lambda x, args_dict=args_dict: args_dict[x] is None, args_key_list)
    undefined_arg_key = filter(lambda x: x in cfg_key_list, undefined_arg_key)
    for key_name in undefined_arg_key:
        args_dict[key_name] = cfg[key_name]

    # add args which are not included in args parser
    uncontained_arg_key = filter(lambda x: not (x in args_key_list), cfg_key_list)
    for key_name in uncontained_arg_key:
        args_dict[key_name] = cfg[key_name]

    return args


def load_data(process_id, args):

    if args.dataset[:5].lower() == "brats":
        from fedml_api.data_preprocessing.brats.data_loader import load_partition_data_distributed_brats as data_loader
    elif args.dataset.lower() == 'path':
        from fedml_api.data_preprocessing.exp2_path.data_loader import load_partition_data_distributed_path as data_loader
    elif args.dataset.lower() == 'heart':
        from fedml_api.data_preprocessing.exp1_heart.data_loader import load_partition_data_distributed_heart as data_loader


    train_data_num, train_data_global, test_data_global, local_data_num, train_data_local, test_data_local, class_num = data_loader(
        process_id, args.dataset, args.data_dir, args.partition_method, args.partition_alpha,
        args.client_num_in_total, args.batch_size)
    dataset = [train_data_num, train_data_global, test_data_global, local_data_num, train_data_local, test_data_local,
               class_num]

    return dataset


def create_model(args, output_dim, img_size=torch.Size([513, 513])):
    if args.model.lower() == "deeplabv3_plus":
        model = DeepLabV3_plus(backbone=args.backbone,
                          image_size=img_size,
                          n_classes=output_dim,
                          output_stride=args.outstride,
                          pretrained=args.backbone_pretrained,
                          freeze_bn=args.freeze_bn,
                          sync_bn=args.sync_bn)

        logging.info('Args.Backbone: {}'.format(args.backbone_freezed))

        if args.backbone_freezed:
            logging.info('Freezing Backbone')
            for param in model.feature_extractor.parameters():
                param.requires_grad = False
        else:
            logging.info('Finetuning Backbone')

        num_params = count_parameters(model)
        logging.info("DeepLabV3_plus Model Size = {0} M".format(str(num_params)))
    elif args.model.lower() == 'unet':
        model = Unet(in_channel=args.input_nc, out_channel=args.output_nc)

        num_params = count_parameters(model)
        logging.info("Unet Model Size = {0} M".format(str(num_params)))
    else:
        raise ('Not Implemented Error')

    return model


if __name__ == "__main__":
    now = datetime.datetime.now()
    time_start = now.strftime("%Y-%m-%d %H:%M:%S")
    
    logging.info("Executing Image Segmentation at time: {0}".format(time_start))
    
    # initialize distributed computing (MPI)
    comm, process_id, worker_number = FedML_init()

    # parse python script input parameters
    parser = argparse.ArgumentParser()
    args = add_args(parser)

    with open(args.cfg, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    args = cfg_parse(args, cfg)

    # customize the process name
    str_process_name = "FedSeg (distributed):" + str(process_id)
    setproctitle.setproctitle(str_process_name)

    # customize the log format
    logging.basicConfig(filename='info.log',
                        level=logging.INFO,
                        format=str(process_id) + ' - %(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S')

    hostname = socket.gethostname()
    logging.info("#############process ID = " + str(process_id) +
                 ", host name = " + hostname + "########" +
                 ", process ID = " + str(os.getpid()) +
                 ", process Name = " + str(psutil.Process(os.getpid())))
    logging.info(args)
    # initialize the wandb machine learning experimental tracking platform (https://www.wandb.com/).
    if process_id == 0:
        wandb.init(
            # project="federated_nas",
            project="fedml",
            name="FedSeg(d)" + args.dataset + '-' + str(args.partition_method) + "-r" + str(args.comm_round) + "-e" + str(
                args.epochs) + "-lr" + str(
                args.lr),
            config=args
        )

    # Set the random seed. The np.random seed determines the dataset partition.
    # The torch_manual_seed determines the initial weight.
    # We fix these two, so that we can reproduce the result.
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = True  # fixed input size [256, 256]

    logging.info("process_id = %d, size = %d" % (process_id, worker_number))
    device = mapping_processes_to_gpu_device_from_yaml_file(process_id, worker_number, args.gpu_mapping_file, args.gpu_mapping_key)

    # load data
    dataset = load_data(process_id, args)
    [train_data_num, train_data_global, test_data_global, data_local_num_dict,
     train_data_local_dict, test_data_local_dict, class_num] = dataset

    # create model.
    # Note if the model is DNN (e.g., ResNet), the training will be very slow.
    # In this case, please use our FedML distributed version (./fedml_experiments/distributed/fedseg)
    model = create_model(args, output_dim=class_num)

    logging.info("Calling FedML_FedSeg_distributed")

    # start "federated segmentation (FedSeg)"
    FedML_FedSeg_distributed(process_id, worker_number, device, comm, model, train_data_global, test_data_global, data_local_num_dict,
                             train_data_local_dict, test_data_local_dict, args)
