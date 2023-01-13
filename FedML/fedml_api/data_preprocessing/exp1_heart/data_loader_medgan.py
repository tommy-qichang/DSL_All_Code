import logging

import torch.utils.data as data
import numpy as np
import h5py
import os

from .data_utility import init_transform, count_samples, build_pairs


logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def get_transforms():
    transforms_train = ["Resize", "RandomCrop", "RandomFlip", "ToTensorScale", "NormalizeBoth"]
    transforms_args_train = {
        "Resize": [286],
        "RandomCrop": [256],
        "RandomFlip": [True, True],
        "ToTensorScale": ['float', 1.0, 7.0],
        "NormalizeBoth": [0.5, 0.5]
    }

    transforms_test = ["Resize", "ToTensorScale", "NormalizeBoth"]
    transforms_args_test = {
        "Resize": [256],
        "ToTensorScale": ['float', 1.0, 7.0],
        "NormalizeBoth": [0.5, 0.5]
    }

    return init_transform(transforms_train, transforms_args_train), init_transform(transforms_test, transforms_args_test)


def get_dataloader(h5_train, h5_test, train_bs, test_bs, channel_in=3):
    transform_train, transform_test = get_transforms()

    if h5_train is not None:
        train_ds = GeneralDataset(h5_train,
                                  channel_in=channel_in,
                                  path="train",
                                  transforms=transform_train)
        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=True)
    else:
        train_dl = None

    if h5_test is not None:
        test_ds = GeneralDataset(h5_test,
                                 channel_in=channel_in,
                                 path="train",
                                 sample_rate=0.001,
                                 transforms=transform_test)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=False)
    else:
        test_dl = None

    return train_dl, test_dl


# Get a partition map for each client
def partition_data(datadir, partition, n_nets, alpha):
    logging.info("*********partition data***************")
    data_dict = dict()
    data_dict['test_all'] = os.path.join(datadir, 'whs_ct_train_2d_iso.h5')
    data_dict['train_all'] = None

    if partition == "homo":
        ## todo
        raise('Not Implemented Error')

    # non-iid data distribution
    elif partition == "hetero":
        data_dict[0] = os.path.join(datadir, 'asoca_train_2d_iso.h5')
        data_dict[1] = os.path.join(datadir, 'miccai2008_train_2d_iso.h5')
        data_dict[2] = os.path.join(datadir, 'whs_ct_train_2d_iso.h5')

    return data_dict


def load_partition_data_distributed_heart(process_id, dataset, data_dir, partition_method, partition_alpha,
                                         client_number, batch_size, channel_in):
    data_dict = partition_data(data_dir, partition_method, client_number, partition_alpha)
    #logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
    train_data_num = None
    class_num = 8  # MMWHS labels

    # get global test data
    if process_id == 0:
        train_data_global, test_data_global = get_dataloader(None, data_dict['test_all'], batch_size, batch_size, channel_in)
        # logging.info("train_dl_global number = " + str(len(train_data_global)))
        logging.info("test_dl_global number = " + str(len(test_data_global)))
        train_data_local_dict = None
        test_data_local_dict = None
        data_local_num_dict = None
    else:
        # get local dataset
        client_id = process_id - 1
        local_data_num = count_samples(data_dict[client_id], 'train')
        train_data_local, test_data_local = get_dataloader(data_dict[client_id], None, batch_size, batch_size, channel_in)
        if test_data_local:
            logging.info("process_id = %d, local_sample = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
                process_id, local_data_num, len(train_data_local), len(test_data_local)))
        else:
            logging.info("process_id = %d, local_sample = %d, batch_num_train_local = %d" % (
                process_id, local_data_num, len(train_data_local)))

        data_local_num_dict = {client_id: local_data_num}
        train_data_local_dict = {client_id: train_data_local}
        test_data_local_dict = {client_id: test_data_local}
        train_data_global = None
        test_data_global = None
    return train_data_num, train_data_global, test_data_global, data_local_num_dict, train_data_local_dict, \
           test_data_local_dict, class_num


class GeneralDataset(data.Dataset):
    def __init__(self, h5_filepath, channel_in=3, path='train', transforms=None, sample_rate=1.0, has_label=True):
        """
        :param h5_filepath:string h5 file path to load.
        :param transforms: my_transforms
        """
        # logging.info("Load dataset: "+h5_filepath)
        super(GeneralDataset, self).__init__()
        self.h5_filepath = h5_filepath
        self.transforms = transforms
        self.has_label = has_label
        self.channel_in = channel_in
        h5_file = h5py.File(self.h5_filepath, 'r')
        train_db = h5_file[path]
        self.im, self.label, self.keys = build_pairs(train_db, sample_rate)
        h5_file.close()

    def __getitem__(self, index):
        if self.channel_in > 1:
            A = np.repeat(self.label[index], self.channel_in, axis=0).astype("float32")
        else:
            A = np.copy(self.label[index]).astype("float32")
        B = np.copy(self.im[index])
        key = self.keys[index]

        datapoint = {'image': B, 'mask': A, "misc": {
                "index": index,
                "key": key
            }}

        if self.transforms is not None:
            datapoint = self.transforms(datapoint)

        # print('dataloader debug:')
        # print(datapoint['image'].shape, datapoint['image'].max(), datapoint['image'].min(), datapoint['mask'].shape, datapoint['mask'].max(), datapoint['mask'].min())

        return {'A': datapoint['mask'], 'B': datapoint['image'], 'key': key}

    def __len__(self):
        return len(self.im)

