import logging

import torch.utils.data as data
import numpy as np
import h5py
import os

from .data_utility import init_transform, count_samples, get_img_list_h5


logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def get_transforms():
    transforms_train = ["FilterLabel", "Resize", "RandomCrop", "RandomFlip", "HistNorm", "ToTensor", "NormalizeChannel"]
    transforms_args_train = {
        "FilterLabel": [[1, 2, 3, 4], [1, 1, 1, 1]],
        "Resize": [286],
        "RandomCrop": [256],
        "RandomFlip": [True, True]
    }

    transforms_test = ["FilterLabel", "CropPadding", "HistNorm", "ToTensor", "NormalizeChannel"]
    transforms_args_test = {
        "CropPadding": [256],
        "FilterLabel": [[1, 2, 3, 4], [1, 1, 1, 1]]
    }

    return init_transform(transforms_train, transforms_args_train), init_transform(transforms_test, transforms_args_test)


def get_dataloader(h5_train, h5_test, train_bs, test_bs, missing_channel=None):
    transform_train, transform_test = get_transforms()

    if h5_train is not None:
        train_ds = GeneralDataset(h5_train,
                                  paths=["train"],
                                  missing_channel=missing_channel,
                                  transforms=transform_train)
        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=True, pin_memory=True)
    else:
        train_dl = None

    if h5_test is not None:
        test_ds = GeneralDataset(h5_test,
                                 paths=["val"],
                                 missing_channel=missing_channel,
                                 transforms=transform_test)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=False, pin_memory=True)
    # elif h5_train is not None:
    #     test_ds = GeneralDataset(h5_train,
    #                              paths=["train"],
    #                              sample_rate=0.1,
    #                              missing_channel=missing_channel,
    #                              transforms=transform_test)
    #     test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=False, pin_memory=True)
    else:
        test_dl = None

    return train_dl, test_dl


# Get a partition map for each client
def partition_data(datadir, partition, n_nets, alpha, datasetname):
    logging.info("*********partition data***************")
    data_dict = dict()
    data_dict['test_all'] = None
    data_dict['train_all'] = None  #os.path.join(datadir, 'General_format_BraTS18_train_2d_4ch.h5')
    # if 'exp2' in datasetname:
    #     data_dict['test_all'] = os.path.join(datadir, 'exp2_test.h5')  # T1, T2, Flair
    #     data_dict['train_all'] = os.path.join(datadir, 'exp2_train.h5')
    # else:
        # data_dict['test_all'] = os.path.join(datadir, 'General_format_BraTS18_test_2d_3ch_new.h5')  # T1c, T2, Flair
        # data_dict['train_all'] = os.path.join(datadir, 'General_format_BraTS18_train_2d_3ch_new.h5')

    if partition == "homo":
        ## todo
        raise('Not Implemented Error')
        # total_num = n_train
        # idxs = np.random.permutation(total_num)
        # batch_idxs = np.array_split(idxs, n_nets)  # As many splits as n_nets = number of clients
        # net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}

    # non-iid data distribution
    elif partition == "hetero":
        data_dict[0] = os.path.join(datadir, 'General_format_BraTS18_train_three_center_0_2d_4ch')
        data_dict[1] = os.path.join(datadir, 'General_format_BraTS18_train_three_center_1_2d_4ch')
        data_dict[2] = os.path.join(datadir, 'General_format_BraTS18_train_three_center_2_2d_4ch')
        # if 'exp2' in datasetname:
        #     data_dict[0] = os.path.join(datadir, 'General_format_BraTS18_train_three_center_0_2d_3ch.h5')  # T1, T2, Flair
        #     data_dict[1] = os.path.join(datadir, 'General_format_BraTS18_train_three_center_1_2d_3ch.h5')
        #     data_dict[2] = os.path.join(datadir, 'General_format_BraTS18_train_three_center_2_2d_3ch.h5')
        # else:
        #     data_dict[0] = os.path.join(datadir, 'General_format_BraTS18_train_three_center_2d_3ch_new_0.h5')  # T1c, T2, Flair
        #     data_dict[1] = os.path.join(datadir, 'General_format_BraTS18_train_three_center_2d_3ch_new_1.h5')
        #     data_dict[2] = os.path.join(datadir, 'General_format_BraTS18_train_three_center_2d_3ch_new_2.h5')

    return data_dict


def load_partition_data_distributed_brats(process_id, datasetname, data_dir, partition_method, partition_alpha,
                                         client_number, batch_size):
    data_dict = partition_data(data_dir, partition_method, client_number, partition_alpha, datasetname)
    #logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
    train_data_num = None  #count_samples(data_dict['train_all'], 'train')
    class_num = 2  # tumor and normal

    # get global test data
    if process_id == 0:
        train_data_global, test_data_global = None, None #get_dataloader(None, data_dict['test_all'], batch_size, batch_size)
        # logging.info("train_dl_global number = " + str(len(train_data_global)))
        if test_data_global:
            logging.info("test_dl_global number = " + str(len(test_data_global)))
        train_data_local_dict = None
        test_data_local_dict = None
        data_local_num_dict = None
    else:
        # get local dataset
        client_id = process_id - 1
        # ['T1', 'T2', 'Flair', 'T1c']

        if 'miss' in datasetname:
            missing_channel = client_id
            if client_id == 0:    # other
                missing_channel = 1  # miss T2
            elif client_id == 1:  # CBICA
                missing_channel = 2  # miss Flair
            elif client_id == 2:  # TCIA
                missing_channel = 3  # miss T1c
        else:
            missing_channel = None
        local_data_num = count_samples(data_dict[client_id]+'_trainset.h5', 'train')
        train_data_local, test_data_local = get_dataloader(data_dict[client_id]+'_trainset.h5', data_dict[client_id]+'_valset.h5', batch_size, batch_size, missing_channel=missing_channel)
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
    def __init__(self, h5_filepath, transforms=None, paths=["train"], sample_rate=1.0, missing_channel=None):
        """
        :param h5_filepath:string h5 file path to load.
        :param transforms: my_transforms
        :param paths: array of all keys which will loaded
        """
        # logging.info("Load dataset: "+h5_filepath)
        super(GeneralDataset, self).__init__()
        self.h5_filepath = h5_filepath
        self.transforms = transforms
        # print(h5_filepath, paths)
        self.img_list = get_img_list_h5(h5_filepath, paths)
        # print('len of img_list:', len(self.img_list))
        self.h5_file = h5py.File(self.h5_filepath, 'r')
        self.missing_channel = missing_channel
        if 1 > sample_rate > 0.:
            sample_size = max(1, int(len(self.img_list)*sample_rate))
            sample_idx = np.random.choice(len(self.img_list), sample_size, replace=False)
            self.img_list = [self.img_list[i] for i in sample_idx]

    def __getitem__(self, index):
        img_path, label_path = self.img_list[index]

        if img_path not in self.h5_file:
            print(img_path, label_path)
        img = self.h5_file[img_path][()]
        label = self.h5_file[label_path][()]

        datapoint = {'image': img, 'mask': label, "misc": {
                "index": index,
                "img_path": img_path,
                "label_path": label_path
            }}

        if self.transforms is not None:
            datapoint = self.transforms(datapoint)

        if self.missing_channel is not None:
            datapoint['image'][self.missing_channel] = datapoint['image'][self.missing_channel] * 0

        return {'image': datapoint['image'], 'label': datapoint['mask']}

    def __len__(self):
        return len(self.img_list)
