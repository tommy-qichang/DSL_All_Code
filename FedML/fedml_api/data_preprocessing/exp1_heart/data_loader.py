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
    transforms_train = ["Windowfilter","Unsqueeze","Resize", "RandomCrop", "RandomFlip","ToTensor","NormalizeMinmax","Normalize"]
    transforms_args_train = {
        "Windowfilter": [200, 1000],
        "Resize": [286],
        "RandomCrop": [256],
        "RandomFlip": [True, True],
        "NormalizeMinmax": [0, 255],
        "Normalize": [[127.5], [127.5]]
    }

    transforms_test = ["Windowfilter","Unsqueeze","CropPadding", "ToTensor","NormalizeMinmax","Normalize"]
    transforms_args_test = {
        "Windowfilter": [200, 1000],
        "CropPadding": [256],
        "NormalizeMinmax": [0, 255],
        "Normalize": [[127.5], [127.5]]
    }

    return init_transform(transforms_train, transforms_args_train), init_transform(transforms_test, transforms_args_test)


def get_dataloader(h5_train, h5_test, train_bs, test_bs):
    transform_train, transform_test = get_transforms()

    train_dl = None
    test_dl = None

    if h5_test is not None:
        test_ds = GeneralDataset(h5_test,
                                 paths=["val"],
                                 transforms=transform_test)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=False, pin_memory=True)
        if h5_train is not None:
            train_ds = GeneralDataset(h5_train,
                                      paths=["train"],
                                      transforms=transform_train)
            train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=True, pin_memory=True)
    elif h5_train is not None:
            train_ds = GeneralDataset(h5_train,
                                      paths=["train"],
                                      sample_rate=0.8,
                                      transforms=transform_train)
            train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=True, pin_memory=True)

            test_sample_id_list = train_ds.get_exclude_list()
            test_ds = GeneralDataset(h5_train,
                                     paths=["train"],
                                     sample_id_list=test_sample_id_list,
                                     transforms=transform_test)
            test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=False, pin_memory=True)

    return train_dl, test_dl


# Get a partition map for each client
def partition_data(datadir, partition, n_nets, alpha, datasetname):
    logging.info("*********partition data***************")
    data_dict = dict()
    data_dict['test_all'] = None  #os.path.join(datadir, 'whs_ct_train_2d_iso.h5')
    data_dict['train_all'] = None

    if partition == "homo":
        ## todo
        raise('Not Implemented Error')

    # non-iid data distribution
    elif partition == "hetero":
        data_dict[0] = [os.path.join(datadir, 'seg_asoca_train_2d_iso.h5'), os.path.join(datadir, 'seg_asoca_val_2d_iso.h5')]
        data_dict[1] = [os.path.join(datadir, 'seg_miccai2008_train_2d_iso.h5'), os.path.join(datadir, 'seg_miccai2008_val_2d_iso.h5')]
        data_dict[2] = [os.path.join(datadir, 'seg_whs_ct_train_2d_iso.h5'), os.path.join(datadir, 'seg_whs_ct_val_2d_iso.h5')]

    return data_dict


def load_partition_data_distributed_heart(process_id, datasetname, data_dir, partition_method, partition_alpha,
                                         client_number, batch_size):
    data_dict = partition_data(data_dir, partition_method, client_number, partition_alpha, datasetname)
    #logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
    train_data_num = None  #count_samples(data_dict['train_all'], 'train')
    class_num = 8  # tumor and normal

    # get global test data
    if process_id == 0:
        train_data_global, test_data_global = None, None  #get_dataloader(None, data_dict['test_all'], batch_size, batch_size)
        # logging.info("train_dl_global number = " + str(len(train_data_global)))
        if test_data_global:
            logging.info("test_dl_global number = " + str(len(test_data_global)))
        train_data_local_dict = None
        test_data_local_dict = None
        data_local_num_dict = None
    else:
        # get local dataset
        client_id = process_id - 1
        local_data_num = count_samples(data_dict[client_id][0], 'train')
        train_data_local, test_data_local = get_dataloader(data_dict[client_id][0], data_dict[client_id][1], batch_size, batch_size)
        logging.info("process_id = %d, local_sample = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
            process_id, local_data_num, len(train_data_local), len(test_data_local)))

        data_local_num_dict = {client_id: local_data_num}
        train_data_local_dict = {client_id: train_data_local}
        test_data_local_dict = {client_id: test_data_local}
        train_data_global = None
        test_data_global = None
    return train_data_num, train_data_global, test_data_global, data_local_num_dict, train_data_local_dict, \
           test_data_local_dict, class_num


class GeneralDataset(data.Dataset):
    def __init__(self, h5_filepath, transforms=None, paths=["train"], sample_rate=1.0, sample_id_list=None):
        """
        :param h5_filepath:string h5 file path to load.
        :param transforms: my_transforms
        :param paths: array of all keys which will loaded
        """
        # logging.info("Load dataset: "+h5_filepath)
        super(GeneralDataset, self).__init__()
        self.h5_filepath = h5_filepath
        self.transforms = transforms
        self.img_list = get_img_list_h5(h5_filepath, paths)
        self.h5_file = h5py.File(self.h5_filepath, 'r')
        self.exclude_id_list = []
        if sample_id_list:
            self.img_list = [self.img_list[i] for i in sample_id_list]
        elif 1 > sample_rate > 0.:
            sample_size = max(1, int(len(self.img_list)*sample_rate))
            sample_idx = np.random.choice(len(self.img_list), sample_size, replace=False)
            self.exclude_id_list = [i for i in np.arange(len(self.img_list)) if i not in sample_idx]
            self.img_list = [self.img_list[i] for i in sample_idx]

    def get_exclude_list(self):
        return self.exclude_id_list

    def __getitem__(self, index):
        img_path, label_path = self.img_list[index]

        img = self.h5_file[img_path][()]
        label = self.h5_file[label_path][()]

        datapoint = {'image': img, 'mask': label, "misc": {
                "index": index,
                "img_path": img_path,
                "label_path": label_path
            }}

        if self.transforms is not None:
            datapoint = self.transforms(datapoint)

        return {'image': datapoint['image'], 'label': datapoint['mask']}

    def __len__(self):
        return len(self.img_list)
