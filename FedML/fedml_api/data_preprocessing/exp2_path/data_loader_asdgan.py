import logging

import torch.utils.data as data
import torch
import numpy as np
import h5py
import os
# from skimage.transform import resize

from .data_utility import init_transform, count_samples, build_pairs


logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def get_transforms_G():
    # transforms_train = ["RandomCrop", "RandomFlip"]
    # transforms_args_train = {
    #     "RandomCrop": [256],
    #     "RandomFlip": [True, True]
    # }

    transforms_test = ["CenterCrop", "ToTensorScale", "Normalize"]
    transforms_args_test = {
        "CenterCrop": [256],
        "ToTensorScale": ['float', 255, 255],
        "Normalize": [0.5, 0.5]  # only normalize image
    }

    # return init_transform(transforms_train, transforms_args_train), init_transform(transforms_test, transforms_args_test)
    return None, init_transform(transforms_test, transforms_args_test)


def get_transforms_D():
    transforms_train = ["ToTensorScale", "Normalize"]
    transforms_args_train = {
        # "RandomCrop": [256],
        # "RandomFlip": [True, True],
        "ToTensorScale": ['float', 255, 255],
        "Normalize": [0.5, 0.5]  # only normalize image
    }

    return init_transform(transforms_train, transforms_args_train), None


def get_dataloader_G(h5_train, h5_test, train_bs, test_bs, sample_method, channel_in=1):
    transform_train, transform_test = get_transforms_G()

    train_ds = DatasetG(sample_method=sample_method,
                        transforms=transform_train,
                        channel_in=channel_in)

    if h5_test is not None:
        test_ds = TestDataset(h5_test,
                              channel_in=channel_in,
                              sample_rate=0.01,
                              transforms=transform_test)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=False, pin_memory=True)
    else:
        test_dl = None

    return train_ds, test_dl


def get_dataloader_D(h5_train, h5_test, train_bs, test_bs, channel_in=1):
    transform_train, transform_test = get_transforms_D()

    train_ds = DatasetD(h5_train,
                        im_size=(256, 256),  # None when random augmentation in transforms_G
                        channel_in=channel_in,
                        # sample_rate=0.1,  # debug
                        transforms=transform_train)

    test_dl = None

    return train_ds, test_dl


# Get a partition map for each client
def partition_data(datadir, partition, n_clients):
    logging.info("*********partition data***************")
    data_dict = dict()
    data_dict['test_all'] = os.path.join(datadir, 'train_all.h5')
    data_dict['train_all'] = None

    if partition == "homo":
        ## todo
        raise('Not Implemented Error')

    # non-iid data distribution
    elif partition == "hetero":
        data_dict[0] = os.path.join(datadir, 'train_breast.h5')
        data_dict[1] = os.path.join(datadir, 'train_kidney.h5')
        data_dict[2] = os.path.join(datadir, 'train_liver.h5')
        data_dict[3] = os.path.join(datadir, 'train_prostate.h5')

    return data_dict


def load_partition_data_distributed_path(process_id, dataset, data_dir, partition_method,
                                         client_number, batch_size, sample_method, channel_in=1, mask_input=None, mask_noise=None):
    data_dict = partition_data(data_dir, partition_method, client_number)
    #logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
    train_data_num = None

    class_num = 2  # nuclei or background, in fact nuclei have different labels (instance level)

    # get mask data for central server
    if process_id == 0:
        train_data_global, test_data_global = get_dataloader_G(None, data_dict['test_all'], batch_size, batch_size, sample_method, channel_in)
        # logging.info("train_dl_global number = " + str(len(train_data_global)))
        # logging.info("test_dl_global number = " + str(len(test_data_global)))
        train_data_local_dict = None
        test_data_local_dict = None
        data_local_num_dict = None
    else:
        # get local dataset
        client_id = process_id - 1
        local_data_num = count_samples(data_dict[client_id], 'images')
        train_data_local, test_data_local = get_dataloader_D(data_dict[client_id], None, batch_size, batch_size, channel_in)
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


class DatasetG(data.Dataset):
    def __init__(self, sample_method='uniform', transforms=None, channel_in=1):
        """
        sample_method:  'uniform' or 'balance'.
                        'uniform': sample all labels from all clients equally.
                        'balance': sample same amount of data from each client
        :param transforms: my_transforms
        """
        super(DatasetG, self).__init__()
        self.sample_method = sample_method
        self.transforms = transforms
        self.label_dict = {}
        self.key_dict = {}
        self.count_dict = {}
        self.key_2_id_dict = {}
        self.all_key_list = []
        self.channel_in = channel_in

    def add_client_labels(self, client_id, labels, keys):
        assert(len(labels) == len(keys))
        self.label_dict[client_id] = labels
        self.key_dict[client_id] = keys
        self.count_dict[client_id] = len(keys)

        old_len = len(self.all_key_list)
        self.all_key_list.extend(list(keys))
        for idx, key in enumerate(keys):
            self.key_2_id_dict[key] = idx + old_len

    def get_key_str(self, id):
        return self.all_key_list[id]

    def get_key_str_list(self, ids):
        return [self.all_key_list[i] for i in ids]

    def transform_A(self, A, key):
        datapoint = {'mask': np.copy(A), "misc": {"key": key}}

        if self.transforms is not None:
            transform_para = []
            datapoint = self.transforms(datapoint)
            transform_funcs = self.transforms.transforms
            for func in transform_funcs:
                transform_para.append(datapoint['misc'][type(func).__name__])
        else:
            transform_para = [0]

        return datapoint['mask'], transform_para

    def __getitem__(self, index):

        if self.sample_method == 'uniform':  # return one sample
            client_index = 0
            for client_id, cnt in self.count_dict.items():
                if index < cnt:
                    client_index = client_id
                    break
                else:
                    index -= cnt
            A = self.label_dict[client_index][index]
            key = self.key_dict[client_index][index]
            A, transform_para = self.transform_A(A, key)
            As = [A]
            keys_id = [self.key_2_id_dict[key]]
            client_ids = [client_index]
            trans_paras = [transform_para]

        else:  # 'balance': return N samples, each from one of N clients. oversampling from clients with small dataset
            client_ids = []
            As = []
            keys_id = []
            trans_paras = []
            for client_id, cnt in self.count_dict.items():
                client_ids.append(client_id)
                A = self.label_dict[client_id][index % cnt]
                key = self.key_dict[client_id][index % cnt]
                A, transform_para = self.transform_A(A, key)
                As.append(A)
                keys_id.append(self.key_2_id_dict[key])
                trans_paras.append(transform_para)

        As = np.array(As, dtype=np.float32)
        As = As / 255.0

        if self.channel_in > As.shape[1]:
            As = np.repeat(As, self.channel_in, axis=1)

        # return torch.from_numpy(np.array(client_ids)), torch.tensor(As, dtype=torch.float32), torch.tensor(np.array(keys_id))

        return torch.from_numpy(np.array(client_ids, dtype=np.uint8)), torch.from_numpy(As), \
               torch.from_numpy(np.array(keys_id, dtype=np.int16)), torch.from_numpy(np.array(trans_paras, dtype=np.uint8))

    def __len__(self):
        if self.sample_method == 'uniform':
            cnt_all = 0
            for k, cnt in self.count_dict.items():
                cnt_all += cnt
            return cnt_all
        else:
            cnt_max = 0
            for k, cnt in self.count_dict.items():
                cnt_max = max(cnt_max, cnt)
            return cnt_max


class DatasetD:
    def __init__(self, h5_filepath, im_size=None, transforms=None, sample_rate=1, channel_in=1):
        """
        :param h5_filepath:string h5 file path to load.
        :param transforms: my_transforms
        :param paths: array of all keys which will loaded
        """
        # logging.info("Load dataset: "+h5_filepath)
        # super(DatasetD, self).__init__()
        self.h5_filepath = h5_filepath
        self.transforms = transforms
        self.channel_in = channel_in
        h5_file = h5py.File(self.h5_filepath, 'r')
        self.dcm, self.label, self.keys = build_pairs(h5_file, sample_rate, im_size)
        h5_file.close()
        self.index_dict = dict(zip(self.keys, np.arange(len(self.keys))))
        # self.im_size = im_size

    def collect_label(self):
        return self.keys, self.label

    def get_data(self, key_batch, transform_params):
        data_batch = []
        label_batch = []
        for key, trans_para in zip(key_batch, transform_params):
            index = self.index_dict[key]
            if self.channel_in > 1:
                A = np.repeat(self.label[index], self.channel_in, axis=0).astype("float32")
            else:
                A = np.copy(self.label[index]).astype("float32")
            B = np.copy(self.dcm[index])

            if len(trans_para) == 2:
                datapoint = {'image': B, 'mask': A, "misc": {"RandomCrop": trans_para[0], "RandomFlip": trans_para[1]}}
            else:
                datapoint = {'image': B, 'mask': A, "misc": {"index": index, "key": key}}

            if self.transforms is not None:
                datapoint = self.transforms(datapoint)
            data_batch.append(datapoint['image'])
            label_batch.append(datapoint['mask'])

        return torch.stack(data_batch), torch.stack(label_batch)

    def __len__(self):
        return len(self.dcm)


class TestDataset(data.Dataset):
    def __init__(self, h5_filepath, transforms=None, sample_rate=0.5, channel_in=1):
        """
        :param h5_filepath:string h5 file path to load.
        :param transforms: my_transforms
        :param paths: array of all keys which will loaded
        """
        # logging.info("Load dataset: "+h5_filepath)
        # super(TestDataset, self).__init__()
        self.h5_filepath = h5_filepath
        self.transforms = transforms
        self.channel_in = channel_in
        h5_file = h5py.File(self.h5_filepath, 'r')
        self.dcm, self.label, self.keys = build_pairs(h5_file, sample_rate)
        h5_file.close()

    def __getitem__(self, index):
        key = self.keys[index]
        if self.channel_in > 1:
            A = np.repeat(self.label[index], self.channel_in, axis=0).astype("float32")
        else:
            A = np.copy(self.label[index]).astype("float32")
        B = np.copy(self.dcm[index])

        datapoint = {'image': B, 'mask': A, "misc": {
                "index": index,
                "key": key
            }}

        if self.transforms is not None:
            datapoint = self.transforms(datapoint)

        return {'A': datapoint['mask'], 'B': datapoint['image'], 'key': key}

    def __len__(self):
        return len(self.dcm)
