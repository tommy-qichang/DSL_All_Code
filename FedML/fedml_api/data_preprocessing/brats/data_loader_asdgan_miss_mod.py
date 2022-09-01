import logging

import torch.utils.data as data
import torch
import numpy as np
import h5py
import os

from .data_utility import init_transform, count_samples, build_pairs


logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def get_transforms_G():
    transforms_train = ["RandomCrop", "RandomFlip"]
    transforms_args_train = {
        "RandomCrop": [256],
        "RandomFlip": [True, True]
    }

    transforms_test = ["Resize", "ToTensorScale", "Normalize"]
    transforms_args_test = {
        "Resize": [256],
        "ToTensorScale": ['float', 255, 5],
        "Normalize": [0.5, 0.5]  # only normalize image
    }

    return init_transform(transforms_train, transforms_args_train), init_transform(transforms_test, transforms_args_test)


def get_transforms_D():
    transforms_train = ["RandomCrop", "RandomFlip", "ToTensorScale", "Normalize"]
    transforms_args_train = {
        "RandomCrop": [256],
        "RandomFlip": [True, True],
        "ToTensorScale": ['float', 255, 5],
        "Normalize": [0.5, 0.5]  # only normalize image
    }

    return init_transform(transforms_train, transforms_args_train), None


def get_dataloader_G(h5_train, h5_test, train_bs, test_bs, sample_method, channel=None, channel_in=1, use_brain_mask=False, noise_rate=0.):
    transform_train, transform_test = get_transforms_G()

    train_ds = DatasetG(sample_method=sample_method,
                        transforms=transform_train,
                        channel_in=channel_in,
                        use_brain_mask=use_brain_mask,
                        noise_rate=noise_rate)

    if h5_test is not None:
        test_ds = TestDataset(h5_test,
                              channel=channel,
                              channel_in=channel_in,
                              path="train",
                              sample_rate=0.01,
                              transforms=transform_test,
                              use_brain_mask=use_brain_mask)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=False, num_workers=1, pin_memory=True)
    else:
        test_dl = None

    return train_ds, test_dl


def get_dataloader_D(h5_train, h5_test, train_bs, test_bs, channel=None, channel_in=1, use_brain_mask=False):
    transform_train, transform_test = get_transforms_D()

    train_ds = DatasetD(h5_train,
                        im_size=(286, 286),
                        channel=channel,
                        channel_in=channel_in,
                        # sample_rate=0.1,  # debug
                        transforms=transform_train,
                        use_brain_mask=use_brain_mask)

    test_dl = None

    return train_ds, test_dl


# Get a partition map for each client
def partition_data(datadir, partition, n_clients):
    logging.info("*********partition data***************")
    data_dict = dict()
    data_dict['test_all'] = os.path.join(datadir, 'General_format_BraTS18_train_2d_4ch.h5')  # ['T1', 'T2', 'Flair', 'T1c']
    data_dict['train_all'] = None

    if partition == "homo":
        ## todo
        raise ('Not Implemented Error')

    # non-iid data distribution
    elif partition == "hetero":
        data_dict[0] = os.path.join(datadir, 'General_format_BraTS18_train_three_center_0_2d_4ch.h5')
        data_dict[1] = os.path.join(datadir, 'General_format_BraTS18_train_three_center_1_2d_4ch.h5')
        data_dict[2] = os.path.join(datadir, 'General_format_BraTS18_train_three_center_2_2d_4ch.h5')

    return data_dict


def load_partition_data_distributed_brats_mm(process_id, dataset, data_dir, partition_method,
                                         client_number, batch_size, sample_method, channel_in=1, use_brain_mask=False, noise_rate=0.):
    data_dict = partition_data(data_dir, partition_method, client_number)
    #logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
    train_data_num = None

    class_num = 6  # tumor and skull and normal

    # get mask data for central server
    if process_id == 0:
        if 't2' in dataset:
            channel = 1
        elif 't1' in dataset:
            channel = 0
        elif 'flair' in dataset:
            channel = 2
        else:
            channel = None

        train_data_global, test_data_global = get_dataloader_G(None, data_dict['test_all'], batch_size, batch_size, sample_method, channel, channel_in, use_brain_mask, noise_rate)
        # logging.info("train_dl_global number = " + str(len(train_data_global)))
        # logging.info("test_dl_global number = " + str(len(test_data_global)))
        train_data_local_dict = None
        test_data_local_dict = None
        data_local_num_dict = None
    else:
        # get local dataset
        client_id = process_id - 1
        # ['T1', 'T2', 'Flair', 'T1c']
        if client_id == 0:
            channel = [0, 2, 3]
        elif client_id == 1:
            channel = [0, 1, 3]
        elif client_id == 2:
            channel = [0, 1, 2]
        else:
            channel = None
        local_data_num = count_samples(data_dict[client_id], 'train')
        train_data_local, test_data_local = get_dataloader_D(data_dict[client_id], None, batch_size, batch_size, channel, channel_in, use_brain_mask)
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
    def __init__(self, sample_method='uniform', transforms=None, use_brain_mask=False, noise_rate=0., channel_in=1):
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
        self.use_brain_mask = use_brain_mask
        self.noise_rate = noise_rate
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
        As = As / 255.0  #As.max()

        if self.use_brain_mask and np.random.rand() < self.noise_rate:  # add noise u
            label_brain = np.min(As[As>0])
            noise_u = np.random.normal(0, label_brain/3, As.shape)
            noise_u[noise_u > label_brain*2/3] = label_brain*2/3
            As[As==label_brain] = As[As==label_brain] + noise_u[As==label_brain]

        if self.channel_in > As.shape[1]:
            As = np.repeat(As, self.channel_in, axis=1)

        # return torch.tensor(np.array(client_ids)), torch.tensor(As, dtype=torch.float32), torch.tensor(np.array(keys_id))

        return torch.tensor(np.array(client_ids, dtype=np.uint8)), torch.tensor(As, dtype=torch.float32), \
               torch.tensor(np.array(keys_id, dtype=np.int16)), torch.tensor(np.array(trans_paras, dtype=np.uint8))

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
    def __init__(self, h5_filepath, im_size=(256, 256), channel=None, transforms=None, path="train", sample_rate=1, use_brain_mask=False, channel_in=1):
        """
        :param h5_filepath:string h5 file path to load.
        :param transforms: my_transforms
        :param paths: array of all keys which will loaded
        """
        # logging.info("Load dataset: "+h5_filepath)
        # super(DatasetD, self).__init__()
        self.h5_filepath = h5_filepath
        self.transforms = transforms
        self.channel = channel
        self.channel_in = channel_in
        h5_file = h5py.File(self.h5_filepath, 'r')
        train_db = h5_file[path]
        self.dcm, self.label, self.keys = build_pairs(train_db, channel, sample_rate, use_brain_mask=use_brain_mask, im_size=im_size)
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
    def __init__(self, h5_filepath, channel=None, transforms=None, path="train", sample_rate=0.5, use_brain_mask=False, channel_in=1):
        """
        :param h5_filepath:string h5 file path to load.
        :param transforms: my_transforms
        :param paths: array of all keys which will loaded
        """
        # logging.info("Load dataset: "+h5_filepath)
        # super(TestDataset, self).__init__()
        self.h5_filepath = h5_filepath
        self.transforms = transforms
        self.channel = channel
        self.channel_in = channel_in
        h5_file = h5py.File(self.h5_filepath, 'r')
        train_db = h5_file[path]
        self.dcm, self.label, self.keys = build_pairs(train_db, channel, sample_rate, use_brain_mask=use_brain_mask)
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
