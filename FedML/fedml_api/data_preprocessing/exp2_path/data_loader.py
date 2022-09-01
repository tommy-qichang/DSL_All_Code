import logging

import os

from .data_utility import count_samples
from fedml_api.data_preprocessing.exp2_path import my_transforms
from .base_data_loader import BaseDataLoader
from .Nuclei_seg_dataset import NucleiSegTrainDataset, NucleiSegTestDataset


logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def get_dataloader(h5_train, h5_test, train_bs, test_bs, img_size, validation_split=0.0):
    train_dl = None
    test_dl = None
    if h5_train is not None:
        train_dl = NucleiSegDataLoader(h5_train, train_bs, training=True, validation_split=validation_split, img_size=img_size)
        test_dl = train_dl.split_validation()
    #
    # if h5_test is not None:
    #     test_dl = NucleiSegDataLoader(h5_test, test_bs, shuffle=False, training=False, img_size=img_size)

    return train_dl, test_dl


class NucleiSegDataLoader(BaseDataLoader):
    def __init__(self, h5_filepath, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True,
                 img_size=224):
        self.h5_filepath = h5_filepath
        if training:
            trsfm = my_transforms.Compose([
                my_transforms.RandomResize(0.8, 1.25),
                my_transforms.RandomHorizontalFlip(),
                my_transforms.RandomAffine(0.3),
                my_transforms.RandomRotation(90),
                my_transforms.RandomCrop(img_size),
                my_transforms.LabelEncoding(),
                my_transforms.ToTensor(),
                my_transforms.Normalize((0.7442, 0.5381, 0.6650), (0.1580, 0.1969, 0.1504))]
            )
            self.dataset = NucleiSegTrainDataset(self.h5_filepath, data_transform=trsfm)
        else:
            trsfm = my_transforms.Compose([
                my_transforms.LabelEncoding(),
                my_transforms.ToTensor(),
                my_transforms.Normalize((0.7442, 0.5381, 0.6650), (0.1580, 0.1969, 0.1504))]
            )
            self.dataset = NucleiSegTestDataset(self.h5_filepath, data_transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


# Get a partition map for each client
def partition_data(datadir, partition, n_nets, alpha, datasetname):
    logging.info("*********partition data***************")
    data_dict = dict()
    data_dict['test_all'] = None
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


def load_partition_data_distributed_path(process_id, datasetname, data_dir, partition_method, partition_alpha,
                                         client_number, batch_size):
    data_dict = partition_data(data_dir, partition_method, client_number, partition_alpha, datasetname)
    #logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
    train_data_num = None  #count_samples(data_dict['train_all'], 'train')
    class_num = 2  #  not accurate

    # get global test data
    if process_id == 0:
        train_data_global, test_data_global = get_dataloader(None, None, batch_size, batch_size, 224)
        # logging.info("train_dl_global number = " + str(len(train_data_global)))
        if test_data_global:
            logging.info("test_dl_global number = " + str(len(test_data_global)))
        train_data_local_dict = None
        test_data_local_dict = None
        data_local_num_dict = None
    else:
        # get local dataset
        client_id = process_id - 1
        local_data_num = count_samples(data_dict[client_id], 'images')
        train_data_local, test_data_local = get_dataloader(data_dict[client_id], None, batch_size, batch_size, 224, 0.2)
        logging.info("process_id = %d, local_sample = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
            process_id, local_data_num, len(train_data_local), len(test_data_local)))

        data_local_num_dict = {client_id: local_data_num}
        train_data_local_dict = {client_id: train_data_local}
        test_data_local_dict = {client_id: test_data_local}
        train_data_global = None
        test_data_global = None
    return train_data_num, train_data_global, test_data_global, data_local_num_dict, train_data_local_dict, \
           test_data_local_dict, class_num

