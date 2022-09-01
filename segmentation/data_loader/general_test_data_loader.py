"""
General Data Loader and General Dataset
General purpose data loader and dataset.
## GPL License
## Author: Qi Chang<qc58@cs.rutgers.edu>
"""
import h5py
import math
import numpy as np
from data_loader.general_data_loader import GeneralDataset
from skimage.util import view_as_windows
from torch.utils import data
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler


class GeneralTestDataLoader(DataLoader):
    """
    General data loader, which try to cover the most of the dataset loader requirements,
    so you do not need to customize data loader and dataset.
    If you are interested in using this general dataloader directly,
    the HDF5 dataset must follow the format below:
    1. type 1 train/validation in one file: one hdf5 contains "train"/"val" two keys, which has all records for training and
    validation. In each record, has "data"/"label" indicates data(dimension: ch x h x w (xd)) and label .
    eg: data_dir="xxx.h5", validation_split=[["train"],["val"]]
    2. type 2 train/validation in separate hdf5 files: two hdf5 contains keys: "train" and "val" separately.
    eg: data_dir=["xxx_train.h5","xxx_val.h5"], validation_split=[["train"],["val"]]
    3. type 3 train/split validation in one file with percentage: one hdf5 contains "train" key, validation_split is a
    float number which indicates the percentage of splitting training dataset as train and validation set.
    eg: data_dir="xxx.h5", validation_split=0.2
    4. type 4 cross-validation in one file with subset datasets: one hdf5 contains multiple keys,
    for instance ["1","2","3","4","5"], and the validation_split is a two dimension array indicates which subsets are
    for training, which subsets are for testing.
    eg: data_dir="xxx.h5", validation_split=[["1","2","3","4"],["5"]]
    """

    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers, collate_fn=default_collate,
                 training=False, transforms=None, has_label=True):
        """
        :param data_dir: str|array: String should be the source of h5 file for GeneralDataset.
                                    Array should contain 2 string elements, contain the path to the training('train')
                                     and validation('val').
        :param batch_size:
        :param shuffle:
        :param num_workers:
        :param collate_fn:
        """
        self.root_path = "test"

        assert isinstance(data_dir, str), "The test path should only be string"
        self.test_dataset = GeneralDataset(data_dir, transforms=transforms, paths=["test"])
        # In test, should not shuffle at all.
        self.shuffle = shuffle
        self.init_kwargs = {
            'dataset': self.test_dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }
        self.shuffle = shuffle

        super().__init__(**self.init_kwargs)
