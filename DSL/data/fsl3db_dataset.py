import os.path
import random
import sys

import cv2
import h5py
import numpy as np
import scipy.ndimage as ndimage
from PIL import Image

from data.base_dataset import BaseDataset, get_params, get_transform
from data.brats1db_dataset import Brats1dbDataset
from data.brats_dataset import BratsDataset
from data.fsl1db_dataset import Fsl1dbDataset


class Fsl3dbDataset(BaseDataset):

    def __init__(self, opt):

        self.split_db = []
        for i in range(opt.d_size):
            self.split_db.append(Fsl1dbDataset(opt, i))

    def __getitem__(self, index):

        result = {}
        for k, v in enumerate(self.split_db):
            database = v
            if index >= len(database):
                index = index % len(database)

            index_value = database[index]
            result['A_' + str(k)] = index_value['A']
            result['B_' + str(k)] = index_value['B']
            result['Seg_' + str(k)] = index_value['Seg']
            result['Mask_' + str(k)] = index_value['Mask']
            result['A_paths_' + str(k)] = index_value['A_paths']
            result['B_paths_' + str(k)] = index_value['B_paths']

        return result

    def __len__(self):
        """Return the total number of images in the dataset."""
        length = 0
        for i in self.split_db:
            if len(i) > length:
                length = len(i)

        return length

    def get_center_sizes(self):
        sizes = []
        for i in self.split_db:
            sizes.append(len(i))

        return sizes

