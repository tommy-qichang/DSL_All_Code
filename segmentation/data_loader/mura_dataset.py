import os

import h5py
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class MuraDataset(Dataset):
    MURA_FILE = "mura.hdf5"
    CANDIDATE_TYPE = ["ALL", "ELBOW", "FINGER", "FOREARM", "HAND", "HUMERUS", "SHOULDER", "WRIST"]

    def __init__(self, data_root, data_type="WRIST", train=True, transforms=None):
        self.mura_file = h5py.File(os.path.join(data_root, self.MURA_FILE), 'r')
        if data_type == self.CANDIDATE_TYPE[-1]:
            # try WRIST first
            self.neg_data = self.mura_file["WRIST/0"]
            self.pos_data = self.mura_file["WRIST/1"]
            self.neg_keys = list(self.neg_data.keys())
            self.pos_keys = list(self.pos_data.keys())
            self.transforms = transforms

    def __getitem__(self, item):
        idx = item + self.start_idx
        idx = str(idx)
        if idx in self.pos_keys:
            result = self.pos_data[idx][()], 1
        else:
            result = self.neg_data[idx][()], 0

        result = Image.fromarray(result[0]), result[1]

        if self.transforms is not None:
            return self.transforms(result[0]), result[1]

    def __len__(self):
        self.start_idx = min(int(self.neg_keys[0]), int(self.pos_keys[0]))
        self.end_idx = max(int(self.neg_keys[-1]), int(self.pos_keys[-1]))
        return self.end_idx - self.start_idx + 1
