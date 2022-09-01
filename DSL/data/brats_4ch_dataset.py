import os.path
import random
import sys

import cv2
import h5py
import numpy as np
import scipy.ndimage as ndimage
from pytictoc import TicToc
import PIL.Image as Image
from data.base_dataset import BaseDataset, get_params, get_transform, get_transform_v2


class Brats4chDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt, idx=None):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        if idx is None:
            h5_name = "General_format_BraTS18_train_2d_4ch.h5"
        else:
            h5_name = f"General_format_BraTS18_train_three_center_{idx}_2d_4ch.h5"
        print(f"Load dataset: {h5_name}")
        self.is_test = False
        self.real_tumor = False
        self.extend_len = 0
        self.multi_label = True

        print(f"Memory GAN using modality:{opt.memory_gan}")
        self.memory_gan_modality = opt.memory_gan
        BaseDataset.__init__(self, opt)
        self.brats_file = h5py.File(os.path.join(opt.dataroot, h5_name), 'r')
        train_db = self.brats_file['train']
        self.dcm, self.label, self.seg, self.keys, self.masks = self.build_pairs(train_db)

        # self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        # self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        assert (self.opt.load_size >= self.opt.crop_size)  # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def build_pairs(self, dataset):
        keys = dataset.keys()
        keys = list(keys)[:5000]
        dcm_arr = []
        label_arr = []
        seg_arr = []
        keys_arr = []
        mask_arr = []

        for idx, key in enumerate(keys):
            print(f"build key:{key}")
            sys.stdout.flush()
            dcm = dataset[f"{key}/data"][()]

            label = dataset[f"{key}/label"][()]
            dcm_list = []
            for ch in range(dcm.shape[0]):
                dcm_list.append(dcm[ch] * ((pow(2, 8) - 1) / (dcm[ch].max() + 1e-8)))
            # dcm = dcm * ((pow(2, 8) - 1) / (dcm.max() + 1e-8))
            dcm = np.stack(dcm_list, axis=0)


            dcm = dcm.astype("uint8")
            seg_label = np.copy(label).astype("uint8")
            seg_label = seg_label * (255 / 4)

            # If skull 3 channel with different modality, the skull had some colorful images which is not right.
            # label = self.merge_skull(np.stack((dcm[1], dcm[1], dcm[1]), 0), label)
            label, mask = self.merge_skull(dcm, label)

            label2, dcm2, seg_label2, mask2 = data_augment_input(label,dcm,seg_label,mask)
            dcm_arr.append(dcm2)
            label_arr.append(label2)
            seg_arr.append(seg_label2)
            mask_arr.append(mask2)
            keys_arr.append(f"{key}")

            if self.is_test:
                # print(f"add new data with augmentation...")
                label2, dcm2, seg_label2, mask2 = data_augment_input(label,dcm,seg_label,mask)
                dcm_arr.append(dcm2)
                label_arr.append(label2)
                seg_arr.append(seg_label2)
                mask_arr.append(mask2)
                keys_arr.append(f"{key}_2")

                label2, dcm2, seg_label2, mask2 = data_augment_input(label,dcm,seg_label,mask)
                dcm_arr.append(dcm2)
                label_arr.append(label2)
                seg_arr.append(seg_label2)
                mask_arr.append(mask2)
                keys_arr.append(f"{key}_3")

        return dcm_arr, label_arr, seg_arr, keys_arr, mask_arr

    def seg_in_skull(self, seg, mask):
        # ndimage.binary_fill_holes(skull_mask)
        seg = mask * seg
        return seg

    def merge_skull(self, skull_mask, slice_label, default_skull_value=5):

        # Add skull structure into label
        skull_mask = ndimage.binary_fill_holes(skull_mask)
        mask = skull_mask.copy().astype("uint8")
        mask[mask>0]=255
        skull_mask = cv2.Laplacian(skull_mask.astype("uint8"), cv2.CV_8U)
        skull_mask[skull_mask > 0] = default_skull_value
        slice_label = slice_label + skull_mask
        slice_label = slice_label * (255 / 4)
        slice_label = np.floor(slice_label)
        slice_label = slice_label.astype("uint8")
        # slice_label[slice_label > 0] = 255

        return slice_label, mask

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        A = self.label[index]
        B = self.dcm[index]
        seg = self.seg[index]
        key = self.keys[index]
        mask = self.masks[index]

        # A = Image.fromarray(np.moveaxis(A,0,-1), mode="CMYK")
        # B = Image.fromarray(np.moveaxis(B,0,-1), mode="CMYK")
        A = Image.fromarray(A, mode="CMYK")
        B = Image.fromarray(B, mode="CMYK")
        seg = Image.fromarray(seg).convert('RGB')
        mask = Image.fromarray(mask, mode="CMYK")

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)

        if not self.opt.isTrain:
            # if True:
            print(f"In Evaluation stage, disable crop_pos and flip augmentation....")
            transform_params['crop_pos'] = (0, 0)
            transform_params['vflip'] = False
            transform_params['hflip'] = False

        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1), method=Image.NEAREST, channel_num=4)
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1), channel_num=4)
        seg_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1), method=Image.NEAREST)
        mask_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1), channel_num=4)

        A = A_transform(A)
        B = B_transform(B)
        seg = seg_transform(seg)
        mask = mask_transform(mask)
        mask[mask<0]=0
        # seg[seg < 0] = 0

        return {'A': A, 'B': B, 'A_paths': str(index), 'B_paths': str(index), 'Seg': seg[:1, :, :], 'Mask': mask,'Key':key}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.dcm)
