import os.path
import random
import sys

import cv2
import h5py
import numpy as np
import scipy.ndimage as ndimage
from PIL import Image
import torchvision.transforms.functional as F
from pytictoc import TicToc

from data.base_dataset import BaseDataset, get_params, get_transform
from util.util import window_image


def data_augment_input(label, dcm, seg_label, mask, just_crop=False):
    label = Image.fromarray(np.copy(label))
    dcm = Image.fromarray(np.copy(dcm))
    seg = Image.fromarray(np.copy(seg_label)).convert('RGB')
    mask = Image.fromarray(np.copy(mask))

    # if random.random() > 0.5 and not just_crop:
    #     # Vertical flip
    #     label = label.transpose(Image.FLIP_TOP_BOTTOM)
    #     dcm = dcm.transpose(Image.FLIP_TOP_BOTTOM)
    #     seg = seg.transpose(Image.FLIP_TOP_BOTTOM)
    #     mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
    # if random.random() > 0.5 and not just_crop:
    #     # Horizontal flip
    #     label = label.transpose(Image.FLIP_LEFT_RIGHT)
    #     dcm = dcm.transpose(Image.FLIP_LEFT_RIGHT)
    #     seg = seg.transpose(Image.FLIP_LEFT_RIGHT)
    #     mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() > 0.8 and not just_crop:
        deg = random.randint(-15,15)
        label = label.rotate(deg)
        dcm = dcm.rotate(deg)
        seg = seg.rotate(deg)
        mask = mask.rotate(deg)

    load_size = 286
    crop_size = 256
    if random.random() > 0.8 and not just_crop:
        load_size = load_size + random.randint(-20,20)
    label = label.resize((load_size,load_size))
    dcm = dcm.resize((load_size,load_size))
    seg = seg.resize((load_size,load_size))
    mask = mask.resize((load_size,load_size))

    start_x = 0
    start_y = 0
    if random.random() > 0.5 and not just_crop:
        start_x = random.randint(0, load_size - crop_size)
        start_y = random.randint(0, load_size - crop_size)

    label = label.crop((start_x, start_y, start_x + crop_size, start_y + crop_size))
    dcm = dcm.crop((start_x, start_y, start_x + crop_size, start_y + crop_size))
    seg = seg.crop((start_x, start_y, start_x + crop_size, start_y + crop_size))
    mask = mask.crop((start_x, start_y, start_x + crop_size, start_y + crop_size))

    return np.array(label),np.array(dcm),np.array(seg),np.array(mask)


class Fsl1dbDataset(BaseDataset):
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
            # if opt.modality == "mri":
            #     h5_name = "fsl_ACDC_mri_iso_train_2d.h5"
            # elif opt.modality == "ct":
            #     h5_name = "fsl_whs_ct_train_2d.h5"
            # elif opt.modality == "mri2":
            #     h5_name = "fsl_whs_mri_train_2d.h5"
            if opt.modality == "ct1":
                h5_name = "whs_ct_train_2d_iso.h5"
            elif opt.modality == "ct2":
                h5_name = "miccai2008_train_2d_iso.h5"
            elif opt.modality == "ct3":
                h5_name = "asoca_train_2d_iso.h5"
            elif opt.modality == "ct":
                h5_name = "ct_all_train_2d_iso.h5"
            # elif opt.modality == "mri":
            #     h5_name = "whs_mri_train_2d_iso_new.h5"

            # Check #83
            # h5_name = "General_format_BraTS18_train_2d_1ch_new.h5"
            # h5_name = "General_format_BraTS18_train_2d_3ch.h5"
            # h5_name = "General_format_BraTS18_train_2d_3ch_new.h5"
            # h5_name = "General_format_BraTS18_train_2d_4ch.h5"
        elif idx == 0:

            # h5_name = f"General_format_BraTS18_train_tree_center_{idx}_2d_1ch_new.h5"
            h5_name = "whs_ct_train_2d_iso.h5"
        elif idx == 1:
            h5_name = "miccai2008_train_2d_iso.h5"
        elif idx == 2:
            h5_name = "asoca_train_2d_iso.h5"
        # elif idx == 3:
        #     h5_name = "whs_mri_train_2d_iso_new.h5"
            # h5_name = f"General_format_BraTS18_train_three_center_{idx}_2d_3ch.h5"
            # h5_name = f"General_format_BraTS18_train_random_split_{idx}_1ch.h5"
            # h5_name = f"General_format_BraTS18_train_three_center_{idx}_2d_4ch.h5"
        print(f"Load dataset for fsl1: {h5_name}")
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

    def build_pairs(self, dataset, is_ct=True):
        keys = dataset.keys()
        keys = list(keys)[:1000]
        dcm_arr = []
        label_arr = []
        seg_arr = []
        keys_arr = []
        mask_arr = []

        for idx, key in enumerate(keys):
            print(f"build key:{key}")
            sys.stdout.flush()
            dcm = dataset[f"{key}/data"][()]
            if is_ct:
                dcm = window_image(dcm, 200, 1000)
            # Just choose T2 modality.
            # if idx == 0:
            #     print(f"#####Warning: Just use T2 modality....#####")
            # dcm = np.stack((dcm[1], dcm[1], dcm[1]))

            label = dataset[f"{key}/label"][()]
            dcm_list = []
            if len(dcm.shape)==3:
                for ch in range(dcm.shape[0]):
                    dcm_list.append(dcm[ch] * ((pow(2, 8) - 1) / (dcm[ch].max() + 1e-8)))
                # dcm = dcm * ((pow(2, 8) - 1) / (dcm.max() + 1e-8))
                dcm = np.stack(dcm_list, axis=0)
            else:
                dcm = ((dcm - dcm.min()) / (dcm.max() - dcm.min())) * (pow(2, 8)-1)


            ## normalize per channel (T1/T1c/T2/Flair)
            # for i in range(dcm.shape[0]):
            #     dcm[i] = dcm[i] * ((pow(2, 8) - 1) / (dcm[i].max() + 1e-8))

            # ch1:mean:109.81857307086526, std:198.0402913849317|
            # ch2:mean:107.5048806918867, std:203.84568207898917|
            # ch3:mean:64.17097195910556, std:118.93537507329793

            # mean = [110, 107, 64]
            # std = [198, 203, 118]
            # dcm = F.normalize(dcm, mean, std)

            dcm = dcm.astype("uint8")
            seg_label = np.copy(label).astype("uint8")
            seg_label = seg_label * (255 / 10)

            # If skull 3 channel with different modality, the skull had some colorful images which is not right.
            # label = self.merge_skull(np.stack((dcm[1], dcm[1], dcm[1]), 0), label)
            _, mask = self.merge_skull(dcm, label)

            label = seg_label

            label2, dcm2, seg_label2, mask2 = data_augment_input(label,dcm,seg_label,mask)
            dcm_arr.append(dcm2)
            label_arr.append(label2)
            seg_arr.append(seg_label2)
            keys_arr.append(f"{key}")
            mask_arr.append(mask2)
            # dcm_arr.append(dcm)
            # label_arr.append(label)
            # seg_arr.append(seg_label)
            # keys_arr.append(f"{key}")
            # mask_arr.append(mask)

            # # DEBUG:vis
            #
            # img_A_j = (label * (255/label.max())).astype(np.uint8)
            # img_B_j = (dcm * (255/dcm.max())).astype(np.uint8)
            # img_A_j = np.moveaxis(img_A_j, 0, -1)
            # img_B_j = np.moveaxis(img_B_j, 0, -1)
            #
            # img_A_j = Image.fromarray(img_A_j).convert('RGB')
            # img_B_j = Image.fromarray(img_B_j).convert('RGB')
            # img_A_j.save(f"/ajax/users/qc58/work/projects/AsynDGANv2/imgs/ccA_{key}.jpg")
            # img_B_j.save(f"/ajax/users/qc58/work/projects/AsynDGANv2/imgs/ccB_{key}.jpg")



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

                # factor = 4
                # add_dcm_arr = []
                # add_label_arr = []
                # add_seg_arr = []
                # if factor >= 1:
                    #
                    # for i in range(len(dcm_arr)):
                    #     dcm = dcm_arr[i]
                    #
                    #     times = 1
                    #
                    #     skull_mask = np.zeros_like(dcm)
                    #     skull_mask[dcm > 0] = 1
                    #     while times <= factor:
                    #         random_id = random.randrange(0, len(dcm_arr) - 1, 10)
                    #         seg = seg_arr[random_id]
                    #
                    #
                    #         # seg[seg > 0] = 1
                    #         seg = self.seg_in_skull(seg, skull_mask)
                    #         seg = seg.astype("uint8")
                    #
                    #         seg_mask = np.copy(seg)
                    #         seg_mask[seg_mask>0]=1
                    #         if np.sum(seg_mask) < 10:
                    #             continue
                    #         label = np.copy(seg)
                    #         label = label * 4 / 255
                    #
                    #         label = self.merge_skull(skull_mask, label)
                    #         # label[label > 0] = 255
                    #
                    #
                    #         add_dcm_arr.append(dcm)
                    #         add_label_arr.append(label)
                    #         add_seg_arr.append(seg)
                    #         times += 1
                    #     print(f'append syn label:{i}')
                    # print(f"##debug: orig_dcm last length:{len(dcm_arr)}")
                    # dcm_arr = dcm_arr + add_dcm_arr
                    # label_arr = label_arr + add_label_arr
                    # seg_arr = seg_arr + add_seg_arr
                    # print(f"##debug: updated_dcm last length:{len(dcm_arr)}")

        return dcm_arr, label_arr, seg_arr, keys_arr, mask_arr

    def seg_in_skull(self, seg, mask):
        # ndimage.binary_fill_holes(skull_mask)
        seg = mask * seg
        return seg

    def merge_skull(self, skull_mask, slice_label, default_skull_value=8):
        # stack_ch_masks = []
        # # Fix the multi-modality image skull will have some color strip which is not right.
        # for i in range(skull_mask.shape[0]):
        #     # Add skull structure into label
        #     skull_mask_i = skull_mask[i]
        #     skull_mask_i = ndimage.binary_fill_holes(skull_mask_i)
        #     skull_mask_i = cv2.Laplacian(skull_mask_i.astype("uint8"), cv2.CV_8U)
        #     skull_mask_i[skull_mask_i > 0] = default_skull_value
        #     stack_ch_masks.append(skull_mask_i)
        #
        # skull_mask_final = np.stack(stack_ch_masks, 0)
        # slice_label = slice_label + skull_mask_final

        # Add skull structure into label
        skull_mask = ndimage.binary_fill_holes(skull_mask)
        mask = skull_mask.copy().astype("uint8")
        mask[mask>0]=255
        skull_mask = cv2.Laplacian(skull_mask.astype("uint8"), cv2.CV_8U)
        skull_mask[skull_mask > 0] = default_skull_value
        slice_label = slice_label + skull_mask
        slice_label = slice_label * (255 / 10)
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
        # A = Image.fromarray(np.moveaxis(A,0,-1)).convert('RGB')
        # B = Image.fromarray(np.moveaxis(B,0,-1)).convert('RGB')
        if len(A.shape)==2:
            A = Image.fromarray(A).convert('RGB')
            B = Image.fromarray(B).convert('RGB')
        else:
            A = Image.fromarray(np.moveaxis(A,0,-1)).convert('RGB')
            B = Image.fromarray(np.moveaxis(B,0,-1)).convert('RGB')
        seg = Image.fromarray(seg).convert('RGB')
        mask = Image.fromarray(mask).convert('RGB')

        # read a image given a random integer index
        # AB_path = self.AB_paths[index]
        # AB = Image.open(AB_path).convert('RGB')
        # # split AB image into A and B
        # w, h = AB.size
        # w2 = int(w / 2)
        # A = AB.crop((0, 0, w2, h))
        # B = AB.crop((w2, 0, w, h))

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)

        # Watch out! only applicable for test.
        if not self.opt.isTrain:
        # if True:
            print(f"In Evaluation stage, disable crop_pos and flip augmentation....")
            transform_params['crop_pos'] = (0, 0)
            transform_params['vflip'] = False
            transform_params['hflip'] = False

        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1), method=Image.NEAREST)
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))
        seg_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1), method=Image.NEAREST)
        mask_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1), method=Image.NEAREST)

        A = A_transform(A)
        B = B_transform(B)
        seg = seg_transform(seg)
        mask = mask_transform(mask)
        mask[mask<0]=0
        # seg[seg < 0] = 0

        return {'A': A, 'B': B, 'A_paths': str(index), 'B_paths': str(index), 'Seg': seg[:1, :, :], 'Mask': mask, 'Key':key}

    def __len__(self):
        """Return the total number of images in the dataset."""
        # return 50
        return len(self.dcm)
