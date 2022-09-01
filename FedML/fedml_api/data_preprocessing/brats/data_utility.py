import numpy as np
import h5py
import importlib
import stringcase
import scipy.ndimage as ndimage
import cv2
from skimage.transform import resize
from fedml_api.data_preprocessing.my_transforms.compose import Compose


def init_transform(transforms, transforms_args):
    transform_instances = []
    for module_name in transforms:
        transform = importlib.import_module(f"fedml_api.data_preprocessing.my_transforms.{stringcase.snakecase(module_name)}")
        if module_name in transforms_args:
            transform_arg = transforms_args[module_name]
        else:
            transform_arg = []

        instance = getattr(transform, module_name)(*transform_arg)
        transform_instances.append(instance)

    compose = Compose(transform_instances)
    return compose


def mr_normalization(img, bd_low=0.1, bd_up=99.9, mean_i=None, std_i=None):
    # exclude some outlier intensity if necessary
    # print('norm: ', np.min(img), np.max(img), bd_low, np.percentile(img, bd_low), bd_up, np.percentile(img, bd_up), np.mean(img), np.std(img))
    img[img>np.percentile(img, bd_up)] = np.percentile(img, bd_up)
    img[img<np.percentile(img, bd_low)] = np.percentile(img, bd_low)

    if mean_i is not None and std_i is not None:
        factor_scale = np.std(img)/std_i
        img = img / factor_scale
        factor_shift = np.mean(img) - mean_i
        img = img - factor_shift

    return img


def count_samples(h5_filepath, path='train'):
    count = 0
    with h5py.File(h5_filepath, 'r') as h5_file:
        count = len(h5_file[path].keys())

    return count


def get_img_list_h5(h5_filepath, paths):
    img_list = []

    with h5py.File(h5_filepath, 'r') as h5_file:
        for path in paths:
            for key in h5_file[path].keys():
                img_list.append((f"{path}/{key}/data", f"{path}/{key}/label"))

    return img_list


def merge_skull_brats(skull_mask, slice_label, default_skull_value=5):

    # Add skull structure into label
    skull_mask = ndimage.binary_fill_holes(skull_mask)
    skull_mask = cv2.Laplacian(skull_mask.astype("uint8"), cv2.CV_8U)  # edge detection
    skull_mask[skull_mask > 0] = default_skull_value
    slice_label = slice_label + skull_mask
    # slice_label = slice_label * (255 / np.max(slice_label))  # max is default_skull_value

    return slice_label


def merge_brain_mask_brats(brain_mask, slice_label, default_brain_value=1):

    # Add brain mask into label
    brain_mask = ndimage.binary_fill_holes(brain_mask)
    brain_mask[brain_mask > 0] = default_brain_value
    slice_label = slice_label + brain_mask
    # slice_label = slice_label * (255 / 5)  # max value of label in brats is 4

    return slice_label


def build_pairs(dataset, channel=None, sample_rate=1, use_brain_mask=False, im_size=None):
    keys = list(dataset.keys())
    dcm_arr = []
    label_arr = []
    # seg_arr = []

    if 1 > sample_rate > 0.:
        sample_size = max(1, int(len(keys) * sample_rate))
        sample_idx = np.random.choice(len(keys), sample_size, replace=False)
        keys = [keys[i] for i in sample_idx]

    for key in keys:
        dcm = dataset[f"{key}/data"][()]
        label = dataset[f"{key}/label"][()]
        if channel is None:
            ## normalize per channel (T1/T1c/T2/Flair)
            for i in range(dcm.shape[0]):
                dcm[i] = mr_normalization(dcm[i])
                dcm[i] = dcm[i] * (255 / (dcm[i].max() + 1e-8))
        else:
            if type(channel) is int:
                dcm = mr_normalization(dcm[channel])
                dcm = dcm[np.newaxis, ...] * (255 / (dcm.max() + 1e-8))
            elif type(channel) is list:
                dcm_new = np.zeros([len(channel)]+list(dcm.shape[1:]))
                for i, ch in enumerate(channel):
                    dcm_new[i] = mr_normalization(dcm[ch])
                    dcm_new[i] = dcm_new[i] * (255 / (dcm_new[i].max() + 1e-8))
                dcm = dcm_new
            else:
                assert(type(channel) is int)
        # dcm = dcm.astype("uint8")
        # seg_label = np.copy(label).astype("uint8")
        # seg_label = seg_label * (255 / 4)

        # If different modality has different signal region, the multi-channel label may be inconsistent. use the first-channel dcm only.
        if use_brain_mask:
            label = merge_brain_mask_brats((dcm[0]>0).astype('uint8'), label)
        else:  # skull mask
            label = merge_skull_brats((dcm[0]>0).astype('uint8'), label)

        label = label[np.newaxis, ...].astype("uint8")

        if im_size is not None:
            dcm = np.moveaxis(dcm, 0, -1)
            label = np.moveaxis(label, 0, -1)

            dcm = resize(dcm, im_size, order=1, preserve_range=True)
            label = resize(label, im_size, order=0, preserve_range=True).astype("uint8")

            dcm = np.moveaxis(dcm, -1, 0)
            label = np.moveaxis(label, -1, 0)

        dcm_arr.append(dcm)
        label_arr.append(label)
        # seg_arr.append(seg_label)

    return dcm_arr, label_arr, keys
