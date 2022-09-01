import numpy as np
import h5py
import importlib
import stringcase
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


def window_image(img, window_center, window_width):
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img = np.clip(img, img_min, img_max)
    return img


def cropImageAt(img, center, size, c_val=None):
    if c_val is None:
        c_val = img.min()
    bbox = [[0, 0], [0, 0], [0, 0]]
    bbox[0][0] = int(center[0] - size[0] // 2)
    bbox[0][1] = bbox[0][0] + size[0]
    bbox[1][0] = int(center[1] - size[1] // 2)
    bbox[1][1] = bbox[1][0] + size[1]
    bbox[2][0] = int(center[2] - size[2] // 2)
    bbox[2][1] = bbox[2][0] + size[2]
    shape_old = img.shape
    pad = [[0, 0], [0, 0], [0, 0]]
    for i in range(3):
        if bbox[i][0] < 0:
            pad[i][0] = -bbox[i][0]
            bbox[i][0] = 0

        if bbox[i][1] > shape_old[i]:
            pad[i][1] = bbox[i][1] - shape_old[i]
    img_new = np.pad(img, pad, 'constant', constant_values=c_val)
    bbox[0][1] = bbox[0][0] + size[0]
    bbox[1][1] = bbox[1][0] + size[1]
    bbox[2][1] = bbox[2][0] + size[2]
    return img_new[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1]]


def resize_keep_spacing(image, mask, load_size=[256, 256]):
    if len(image.shape) == 2:
        h, w = image.shape
        image = np.expand_dims(image, 0)
        image = cropImageAt(image, (0, h // 2, w // 2), (1, load_size[0], load_size[1]))
        image = image[0]
    else:
        c, h, w = image.shape
        image = cropImageAt(image, (c // 2, h // 2, w // 2), (c, load_size[0], load_size[1]))

    if len(mask.shape) == 2:
        # segmentation
        mask = np.expand_dims(mask, 0)
        mask = cropImageAt(mask, (0, h // 2, w // 2), (1, load_size[0], load_size[1]))
        mask = mask[0]
    else:
        c, h, w = mask.shape
        mask = cropImageAt(mask, (c // 2, h // 2, w // 2), (c, load_size[0], load_size[1]))
    return image, mask


def build_pairs(dataset, sample_rate=1, im_size=None):
    keys = list(dataset.keys())
    dcm_arr = []
    label_arr = []

    if 1 > sample_rate > 0.:
        sample_size = max(1, int(len(keys) * sample_rate))
        sample_idx = np.random.choice(len(keys), sample_size, replace=False)
        keys = [keys[i] for i in sample_idx]

    for key in keys:
        dcm = dataset[f"{key}/data"][()]
        label = dataset[f"{key}/label"][()]

        dcm = window_image(dcm, 200, 1000)
        dcm = ((dcm - dcm.min()) / (dcm.max() - dcm.min())) #* 255

        dcm = dcm[np.newaxis, ...]
        label = label[np.newaxis, ...].astype("uint8")

        # preprocess (resize)
        if im_size is not None:
            # new_image, new_mask = resize_keep_spacing(image, mask, im_size)

            dcm = np.moveaxis(dcm, 0, -1)
            label = np.moveaxis(label, 0, -1)

            dcm = resize(dcm, im_size, order=1, preserve_range=True)
            label = resize(label, im_size, order=0, preserve_range=True).astype("uint8")

            dcm = np.moveaxis(dcm, -1, 0)
            label = np.moveaxis(label, -1, 0)

        dcm_arr.append(dcm)
        label_arr.append(label)
    return dcm_arr, label_arr, keys
