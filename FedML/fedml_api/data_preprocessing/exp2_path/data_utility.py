import numpy as np
import h5py
import importlib
import stringcase

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


def build_pairs(dataset, sample_rate=1.0, im_size=None):
    keys = list(dataset['images'].keys())
    im_arr = []
    label_arr = []

    if 1 > sample_rate > 0.:
        sample_size = max(1, int(len(keys) * sample_rate))
        sample_idx = np.random.choice(len(keys), sample_size, replace=False)
        keys = [keys[i] for i in sample_idx]

    for key in keys:
        im = dataset[f"images/{key}"][()]
        label = dataset[f"labels/{key}"][()]

        im = np.moveaxis(im, -1, 0)
        assert len(label.shape) == 2
        label = label[np.newaxis, ...].astype("uint8")

        if im_size is not None:
            im, label = resize_keep_spacing(im, label, im_size)  # padded center crop

        im_arr.append(im)
        label_arr.append(label)

    return im_arr, label_arr, keys


def build_pairs_complete(dataset, sample_rate):
    dcm_arr = []
    label_arr = []
    labels_ternary_arr = []
    weight_maps_arr = []

    images = dataset['images']
    labels = dataset['labels']
    labels_ternary = dataset['labels_ternary']
    weight_maps = dataset['weight_maps']

    keys = list(images.keys())

    if 1 > sample_rate > 0.:
        sample_size = max(1, int(len(keys) * sample_rate))
        sample_idx = np.random.choice(len(keys), sample_size, replace=False)
        keys = [keys[i] for i in sample_idx]

    for key in keys:
        img = images[key][()]
        label = labels[key][()]
        label_t = labels_ternary[key][()]
        weight_m = weight_maps[key][()]

        dcm_arr.append(np.moveaxis(img, -1, 0))
        assert len(label.shape) == 2
        label_arr.append(label[np.newaxis, ...].astype("uint8"))

        labels_ternary_arr.append(np.moveaxis(label_t, -1, 0))
        weight_maps_arr.append(weight_m)

    return dcm_arr, label_arr, labels_ternary_arr, weight_maps_arr, keys
