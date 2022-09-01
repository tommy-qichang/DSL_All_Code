import json
import os
from collections import OrderedDict
from itertools import repeat
from pathlib import Path

import numpy as np
import pandas as pd
import re
from functools import singledispatch

import scipy
from scipy.ndimage.filters import gaussian_filter
from types import SimpleNamespace
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg
import PIL.Image as Image

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)


class StringConverter:

    @staticmethod
    def underscore_to_camelcase(underscore: str):
        class_name_handler = list(underscore)
        for gp in re.finditer(r'(^[a-z])|(?:_([a-z]))', underscore):
            class_name_handler[gp.end() - 1] = class_name_handler[gp.end() - 1].upper()

        class_name_handler = "".join(class_name_handler)
        return class_name_handler.replace("_", "")

    @staticmethod
    def camelcase_to_underscore(camelcase: str):
        raise NotImplementedError()


def show_figures(imgs, new_flag=False):
    import matplotlib.pyplot as plt
    if new_flag:
        for i in range(len(imgs)):
            plt.figure()
            plt.imshow(imgs[i])
    else:
        for i in range(len(imgs)):
            plt.figure(i + 1)
            plt.imshow(imgs[i])

    plt.show()


@singledispatch
def dict2obj(o):
    return o


@dict2obj.register(dict)
def handle_obj(obj):
    return SimpleNamespace(**{k: dict2obj(v) for k, v in obj.items()})


@dict2obj.register(list)
def handle_list(lst):
    return [dict2obj(i) for i in lst]


def smooth_obj_one_case(orig_array):
    """
    Smooth label array[x, y, z], with scale or size settings.
    :param orig_array:
    :param resize_type:
    :param scale:
    :return:
    """
    scale_array = []
    for i in range(orig_array.shape[-1]):
        slice = orig_array[:, :, i]
        final_mask = np.zeros_like(slice)
        unique_id = np.unique(slice)[1:].tolist()
        if len(unique_id) == 0:
            scale_array.append(slice)
            continue
        for label_id in unique_id:
            show_img = np.zeros_like(slice)
            show_img[slice == label_id] = 1

            # blurred_img = gaussian_filter(show_img, sigma=4)
            blurred_img = gaussian_filter(show_img, sigma=4)
            blurred_img[blurred_img > 0.5] = 1
            blurred_img[blurred_img <= 0.5] = 0
            final_mask[blurred_img == 1] = label_id
        scale_array.append(final_mask)

    return np.stack(scale_array, axis=-1)


'''
    For the given path, get the List of all files in the directory tree 
'''


def getListOfFiles(dirName, file_type=[]):
    # create a list of file and sub directories
    # names in the given directory
    assert type(file_type) == list, "file_type should be a list."
    listOfFile = os.listdir(dirName)
    allFiles = list()
    allFilesName = []
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        allFilesName.append(entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath, file_type)[0]
        elif len(file_type) == 0:
            allFiles.append(fullPath)
        elif any(x.lower() in fullPath.lower() for x in file_type):
            allFiles.append(fullPath)
    return allFiles, allFilesName

def compute_sdf(img_gt, out_shape, border_zero=False):
    """
    compute the signed distance map of binary mask
    input: segmentation, shape = (batch_size, x, y, z) outshape=(batch_size,2, x, y, z)
    output: the Signed Distance Map (SDM)
    sdf(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation
    normalize sdf to [-1,1]
    """

    img_gt = img_gt.astype(np.uint8)
    normalized_sdf = np.zeros(out_shape)

    for b in range(out_shape[0]):
        for d in range(out_shape[-1]):
            if len(img_gt.shape) == len(out_shape):
                posmask = img_gt[b,0,:,:,d].astype(np.bool)
            else:
                posmask = img_gt[b, :, :, d].astype(np.bool)
            if posmask.any():
                negmask = ~posmask
                posdis = distance(posmask)
                negdis = distance(negmask)

                if border_zero:
                    boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
                    posdis[boundary == 1] = 0

                # boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
#                     sdf = (negdis-np.min(negdis))/(np.max(negdis)-np.min(negdis)) - (posdis-np.min(posdis))/(np.max(posdis)-np.min(posdis))
#                     sdf[boundary==1] = 0
                normalized_sdf[b,0,:,:,d] = negdis
                normalized_sdf[b,1,:,:,d] = posdis

#                     assert np.min(sdf) == -1.0, print(np.min(posdis), np.max(posdis), np.min(negdis), np.max(negdis))
#                     assert np.max(sdf) ==  1.0, print(np.min(posdis), np.min(negdis), np.max(posdis), np.max(negdis))
            else:
                normalized_sdf[b,0,:,:,d] = 1
    return normalized_sdf


def flood_fill_hull(image):
    points = np.transpose(np.where(image))
    hull = scipy.spatial.ConvexHull(points)
    deln = scipy.spatial.Delaunay(points[hull.vertices])
    idx = np.stack(np.indices(image.shape), axis = -1)
    out_idx = np.nonzero(deln.find_simplex(idx) + 1)
    out_img = np.zeros(image.shape)
    out_img[out_idx] = 1
    return out_img, hull

def convert_to_uint8( image, drange=[0,1]):
    image = adjust_dynamic_range(image, drange, [0,255])
    image = np.rint(image).clip(0, 255).astype(np.uint8)
    return image

def adjust_dynamic_range(data, drange_in, drange_out):
    if drange_in != drange_out:
        scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (np.float32(drange_in[1]) - np.float32(drange_in[0]))
        bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
        data = data * scale + bias
    return data


# def compute_sdf(img_gt, out_shape):
#     """
#     compute the signed distance map of binary mask
#     input: segmentation, shape = (batch_size, x, y, z)
#     output: the Signed Distance Map (SDM)
#     sdf(x) = 0; x in segmentation boundary
#              -inf|x-y|; x in segmentation
#              +inf|x-y|; x out of segmentation
#     normalize sdf to [-1,1]
#     """
#
#     img_gt = img_gt.astype(np.uint8)
#     normalized_sdf = np.zeros(out_shape)
#
#     for b in range(out_shape[0]):  # batch size
#         for c in range(out_shape[1]):
#             posmask = img_gt[b].astype(np.bool)
#             if posmask.any():
#                 negmask = ~posmask
#                 posdis = distance(posmask)
#                 negdis = distance(negmask)
#                 boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
#                 sdf = (negdis - np.min(negdis)) / (np.max(negdis) - np.min(negdis)) - (posdis - np.min(posdis)) / (
#                             np.max(posdis) - np.min(posdis))
#                 sdf[boundary == 1] = 0
#                 normalized_sdf[b][c] = sdf
#                 assert np.min(sdf) == -1.0, print(np.min(posdis), np.max(posdis), np.min(negdis), np.max(negdis))
#                 assert np.max(sdf) == 1.0, print(np.min(posdis), np.min(negdis), np.max(posdis), np.max(negdis))
#
#     return normalized_sdf
