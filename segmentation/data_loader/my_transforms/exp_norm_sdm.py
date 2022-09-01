"""
subsample slices and shift the short axis and long axis slices
"""
import cv2
import numpy as np
import random
from scipy import ndimage
from scipy.ndimage.interpolation import shift


class ExpNormSdm:
    """
    Normalized the signed distance map to [-1,1]
    """

    def __call__(self, sample):
        # image: 1 x 256 x 256 x 13, mask: 3 x 256 x 256 x 13
        image, mask, misc = sample['image'], sample['mask'], sample['misc']
        sdm_neg = mask[1:2]
        sdm_pos = mask[2:]
        sdm = (sdm_neg - np.min(sdm_neg)) / (np.max(sdm_neg) - np.min(sdm_neg)+ 1e-7) - (sdm_pos - np.min(sdm_pos)) / (
                    np.max(sdm_pos) - np.min(sdm_pos) + 1e-7)

        new_mask = np.concatenate((mask[:1], sdm))
        return {'image': image, 'mask': new_mask, 'misc': misc}

