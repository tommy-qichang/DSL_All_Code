"""
subsample slices and shift the short axis and long axis slices
"""
import cv2
import numpy as np
import random
from scipy import ndimage
from scipy.ndimage.interpolation import shift


class ExpDtToSeg:
    """
    Select Short axis image and convert the distance transform to binary mask.
    """

    def __init__(self, sa_slices_num=13, hist_clamp=False, include_lachannel=0):
        self.sa_slices_num = sa_slices_num
        self.hist_clamp_enable = hist_clamp
        self.include_lachannel = include_lachannel

    def __call__(self, sample):
        # Here we just use mask instead of image as input.
        # Be careful if you use this script otherwise!
        image, mask, misc = sample['image'],sample['mask'], sample['misc']

        if self.include_lachannel > 0:
            sax_image = image
            laxch = image.shape[0] -1
            if laxch >= self.include_lachannel:
                lax_list = list(range(1, laxch+1))
                lax_indexes = [0] + sorted(random.sample(lax_list, k=self.include_lachannel))
                mask = mask[lax_indexes]
                sax_image = sax_image[lax_indexes]
            else:
                mask = np.pad(mask, ((0, self.include_lachannel - laxch), (0, 0), (0, 0), (0, 0)))
                sax_image = np.pad(image, ((0, self.include_lachannel - laxch), (0, 0), (0, 0), (0, 0)))

        else:
            sax_image = image[0:1]
        # Select Short Axis slices
        sax_mask = mask[0:1]
        laxch, sax_num, x, y = sax_mask.shape
        if sax_num < self.sa_slices_num:
            sax_mask = np.pad(sax_mask, ((0, 0), (0, self.sa_slices_num - sax_num), (0, 0), (0, 0)))
            sax_image = np.pad(sax_image, ((0, 0), (0, self.sa_slices_num - sax_num), (0, 0), (0, 0)))
        elif sax_num > self.sa_slices_num:
            start_idx = np.random.randint(0, sax_num - self.sa_slices_num)
            sax_mask = sax_mask[:, start_idx:(start_idx + self.sa_slices_num)]
            sax_image = sax_image[:, start_idx:(start_idx + self.sa_slices_num)]

        sax_new_mask = np.zeros_like(sax_mask)
        sax_new_mask[sax_mask<0.5] = 1

        if self.hist_clamp_enable:
            sax_image = self.hist_clamp(sax_image)

        return {'image': sax_image.astype("float"), 'mask': sax_new_mask, 'misc': misc}

    def hist_clamp(self, image):
        new_image = []
        bd_up = 99.9
        bd_low = 0.1
        for i in range(image.shape[0]):
            sub_img = image[i]
            sub_img[sub_img > np.percentile(sub_img, bd_up)] = np.percentile(sub_img, bd_up)
            sub_img[sub_img < np.percentile(sub_img, bd_low)] = np.percentile(sub_img, bd_low)
            new_image.append(sub_img)
        new_image = np.stack(new_image, axis=0)
        return new_image



