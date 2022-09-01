"""
subsample slices and shift the short axis and long axis slices
"""
import cv2
import numpy as np
import random
from scipy import ndimage
from scipy.ndimage.interpolation import shift


class ExpRandomShiftNaive:
    """
    Subsample the 3D numpy [h, w, d] by slice max number.
    """

    def __init__(self, random_shift_rate=0.8, random_shift_range=10, sample_number=10, perturbation=True):
        self.perturbation = perturbation

        self.sample_number = sample_number
        self.channel_number = 3
        self.random_shift_rate = random_shift_rate
        self.random_shift_range = random_shift_range

    def move_mass_to_center(self, mask):
        h, w, d = mask.shape
        ch, cw, cd = ndimage.measurements.center_of_mass(mask)
        mask = np.roll(mask, (int(round(h // 2 - ch)), int(round(w // 2 - cw))), axis=(0, 1))
        return mask

    def __call__(self, sample):
        # Here we just use mask instead of image as input. Be careful if you use this script otherwise!
        mask, misc = sample['mask'], sample['misc']
        mask = self.move_mass_to_center(mask)

        mri_img = mask[:, :, ::self.sample_number]
        h, w, d = mri_img.shape

        image = np.zeros((self.channel_number, h, w, d))

        image[1] = np.copy(mri_img)
        image[2] = np.copy(mri_img)

        offset_arr = []

        for i in range(d):
            slice_mri_img = mri_img[:, :, i]

            random_seed = random.random()
            offset_x = offset_y = 0
            if self.perturbation and random_seed <= self.random_shift_rate:
                offset_x = random.randint(-self.random_shift_range, self.random_shift_range)
                offset_y = random.randint(-self.random_shift_range, self.random_shift_range)
                image[0, :, :, i] = shift(slice_mri_img, (offset_x, offset_y), cval=0, mode='nearest')
                # image[0, :, :, i] = shift(slice_mri_img, (offset_x, offset_y), cval=0, mode='nearest')
            else:
                image[0, :, :, i] = slice_mri_img
                # image[0, :, :, i] = slice_mri_img

            offset_arr.append([offset_x, offset_y])

            dt_slice_mask = ndimage.distance_transform_edt(1 - image[0, :, :, i])
            image[0, :, :, i] = dt_slice_mask

            # image[1, :, :, i] = ndimage.distance_transform_edt(1 - image[1, :, :, i])
            # image[2, :, :, i] = ndimage.distance_transform_edt(1 - image[2, :, :, i])

        # offset_arr = np.array(offset_arr) / (119 / 2)
        offset_arr = np.array(offset_arr) / 10

        # offset_arr = offset_arr * 3

        return {'image': image, 'mask': np.array(offset_arr), 'misc': misc}
