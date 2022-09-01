"""
subsample slices and shift the short axis and long axis slices
"""
import cv2
import numpy as np
import random
from scipy import ndimage
from scipy.ndimage.interpolation import shift


class ExpRandomShiftSxlx:
    """
    Subsample the 3D numpy [h, w, d] by slice max number.
    """

    def __init__(self, random_shift_rate=0.8, random_shift_range=10, sample_number=10, perturbation=True,
                 skull_mask=False):
        self.perturbation = perturbation

        self.sample_number = sample_number
        self.channel_number = 3
        self.random_shift_rate = random_shift_rate
        self.random_shift_range = random_shift_range
        self.skull_mask = skull_mask

    def move_mass_to_center(self, mask):
        h, w, d = mask.shape
        ch, cw, cd = ndimage.measurements.center_of_mass(mask)
        mask = np.roll(mask, (int(round(h // 2 - ch)), int(round(w // 2 - cw))), axis=(0, 1))
        return mask

    def __call__(self, sample):
        # Here we just use mask instead of image as input. Be careful if you use this script otherwise!
        mask, misc = sample['mask'], sample['misc']
        mask = self.move_mass_to_center(mask)

        mri_img = mask[:, :, ::self.sample_number].astype('long')
        h, w, d = mri_img.shape

        image = np.zeros((self.channel_number, h, w, d))
        # misc['orig_image'] = np.copy(mri_img)
        # misc['orig_dt'] = np.zeros_like(mri_img)
        orig_image = np.zeros_like(mri_img)
        orig_dt = np.zeros_like(mri_img)

        offset_arr = []

        for i in range(d):
            slice_mri_img = mri_img[:, :, i]

            slice_skull_mask = cv2.Laplacian(slice_mri_img.astype("uint8"), cv2.CV_8U)
            slice_skull_mask[slice_skull_mask > 0] = 1
            orig_image[:, :, i] = slice_skull_mask

            dt_slice_mask = ndimage.distance_transform_edt(1 - slice_skull_mask)
            # misc['orig_dt'][:, :, i] = dt_slice_mask
            # orig_dt[:, :, i] = dt_slice_mask

            random_seed = random.random()
            offset_x = offset_y = 0
            if self.perturbation and random_seed <= self.random_shift_rate:
                offset_x = random.randint(-self.random_shift_range, self.random_shift_range)
                offset_y = random.randint(-self.random_shift_range, self.random_shift_range)
                if self.skull_mask:
                    image[0, :, :, i] = shift(slice_skull_mask, (offset_x, offset_y), cval=0, mode='nearest')
                else:
                    image[0, :, :, i] = shift(slice_mri_img, (offset_x, offset_y), cval=0, mode='nearest')
            else:
                # image[0, :, :, i] = slice_skull_mask
                image[0, :, :, i] = slice_mri_img

            offset_arr.append([offset_x, offset_y])

            if image[0, :, :, i].sum() == 0:
                dt_slice_mask = image[0, :, :, i]
            else:
                dt_slice_mask = ndimage.distance_transform_edt(1 - image[0, :, :, i])
            image[0, :, :, i] = dt_slice_mask

        # offset_arr = np.array(offset_arr) / (119 / 2)
        if self.random_shift_range >0:
            offset_arr = np.array(offset_arr) / self.random_shift_range
        else:
            offset_arr = np.array(offset_arr)


        if self.skull_mask:
            image[1, h // 2, :, :] = orig_image[h // 2, :, :]
            image[2, :, w // 2, :] = orig_image[:, w // 2, :]
        else:
            image[1, h // 2, :, :] = mri_img[h // 2, :, :]
            image[2, :, w // 2, :] = mri_img[:, w // 2, :]
        # image[1] = orig_image
        # image[2] = orig_image

        return {'image': image, 'mask': np.array(offset_arr), 'misc': misc}
