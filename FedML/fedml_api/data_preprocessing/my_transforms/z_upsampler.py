"""
subsample slices for MICCAI 2020 Reconstruction project.
"""
import numpy as np
import random
from scipy.ndimage.interpolation import shift

class ZUpsampler:
    """
    Upsample the 3D numpy label [c, h, w, d] to more dense volume. [c, h, w, d*10].
    """

    def __init__(self):

        self.sample_size_propability = [1, 7, 13, 28, 19, 29, 3]
        self.sample_numbers = [5, 6, 7, 8, 9, 10, 11]

    def __call__(self, sample):
        # Here we just use mask instead of image as input. Be careful if you use this script otherwise!
        image, mask, misc = np.copy(sample['mask']), sample['mask'], sample['misc']
        # image, mask, misc = sample['mask'], sample['mask'], sample['misc']

        h, w, d = image.shape

        input = np.zeros((h, w, d*5+10))
        for i in range(d):
            input[:,:, 5 + i*5] = image[:,:,i]

        # input = np.expand_dims(input, 0)

        return {'image': input, 'mask': input, 'misc': misc}
