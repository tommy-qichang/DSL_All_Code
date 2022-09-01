"""
subsample slices for MICCAI 2020 Reconstruction project.
"""
import numpy as np
import random
from scipy.ndimage.interpolation import shift

class RandomSample:
    """
    Subsample the 3D numpy [c, h, w, d] by slice max number.
    """

    def __init__(self, perturbation=False):

        self.sample_size_propability = [1, 7, 13, 28, 19, 29, 3]
        self.sample_numbers = [5, 6, 7, 8, 9, 10, 11]
        self.perturbation = perturbation

    def __call__(self, sample):
        # Here we just use mask instead of image as input. Be careful if you use this script otherwise!
        image, mask, misc = np.copy(sample['mask']), sample['mask'], sample['misc']
        # image, mask, misc = sample['mask'], sample['mask'], sample['misc']

        chance = random.randint(1, 100)
        sample_number = 10
        for i in range(len(self.sample_numbers)):
            if chance <= sum(self.sample_size_propability[:(i + 1)]):
                sample_number = self.sample_numbers[i]
                break

        # c, h, w, d = image.shape
        image = np.expand_dims(image, 0)

        # Will find the slice index range for label 1
        range_h = np.where(mask == 1)[2]
        range_h_min = range_h.min() + random.randint(0, 10)
        range_h_max = range_h.max() - random.randint(0, 10)
        length_h = range_h_max - range_h_min + 1
        intervel_num = length_h // sample_number

        image[:, :, :, :range_h_min] = 0
        cur_h_idx = range_h_min
        while cur_h_idx < range_h_max:
            if self.perturbation:
                random_seed = random.random()
                if random_seed >=0.9:
                    offset_x = random.randint(-20,20)
                    offset_y = random.randint(-20,20)
                    image[0, :, :, cur_h_idx] = shift(image[0, :, :,cur_h_idx], (offset_x, offset_y), cval=0)
            image[:, :, :, (cur_h_idx + 1):cur_h_idx + intervel_num] = 0
            cur_h_idx += intervel_num
        image[:, :, :, range_h_max + 1:] = 0

        return {'image': image, 'mask': mask, 'misc': misc}
