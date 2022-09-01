import numpy as np


class ReplaceTensor:
    def __init__(self, image_label, mask_label):
        """
        Unsqueeze any image if axis dimension is not 1. For instance, from (X, Y, Z) to (1, X, Y, Z).

        """
        self.image_label = image_label
        self.mask_label = mask_label

    def __call__(self, sample):
        image, mask, misc = sample[self.image_label], sample[self.mask_label], sample['misc']

        return {'image': image, 'mask': mask, "misc": misc}
