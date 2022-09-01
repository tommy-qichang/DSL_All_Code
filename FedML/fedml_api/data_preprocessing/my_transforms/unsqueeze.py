import numpy as np


class Unsqueeze:
    def __init__(self, axis=0, apply_to_label=False):
        """
        Unsqueeze any image if axis dimension is not 1. For instance, from (X, Y, Z) to (1, X, Y, Z).

        """
        self.axis = axis
        self.apply_to_label = apply_to_label

    def __call__(self, sample):
        image, mask, misc = sample['image'], sample['mask'], sample['misc']

        if image.shape[self.axis] != 1:
            image = np.expand_dims(image, self.axis)
        if self.apply_to_label:
            mask = np.expand_dims(mask, self.axis)

        return {'image': image, 'mask': mask, "misc": misc}
