import numpy as np


class Squeeze:
    def __init__(self, axis=0, apply_to_label=False):
        """
        Squeeze any image if axis dimension is not 1. For instance, from (1, X, Y, Z) to (X, Y, Z).

        """
        self.axis = axis
        self.apply_to_label = apply_to_label

    def __call__(self, sample):
        image, mask, misc = sample['image'], sample['mask'], sample['misc']

        if image.shape[self.axis] == 1:
            image = np.squeeze(image, self.axis)
        if self.apply_to_label and mask.shape[self.axis] ==1:
            mask = np.squeeze(mask, self.axis)


        return {'image': image, 'mask': mask, "misc": misc}
