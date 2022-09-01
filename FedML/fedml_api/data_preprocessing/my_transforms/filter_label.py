"""
Only use one specific label for 1 class segmentation.
"""
import numpy as np


class FilterLabel:
    def __init__(self, labels=[1], as_value=[1]):
        """
        Select limited labels id, and reset the id as as_value.
        """
        self.labels = labels
        self.as_value = as_value

    def __call__(self, sample):
        image, mask, misc = sample['image'], sample['mask'], sample['misc']

        new_mask = np.zeros_like(mask)
        for id in range(len(self.labels)):
            new_mask[mask == self.labels[id]] = self.as_value[id]

        return {'image': image, 'mask': new_mask, 'misc': misc}
