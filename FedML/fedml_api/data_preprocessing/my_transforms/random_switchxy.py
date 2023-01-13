import numpy as np
import random


class RandomSwitchxy:
    def __init__(self, training=True):
        self.training = training

    def __call__(self, sample):
        """
        Randomly switch X-Y axis of the numpy image
        Args:
            sample: {'image':..., 'mask':...}
            image size: [c, h, w]
            mask size: [h, w]
        Returns:
            Randomly switch-axis image.
        """
        image, mask, misc = sample['image'], sample['mask'], sample['misc']
        if not self.training:
            # If testing, will not flip.
            return {'image': image, 'mask': mask}

        if len(image.shape) == 2:
            image = np.expand_dims(image, 0)

        if random.random() < 0.5:
            image = np.moveaxis(image, -1, -2).copy()
            mask = np.moveaxis(mask, -1, -2).copy()

            if 'labels_ternary' in misc:
                misc['labels_ternary'] = np.moveaxis(misc['labels_ternary'], -1, -2).copy()
            if 'weight_maps' in misc:
                misc['weight_maps'] = np.moveaxis(misc['weight_maps'], -1, -2).copy()

        return {'image': image, 'mask': mask, 'misc': misc}
