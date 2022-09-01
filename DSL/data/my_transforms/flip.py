import numpy as np
import random


class Flip:
    def __init__(self, horizontal=True, vertical=True, training=True):
        self.horizontal = horizontal
        self.vertical = vertical
        self.training = training

    def __call__(self, sample):
        """
        Randomly flip the numpy image horizontal or/and vertical
        Args:
            sample: {'image':..., 'mask':...}
            image size: [c, h, w]
            mask size: [h, w]
        Returns:
            Randomly flipped image.
        """
        image, mask, misc = sample['image'], sample['mask'], sample['misc']
        if not self.training:
            # If testing, will not flip.
            return {'image': image, 'mask': mask}

        if len(image.shape) == 2:
            image = np.expand_dims(image, 0)

        if self.horizontal:
            image = image[:, :, ::-1].copy()
            if mask is not None and len(mask.shape) == 2:
                mask = mask[:, ::-1].copy()
            elif mask is not None and len(mask.shape) == 3:
                mask = mask[:, , ::-1].copy()

        if self.vertical:
            image = image[:, ::-1, :].copy()
            if mask is not None and len(mask.shape) == 2:
                mask = mask[::-1, :].copy()
            elif mask is not None and len(mask.shape) == 3:
                mask = mask[:, :-1, :].copy()

        return {'image': image, 'mask': mask, 'misc': misc}
