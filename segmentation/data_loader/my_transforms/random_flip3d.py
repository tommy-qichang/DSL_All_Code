import numpy as np
import random


class RandomFlip3d:
    def __init__(self, horizontal=True, vertical=True, deep=False, training=True):
        self.horizontal = horizontal
        self.vertical = vertical
        self.training = training
        self.deep = deep

    def __call__(self, sample):
        """
        Randomly flip the numpy image horizontal or/and vertical
        Args:
            sample: {'image':..., 'mask':...}
            image size: [c, h, w, z]
            mask size: [h, w, z]
        Returns:
            Randomly flipped image.
        """
        image, mask, misc = sample['image'], sample['mask'], sample['misc']
        if not self.training:
            # If testing, will not flip.
            return {'image': image, 'mask': mask, 'misc': misc}

        if len(image.shape) == 3:
            image = np.expand_dims(image, 0)

        if self.vertical and random.random() < 0.5:
            image = image[:, ::-1, :, :].copy()
            if len(mask.shape) == 3:
                mask = mask[::-1, :, :].copy()
            elif len(mask.shape) == 4:
                mask = mask[:, ::-1, :, :].copy()
        if self.horizontal and random.random() < 0.5:
            image = image[:, :, ::-1, :].copy()
            if len(mask.shape) == 3:
                mask = mask[:, ::-1, :].copy()
            elif len(mask.shape) == 4:
                mask = mask[:, :, ::-1, :].copy()
        if self.deep and random.random() <0.5:
            image = image[:, :, :, ::-1].copy()
            if len(mask.shape) == 3:
                mask = mask[:, :, ::-1].copy()
            elif len(mask.shape) == 4:
                mask = mask[:, :, :, ::-1].copy()

        return {'image': image, 'mask': mask, 'misc': misc}
