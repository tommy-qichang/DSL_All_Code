import numpy as np
import random
from scipy import ndimage, misc

class RandomRotate:
    def __init__(self, tilt=False, training=True):
        self.training = training
        self.tilt = tilt

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

        # if random.random() < 0.5:
        #     image = np.moveaxis(image, -1, -2).copy()
        #     mask = np.moveaxis(mask, -1, -2).copy()
        #
        #     if 'labels_ternary' in misc:
        #         misc['labels_ternary'] = np.moveaxis(misc['labels_ternary'], -1, -2).copy()
        #     if 'weight_maps' in misc:
        #         misc['weight_maps'] = np.moveaxis(misc['weight_maps'], -1, -2).copy()
        if self.tilt > 0 and random.random() < 0.5:
            # rotate image and mask += self.tilt degree
            deg = random.randint(-self.tilt, self.tilt)
            image = ndimage.rotate(image, deg, axes=(-1, -2), reshape=False)
            mask = ndimage.rotate(mask, deg, axes=(-1, -2), reshape=False, order=0)
            if 'labels_ternary' in misc:
                misc['labels_ternary'] = ndimage.rotate(misc['labels_ternary'], deg, axes=(-1, -2), reshape=False)
            if 'weight_maps' in misc:
                misc['weight_maps'] = ndimage.rotate(misc['weight_maps'], deg, axes=(-1, -2), reshape=False)

        return {'image': image, 'mask': mask, 'misc': misc}
