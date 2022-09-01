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

        if random.random() < 0.5:
            image = np.moveaxis(image, 1, 2).copy()
            if len(mask.shape) == 2:
                mask = np.moveaxis(mask, 0, 1).copy()
        if "train" in misc['img_path'] and self.tilt and random.random() < 0.5:
            #rotate image and mask +=10deg
            deg = random.randint(-10,10)
            image = ndimage.rotate(image, deg, reshape=False)
            if len(mask.shape) == 2:
                mask = ndimage.rotate(mask, deg, reshape=False,order=0)

        return {'image': image, 'mask': mask, 'misc': misc}
