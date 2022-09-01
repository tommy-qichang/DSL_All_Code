import numpy as np
import random
from .common_func import flip


class RandomFlip:
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
        if self.training:
            if "RandomFlip" in sample['misc']:  # preset parameters
                flag_random_flip = list(map(bool, sample['misc']["RandomFlip"]))
            else:
                flag_random_flip = [False, False]  # vertical, horizontal

                if self.vertical and random.random() < 0.5:
                    flag_random_flip[0] = True
                if self.horizontal and random.random() < 0.5:
                    flag_random_flip[1] = True

        if 'image' in sample:
            image, mask, misc = sample['image'], sample['mask'], sample['misc']
            if not self.training:
                # If testing, will not flip.
                return {'image': image, 'mask': mask, 'misc': misc}

            if len(image.shape) == 2:
                image = np.expand_dims(image, 0)

            misc['RandomFlip'] = flag_random_flip

            image = flip(image, misc['RandomFlip'][0], misc['RandomFlip'][1])
            mask = flip(mask, misc['RandomFlip'][0], misc['RandomFlip'][1])

            if 'labels_ternary' in misc:
                misc['labels_ternary'] = flip(misc['labels_ternary'], misc['RandomFlip'][0], misc['RandomFlip'][1])
            if 'weight_maps' in misc:
                misc['weight_maps'] = flip(misc['weight_maps'], misc['RandomFlip'][0], misc['RandomFlip'][1])

            return {'image': image, 'mask': mask, 'misc': misc}
        else:
            mask, misc = sample['mask'], sample['misc']
            if not self.training:
                # If testing, will not flip.
                return {'mask': mask, 'misc': misc}

            misc['RandomFlip'] = flag_random_flip

            mask = flip(mask, misc['RandomFlip'][0], misc['RandomFlip'][1])

            if 'labels_ternary' in misc:
                misc['labels_ternary'] = flip(misc['labels_ternary'], misc['RandomFlip'][0], misc['RandomFlip'][1])
            if 'weight_maps' in misc:
                misc['weight_maps'] = flip(misc['weight_maps'], misc['RandomFlip'][0], misc['RandomFlip'][1])

            return {'mask': mask, 'misc': misc}
