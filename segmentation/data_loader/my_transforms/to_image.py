import torch
import PIL.Image
import numpy as np

from utils.util import convert_to_uint8


class ToImage:
    def __init__(self, training=True):
        """
        Convert numpy array to PIL Image, and convert back.
        """
        self.training = training

    def __call__(self, sample):
        image, mask, misc = sample['image'], sample['mask'], sample['misc']

        if len(image.shape) == 3:
            stack_img = []
            #should normalize each channel
            for ch in range(image.shape[0]):
                stack_img.append(convert_to_uint8(image[ch], [image[ch].min(),image[ch].max()]))
            stack_img = np.stack(stack_img,axis=0)
            assert stack_img.shape == image.shape
            image = stack_img
        else:
            image = convert_to_uint8(image, [image.min(), image.max()])

        return {'image': image, 'mask': mask, 'misc': misc}
