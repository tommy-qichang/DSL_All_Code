import numpy as np
from .common_func import cropImage2DAt


class CropPadding:
    def __init__(self, load_size, training=True):
        """
        Padding the image: C x H x W, if H or W less than load_size, and will keep the overflow side.

        """
        # TODO: Not just support static filling of the padding, but also mirror, etc...
        if isinstance(load_size, list) and len(load_size) == 2:
            self.load_size = tuple(load_size)
        elif isinstance(load_size, int):
            self.load_size = (load_size, load_size)
        else:
            raise NotImplementedError("The load_size should be integer or tuple")

        self.training = training

    def __call__(self, sample):
        image, mask, misc = sample['image'], sample['mask'], sample['misc']
        if len(image.shape) == 2:
            image = np.expand_dims(image, 0)
        c, h, w = image.shape

        size = (self.load_size[0], self.load_size[1])
        center = (h//2, w//2)

        new_image = cropImage2DAt(image, center, size)
        mask = cropImage2DAt(mask, center, size)

        return {'image': new_image, 'mask': mask, 'misc': misc}

