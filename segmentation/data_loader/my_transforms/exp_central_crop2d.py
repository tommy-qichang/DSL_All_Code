import numpy as np
import random


class ExpCentralCrop2d:
    """
    Crop the 2D numpy [c, h, w] and mask [h, w] in a sample
    The d should be randomly sampled
    """

    def __init__(self, output_size, padding=0, include_lax_tensor=False, training=True):
        assert isinstance(output_size, (int, list, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        elif isinstance(output_size, tuple):
            assert len(output_size) == 2, "The list output size should include two dimensions"
            self.output_size = output_size
        else:
            assert len(output_size) == 2, "The list output size should include two dimensions"
            self.output_size = tuple(output_size)

        if isinstance(padding, int):
            self.padding = (padding, padding, padding)
        else:
            assert len(padding) == 2, "The padding size should include two dimensions"
            self.padding = tuple(padding)

        self.training = training

    def __call__(self, sample):
        image, mask, misc = sample['image'], sample['mask'], sample['misc']

        c, h, w= image.shape
        new_h, new_w = self.output_size

        assert (h - new_h) >= 0 and (w - new_w) >= 0, \
            f"The orig size:({h},{w}) should be equal or larger than crop size:({new_h},{new_w})"

        top = (h - new_h) // 2
        left = (w - new_w) // 2
        # depth = random.randint(0, (d - new_d) // 2)

        image = image[:, top: top + new_h, left: left + new_w]
        if len(mask.shape) == 2:
            mask = mask[top: top + new_h, left: left + new_w]

        misc["central_crop2d"] = [top, top + new_h, left, left + new_w]
        if self.padding != (0, 0, 0):
            mask = mask[self.padding[0]:(mask.shape[0] - self.padding[0]),
                   self.padding[1]:(mask.shape[1] - self.padding[1]),
                   self.padding[2]:(mask.shape[2] - self.padding[2])]

        return {'image': image, 'mask': mask, 'misc': misc}
