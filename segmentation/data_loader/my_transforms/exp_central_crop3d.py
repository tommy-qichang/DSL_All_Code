import numpy as np
import random


class ExpCentralCrop3d:
    """
    Crop the 3D numpy [c, h, w, d] and mask [d, 2] in a sample
    The d should be randomly sampled
    """

    def __init__(self, output_size, padding=0, include_lax_tensor=False, training=True):
        assert isinstance(output_size, (int, list, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size, output_size)
        elif isinstance(output_size, tuple):
            assert len(output_size) == 3, "The list output size should include three dimensions"
            self.output_size = output_size
        else:
            assert len(output_size) == 3, "The list output size should include three dimensions"
            self.output_size = tuple(output_size)

        if isinstance(padding, int):
            self.padding = (padding, padding, padding)
        else:
            assert len(padding) == 3, "The padding size should include three dimensions"
            self.padding = tuple(padding)

        self.training = training
        self.include_lax_tensor = include_lax_tensor

    def __call__(self, sample):
        image, mask, misc = sample['image'], sample['mask'], sample['misc']

        c, h, w, d = image.shape
        new_h, new_w, new_d = self.output_size

        assert (h - new_h) >= 0 and (w - new_w) >= 0 and (d - new_d) >= 0, \
            f"The orig size:({h},{w},{d}) should be equal or larger than crop size:({new_h},{new_w},{new_d})"

        top = (h - new_h) // 2
        left = (w - new_w) // 2
        depth = (d - new_d) // 2
        # depth = random.randint(0, (d - new_d) // 2)

        image = image[:, top: top + new_h, left: left + new_w, depth: depth + new_d]
        if len(mask.shape) == 2:
            if self.include_lax_tensor:
                sax_mask = mask[depth: depth + new_d, :]
                lax_mask = mask[-2:, :]
                mask = np.concatenate((sax_mask,lax_mask))
            else:
                mask = mask[depth: depth + new_d, :]
        elif len(mask.shape) ==4:
            mask = mask[:, top: top + new_h, left: left + new_w, depth: depth + new_d]

        if self.padding != (0, 0, 0):
            mask = mask[self.padding[0]:(mask.shape[0] - self.padding[0]),
                   self.padding[1]:(mask.shape[1] - self.padding[1]),
                   self.padding[2]:(mask.shape[2] - self.padding[2])]

        return {'image': image, 'mask': mask, 'misc': misc}
