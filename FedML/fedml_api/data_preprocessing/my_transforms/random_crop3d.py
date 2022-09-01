import numpy as np


class RandomCrop3d:
    """
    Crop the 3D numpy [c, h, w, d] and mask [h, w, d] in a sample
    """

    def __init__(self, output_size, padding=0, training=True):
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

    def __call__(self, sample):
        image, mask, misc = sample['image'], sample['mask'], sample['misc']

        c, h, w, d = image.shape
        new_h, new_w, new_d = self.output_size

        assert (h - new_h) >= 0 and (w - new_w) >= 0 and (d - new_d) >= 0, \
            f"The orig size:({h},{w},{d}) should be equal or larger than crop size:({new_h},{new_w})"

        if not self.training:
            # Test stage, will set the crop position at 0,0
            top = 0
            left = 0
            depth = 0
        else:
            top = np.random.randint(h - new_h + 1)
            left = np.random.randint(w - new_w + 1)
            depth = np.random.randint(d - new_d + 1)

        image = image[:, top: top + new_h, left: left + new_w, depth: depth + new_d]
        if len(mask.shape) == 3:
            mask = mask[top: top + new_h, left: left + new_w, depth: depth + new_d]

        if self.padding != (0, 0, 0):
            mask = mask[self.padding[0]:(mask.shape[0] - self.padding[0]),
                   self.padding[1]:(mask.shape[1] - self.padding[1]),
                   self.padding[2]:(mask.shape[2] - self.padding[2])]

        return {'image': image, 'mask': mask, 'misc': misc}
