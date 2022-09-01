import numpy as np


class RandomCrop:
    """
    Crop the 2D numpy [c, h, w] and mask [h, w] in a sample
    """

    def __init__(self, output_size, padding=0, training=True):
        assert isinstance(output_size, (int, list))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = tuple(output_size)

        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            assert len(padding) == 2
            self.padding = tuple(padding)

        self.training = training

    def __call__(self, sample):
        image, mask, misc = sample['image'], sample['mask'], sample['misc']

        c, h, w = image.shape
        new_h, new_w = self.output_size

        assert (h - new_h) >= 0 and (w - new_w) >= 0, f"The orig size:({h},{w}) should be equal or larger than " \
                                                      f"crop size:({new_h},{new_w})"

        if not self.training:
            # Test stage, will set the crop position at 0,0
            top = 0
            left = 0
        else:
            top = np.random.randint(h - new_h + 1)
            left = np.random.randint(w - new_w + 1)

        image = image[:, top: top + new_h, left: left + new_w]
        if len(mask.shape) == 2:
            mask = mask[top: top + new_h, left: left + new_w]
        elif len(mask.shape) == 3:
            mask = mask[:, top: top + new_h, left: left + new_w]

        if self.padding != (0, 0):
            mask = mask[self.padding[0]:(mask.shape[0] - self.padding[0]),
                   self.padding[1]:(mask.shape[1] - self.padding[1])]

        return {'image': image, 'mask': mask, 'misc': misc}
