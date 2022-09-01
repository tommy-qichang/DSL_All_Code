import numpy as np


class RandomRotate3d:
    """
    Crop the 3D numpy [c, h, w, d] and mask [h, w, d] in a sample
    """

    def __init__(self, angle, training=True):
        assert isinstance(angle, (float, list, tuple))
        if isinstance(angle, float):
            self.angle = (angle, angle, angle)
        elif isinstance(angle, tuple):
            assert len(angle) == 3, "The list output size should include three dimensions"
            self.angle = angle
        else:
            assert len(angle) == 3, "The list output size should include three dimensions"
            self.angle = tuple(angle)

        self.training = training

    def __call__(self, sample):
        image, mask, misc = sample['image'], sample['mask'], sample['misc']

        c, h, w, d = image.shape
        new_h, new_w, new_d = self.output_size

        if not self.training:
            # Test stage, will set the crop position at 0,0
            top = 0
            left = 0
            depth = 0
        else:
            top = np.random.randint(h - new_h + 1)
            left = np.random.randint(w - new_w + 1)
            depth = np.random.randint(d - new_d + 1)


        if len(mask.shape) == 3:
            mask = mask[top: top + new_h, left: left + new_w, depth: depth + new_d]

        if self.padding != (0, 0, 0):
            mask = mask[self.padding[0]:(mask.shape[0] - self.padding[0]),
                   self.padding[1]:(mask.shape[1] - self.padding[1]),
                   self.padding[2]:(mask.shape[2] - self.padding[2])]

        return {'image': image, 'mask': mask, 'misc': misc}
