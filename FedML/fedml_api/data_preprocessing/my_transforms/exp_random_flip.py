import random


class ExpRandomFlip:
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
        image = sample['mask']

        if self.horizontal and random.random() < 0.5:
            image = image[:, ::-1, :].copy()
        if self.vertical and random.random() < 0.5:
            image = image[::-1, :, :].copy()

        return {'image': image, 'mask': image, 'misc': sample['misc']}
