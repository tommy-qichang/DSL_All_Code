class Homochrome:
    def __init__(self, axis=-1):
        """
        Just select one channel for labels. For instance, from (X, Y, Z, 3) to (X, Y, Z).

        """
        self.axis = axis

    def __call__(self, sample):
        image, mask, misc = sample['image'], sample['mask'], sample['misc']

        if mask.shape[self.axis] == 3:
            mask = mask[:, :, 0]

        return {'image': image, 'mask': mask, 'misc': misc}
