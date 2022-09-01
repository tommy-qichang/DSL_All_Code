import torchvision.transforms.functional as F


class NormalizeBoth(object):
    """Normalize both image and label (with channel dim) tensors given the mean and standard deviation.
    :param mean: mean value.
    :param std: standard deviation value.
    """

    def __init__(self, mean, std, training=True):
        if isinstance(mean, list):
            self.mean = tuple(mean)
            self.std = tuple(std)
        else:
            self.mean = mean
            self.std = std

        self.training = training

    def __call__(self, sample):
        image, mask, misc = sample['image'], sample['mask'], sample['misc']
        if self.std == 0:
            return {"image": image, "mask": mask}
        if isinstance(self.mean, tuple):
            image = F.normalize(image, self.mean, self.std)
            mask = F.normalize(mask, self.mean, self.std)
        elif isinstance(self.mean, int) or isinstance(self.mean, float):
            image = F.normalize(image, [self.mean for _ in range(0, image.shape[0])],
                                [self.std for _ in range(0, image.shape[0])])
            mask = F.normalize(mask, [self.mean for _ in range(0, mask.shape[0])],
                                [self.std for _ in range(0, mask.shape[0])])
        return {"image": image, "mask": mask, "misc": misc}
