import torch
import torchvision.transforms.functional as F
from data_loader.my_transforms.normalize3d import normalize_common


class NormalizeLabel3d(object):
    """Normalize a label rather than the image volume given the mean and standard deviation.
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
        if isinstance(self.mean, int) or isinstance(self.mean, float):
            self.mean = tuple([self.mean for _ in range(0, image.shape[0])])
            self.std = tuple([self.std for _ in range(0, image.shape[0])])

        mask = normalize_common(mask, self.mean, self.std)
        return {"image": image, "mask": mask, "misc": misc}
