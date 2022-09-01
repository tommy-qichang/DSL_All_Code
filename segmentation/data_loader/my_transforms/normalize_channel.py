import torchvision.transforms.functional as F


class NormalizeChannel(object):
    """Normalize a tensor volume with mean and standard deviation estimated
    from the sample itself.
    :param mean: mean value.
    :param std: standard deviation value.
    """
    def __init__(self, training=True):
        self.training = training

    def __call__(self, sample):
        image, mask, misc = sample['image'], sample['mask'], sample['misc']

        assert len(image.shape) == 3, "The channel wised normalization should have at 3 channels."
        mean, std = image.mean(axis=(1,2)), image.std(axis=(1,2))

        image = F.normalize(image, mean,std)

        return {"image":image, "mask":mask, "misc": misc}
