import torchvision.transforms.functional as F


class NormalizeInstanceChannel(object):
    """Normalize a tensor volume with mean and standard deviation estimated
    from the sample itself.
    :param mean: mean value.
    :param std: standard deviation value.
    """
    def __init__(self, training=True):
        self.training = training

    def __call__(self, sample):
        image, mask, misc = sample['image'], sample['mask'], sample['misc']
        means = [image[i].mean() for i in range(image.shape[0])]
        stds = [image[i].std() for i in range(image.shape[0])]

        if means != [] or stds != []:
            image = F.normalize(image, means, stds)
        return {"image":image, "mask":mask, "misc": misc}
