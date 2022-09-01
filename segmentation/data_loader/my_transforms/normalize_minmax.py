import torchvision.transforms.functional as F


class NormalizeMinmax(object):
    """Normalize a tensor volume given min and max value.
    :param min
    :param max
    """

    def __init__(self, min, max, training=True):
        self.min = min
        self.max = max

        self.training = training

    def __call__(self, sample):
        image, mask, misc = sample['image'], sample['mask'], sample['misc']

        image = ((image - image.min())/(image.max() - image.min()))
        image = (image * (self.max - self.min)) + self.min

        return {"image": image, "mask": mask, "misc": misc}
