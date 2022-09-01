import torchvision.transforms.functional as F


class Normalize(object):
    """Normalize a tensor volume given the mean and standard deviation.
    :param mean: mean value.
    :param std: standard deviation value.
    """

    def __init__(self, mean, std, val_mean=None, val_std=None, training=True):
        if isinstance(mean, list):
            self.mean = tuple(mean)
            self.std = tuple(std)
            if val_mean is not None:
                self.val_mean = tuple(val_mean)
                self.val_std = tuple(val_std)
            else:
                self.val_mean = self.mean
                self.val_std = self.std
        else:
            self.mean = mean
            self.std = std
            if val_mean is not None:
                self.val_mean = val_mean
                self.val_std = val_std
            else:
                self.val_mean = self.mean
                self.val_std = self.std

        print(f"mean:{self.mean},std:{self.std},val_mean:{self.val_mean},val_std:{self.val_std}")
        self.training = training

    def __call__(self, sample):
        image, mask, misc = sample['image'], sample['mask'], sample['misc']

        if "train" in misc['img_path'] :
            mean = self.mean
            std = self.std
        else:
            mean = self.val_mean
            std = self.val_std

        if mean == 0 or std == 0:
            return {"image": image, "mask": mask}
        if isinstance(mean, tuple):
            image = F.normalize(image, mean, std)
        elif isinstance(mean, int) or isinstance(mean, float):
            image = F.normalize(image, [mean for _ in range(0, image.shape[0])],
                                [std for _ in range(0, image.shape[0])])
        return {"image": image, "mask": mask, "misc": misc}
