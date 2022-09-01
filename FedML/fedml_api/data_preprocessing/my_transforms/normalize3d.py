import torch
import torchvision.transforms.functional as F


class Normalize3d(object):
    """Normalize a tensor volume given the mean and standard deviation.
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

        image = normalize_common(image, self.mean, self.std)

        # flat_img = image.clone().flatten(1)
        #
        # dtype = flat_img.dtype
        # mean = torch.as_tensor(self.mean, dtype=dtype, device=flat_img.device)
        # std = torch.as_tensor(self.std, dtype=dtype, device=flat_img.device)
        # flat_img.sub_(mean[:, None]).div_(std[:, None])
        # image = flat_img.view(image.shape)
        return {"image": image, "mask": mask, "misc": misc}

def normalize_common(image, mean, std, inplace=False):

    if not inplace:
        image_clone = image.clone()
    image_clone = image_clone.flatten(1)
    dtype = image_clone.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=image_clone.device)
    std = torch.as_tensor(std, dtype=dtype, device=image_clone.device)
    image_clone.sub_(mean[:, None]).div_(std[:, None])
    image_clone = image_clone.view(image.shape)
    return image_clone
