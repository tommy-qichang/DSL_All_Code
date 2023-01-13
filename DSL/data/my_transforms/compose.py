class Compose(object):
    """ Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    """

    def __init__(self, transforms):
        self.transforms = transforms
        self.type = "my_transform"
        self.version = "v0.1"

    def __call__(self, imgs):
        for t in self.transforms:
            imgs = t(imgs)
        return imgs
