import numpy as np
from skimage.transform import resize


class Resize:
    def __init__(self, load_size, interp="bilinear", training=True):
        """
        Resize the image: C x H x W to the C x load_size[0] x load_size[1]
        order will apply to image and mask separately. 0: nearest-neighbor, 1: bi-linear(default), 2:bi-quadratic,
        3: bi-cubic, 4: bi-quartic  Please refer to: https://scikit-image.org/docs/dev/api/skimage.transform.html
        and https://scikit-image.org/docs/dev/api/skimage.transform.html#skimage.transform.warp
        :param load_size: int or tuple
        :interp interpolation method apply to image: str: "nearest","bilinear"
        """
        # TODO: future should support 3D, 4D and w/o mask(classification task).
        if isinstance(load_size, tuple) and len(load_size) == 2:
            self.load_size = load_size
        elif isinstance(load_size, int):
            self.load_size = (load_size, load_size)
        else:
            raise NotImplementedError("The load_size should be integer or tuple")
        if interp == "nearest":
            self.interp = 0
        elif interp == "bilinear":
            self.interp = 1
        else:
            raise NotImplementedError("Right now only support image interpolation method: nearest and linear")
        self.training = training

    def __call__(self, sample):
        image, mask, misc = sample['image'], sample['mask'], sample['misc']

        # Need to combine batch and channel number and move to the end of the matrix as channel.
        c, h, w = image.shape
        # image = image.reshape(-1, h, w)
        # mask = mask.reshape(-1, h, w)
        image = np.moveaxis(image, 0, -1)
        if len(mask.shape) == 3:
            mask = np.moveaxis(mask, 0, -1)

        new_image = resize(image, self.load_size, order=self.interp, preserve_range=True)
        new_mask = resize(mask, self.load_size, order=0, preserve_range=True)

        new_image = np.moveaxis(new_image, -1, 0)  #.reshape((c, self.load_size[0], self.load_size[1]))
        if len(mask.shape) == 3:
            new_mask = np.moveaxis(new_mask, -1, 0)  #.reshape((self.load_size[0], self.load_size[1]))

        return {'image': new_image, 'mask': new_mask, 'misc': misc}
