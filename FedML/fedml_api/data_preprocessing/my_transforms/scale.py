import numpy as np
from skimage.transform import resize


class Scale:
    def __init__(self, load_size: int, interp="bilinear", training=True):
        """
        Scale the image:
        Input format: image: C x H x W  mask: H x W
        The output size will be matched smaller edge of the image.
        If height > width, then the image will be rescaled to (size*height/width, size)
        order will apply to image and mask separately. 0: nearest-neighbor, 1: bi-linear(default), 2:bi-quadratic,
        3: bi-cubic, 4: bi-quartic  Please refer to: https://scikit-image.org/docs/dev/api/skimage.transform.html
        and https://scikit-image.org/docs/dev/api/skimage.transform.html#skimage.transform.warp
        :param load_size: int or tuple
        :interp interpolation method apply to image: str: "nearest","bilinear"
        """
        # TODO: future should support 3D, 4D and w/o mask(classification task).
        if isinstance(load_size, int):
            self.load_size = load_size
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

        if h > w:
            new_h = (self.load_size * h) // w
            new_w = self.load_size
        else:
            new_h = self.load_size
            new_w = (self.load_size * w) // h

        new_image = resize(image, (new_h, new_w), order=self.interp, preserve_range=True)
        new_mask = resize(mask, (new_h, new_w), order=0, preserve_range=True)

        new_image = np.moveaxis(new_image, -1, 0).reshape((c, new_h, new_w))
        # new_mask = np.moveaxis(new_mask, -1, 0).reshape((new_h, new_w))

        return {'image': new_image, 'mask': new_mask, 'misc': misc}
