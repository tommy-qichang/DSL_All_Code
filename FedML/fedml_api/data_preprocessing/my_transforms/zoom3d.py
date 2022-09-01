import numpy as np
from scipy.ndimage import zoom
from skimage.transform import resize


class Zoom3d:
    def __init__(self, zoom_size, interp="bilinear", training=True):
        """
        Resize the image: C x H x W x D to the C x load_size[0] x load_size[1]
        order will apply to image and mask separately. 0: nearest-neighbor, 1: bi-linear(default), 2:bi-quadratic,
        3: bi-cubic, 4: bi-quartic  Please refer to: https://scikit-image.org/docs/dev/api/skimage.transform.html
        and https://scikit-image.org/docs/dev/api/skimage.transform.html#skimage.transform.warp
        :param load_size: int or tuple
        :interp interpolation method apply to image: str: "nearest","bilinear"
        """
        # TODO: future should support 3D, 4D and w/o mask(classification task).
        if (isinstance(zoom_size, tuple) or isinstance(zoom_size, list)) and len(zoom_size) == 3:
            self.zoom_size = tuple(zoom_size)
        elif isinstance(zoom_size, int) or isinstance(zoom_size, float):
            self.zoom_size = (zoom_size, zoom_size, zoom_size)
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

        new_image = zoom(image, self.zoom_size, order=self.interp)
        new_mask = zoom(mask, self.zoom_size, order=0)

        return {'image': new_image, 'mask': new_mask, 'misc': misc}
