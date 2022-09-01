import numpy as np
from skimage.transform import resize


class MoveAxis:
    def __init__(self,from_axis, to_axis, move_label=False, training=True):
        """
        Resize the image: C x H x W to the C x load_size[0] x load_size[1]
        order will apply to image and mask separately. 0: nearest-neighbor, 1: bi-linear(default), 2:bi-quadratic,
        3: bi-cubic, 4: bi-quartic  Please refer to: https://scikit-image.org/docs/dev/api/skimage.transform.html
        and https://scikit-image.org/docs/dev/api/skimage.transform.html#skimage.transform.warp
        :param load_size: int or tuple
        :interp interpolation method apply to image: str: "nearest","bilinear"
        """
        self.from_axis = from_axis
        self.to_axis = to_axis
        self.move_label = move_label
        self.training = training

    def __call__(self, sample):
        image, mask, misc = sample['image'], sample['mask'], sample['misc']

        new_image = np.moveaxis(image, self.from_axis, self.to_axis)
        if self.move_label:
            mask = np.moveaxis(mask, self.from_axis, self.to_axis)


        return {'image': new_image, 'mask': mask, 'misc': misc}
