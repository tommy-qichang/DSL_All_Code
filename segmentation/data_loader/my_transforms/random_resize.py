import numpy as np
from skimage.transform import resize


class RandomResize:
    def __init__(self, scale, interp="bilinear", training=True):
        """
        Resize the image: C x H x W to the C x load_size[0] x load_size[1]
        order will apply to image and mask separately. 0: nearest-neighbor, 1: bi-linear(default), 2:bi-quadratic,
        3: bi-cubic, 4: bi-quartic  Please refer to: https://scikit-image.org/docs/dev/api/skimage.transform.html
        and https://scikit-image.org/docs/dev/api/skimage.transform.html#skimage.transform.warp
        :param load_size: int or tuple
        :interp interpolation method apply to image: str: "nearest","bilinear"
        """
        # TODO: future should support 3D, 4D and w/o mask(classification task).
        if isinstance(scale, list) and len(scale) == 2:
            self.scale = scale
        else:
            raise NotImplementedError("The scale should be a list of 2 elements")
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
        h, w = image.shape[-2:]

        if len(image.shape) == 3:
            image = np.moveaxis(image, 0, -1)
        if len(mask.shape) == 3:
            mask = np.moveaxis(mask, 0, -1)

        scale = np.random.uniform(self.scale[0], self.scale[1])

        new_h = int(h * scale)
        new_w = int(w * scale)

        new_image = resize(image, (new_h, new_w), order=self.interp, preserve_range=True)
        new_mask = resize(mask, (new_h, new_w), order=0, preserve_range=True)

        if 'labels_ternary' in misc:
            if len(misc['labels_ternary'].shape) == 3:
                misc['labels_ternary'] = np.moveaxis(misc['labels_ternary'], 0, -1)
            misc['labels_ternary'] = resize(misc['labels_ternary'], (new_h, new_w), order=self.interp, preserve_range=True).astype(np.uint8)
            if len(misc['labels_ternary'].shape) == 3:
                misc['labels_ternary'] = np.moveaxis(misc['labels_ternary'], -1, 0)
        if 'weight_maps' in misc:
            if len(misc['weight_maps'].shape) == 3:
                misc['weight_maps'] = np.moveaxis(misc['weight_maps'], 0, -1)
            misc['weight_maps'] = resize(misc['weight_maps'], (new_h, new_w), order=self.interp, preserve_range=True).astype(np.uint8)
            if len(misc['weight_maps'].shape) == 3:
                misc['weight_maps'] = np.moveaxis(misc['weight_maps'], -1, 0)

        if len(image.shape) == 3:
            new_image = np.moveaxis(new_image, -1, 0) #.reshape((c, new_h, new_w))
        if len(mask.shape) == 3:
            new_mask = np.moveaxis(new_mask, -1, 0) #.reshape((self.load_size[0], self.load_size[1]))

        return {'image': new_image, 'mask': new_mask, 'misc': misc}
