import numpy as np


class ExpPadding3d:
    def __init__(self, load_size, training=True):
        """
        Padding the image: C x H x W x D, if H, W, D less than load_size, and will keep the overflow side as it is.

        """
        # TODO: Not just support static filling of the padding, but also mirror, etc...
        if isinstance(load_size, list) and len(load_size) == 3:
            self.load_size = tuple(load_size)
        elif isinstance(load_size, int):
            self.load_size = (load_size, load_size, load_size)
        else:
            raise NotImplementedError("The load_size should be integer or tuple with size 3.")

        self.training = training

    def __call__(self, sample):
        image, mask, misc = sample['image'], sample['mask'], sample['misc']
        c, h, w, d = image.shape
        new_h = max(h, self.load_size[0])
        new_w = max(w, self.load_size[1])
        new_d = max(d, self.load_size[2])

        pad_h = ((new_h - h) // 2, (new_h - h) - (new_h - h) // 2) if h < self.load_size[0] else (0, 0)
        pad_w = ((new_w - w) // 2, (new_w - w) - (new_w - w) // 2) if w < self.load_size[1] else (0, 0)
        pad_d = ((new_d - d) // 2, (new_d - d) - (new_d - d) // 2) if d < self.load_size[2] else (0, 0)

        new_image = np.pad(image, ((0, 0), pad_h, pad_w, pad_d), mode="edge")
        # misc['orig_dt'] = np.pad(misc['orig_dt'], (pad_h, pad_w, pad_d))
        if len(mask.shape) == 2:
            # segmentation
            mask = np.pad(mask, (pad_d,(0,0)))
        return {'image': new_image, 'mask': mask, 'misc': misc}
