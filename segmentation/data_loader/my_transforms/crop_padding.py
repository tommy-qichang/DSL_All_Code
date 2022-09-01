import numpy as np


class CropPadding:
    def __init__(self, load_size, training=True):
        """
        Padding the image: C x H x W, if H or W less than load_size, and will keep the overflow side.

        """
        # TODO: Not just support static filling of the padding, but also mirror, etc...
        if isinstance(load_size, list) and len(load_size) == 2:
            self.load_size = tuple(load_size)
        elif isinstance(load_size, int):
            self.load_size = (load_size, load_size)
        else:
            raise NotImplementedError("The load_size should be integer or tuple")

        self.training = training

    def __call__(self, sample):
        image, mask, misc = sample['image'], sample['mask'], sample['misc']
        if len(image.shape) == 2:
            image = np.expand_dims(image, 0)
        c, h, w = image.shape

        size = (c, self.load_size[0], self.load_size[1])

        center = (c//2, h//2, w//2)
        new_image = cropImageAt(image, center, size, c_val=image.min())
        if len(mask.shape) == 2:
            # segmentation
            mask = np.expand_dims(mask, 0)
            mask = cropImageAt(mask, (0, h//2, w//2), (1, self.load_size[0], self.load_size[1]), c_val=mask.min())
            mask = mask[0]
        else:
            c, h, w = mask.shape
            mask = cropImageAt(mask, (c//2, h // 2, w // 2), (c, self.load_size[0], self.load_size[1]), c_val=mask.min())
        return {'image': new_image, 'mask': mask, 'misc': misc}


def cropImageAt(img, center, size, c_val=0):
    bbox = [[0, 0], [0, 0], [0, 0]]
    bbox[0][0] = int(center[0] - size[0] // 2)
    bbox[0][1] = bbox[0][0] + size[0]
    bbox[1][0] = int(center[1] - size[1] // 2)
    bbox[1][1] = bbox[1][0] + size[1]
    bbox[2][0] = int(center[2] - size[2] // 2)
    bbox[2][1] = bbox[2][0] + size[2]
    shape_old = img.shape
    pad = [[0, 0], [0, 0], [0, 0]]
    for i in range(3):
        if bbox[i][0] < 0:
            pad[i][0] = -bbox[i][0]
            bbox[i][0] = 0

        if bbox[i][1] > shape_old[i]:
            pad[i][1] = bbox[i][1] - shape_old[i]
    img_new = np.pad(img, pad, 'constant', constant_values=c_val)
    bbox[0][1] = bbox[0][0] + size[0]
    bbox[1][1] = bbox[1][0] + size[1]
    bbox[2][1] = bbox[2][0] + size[2]
    return img_new[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1]]