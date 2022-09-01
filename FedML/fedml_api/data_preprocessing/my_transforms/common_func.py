import numpy as np


def crop(img, top, left, new_h, new_w, padding=(0, 0)):
    if len(img.shape) == 2:
        img = img[top: top + new_h, left: left + new_w]
        if padding != (0, 0):
            img = img[padding[0]:(img.shape[0] - padding[0]),
                      padding[1]:(img.shape[1] - padding[1])].copy()
    else:
        img = img[:, top: top + new_h, left: left + new_w]
        if padding != (0, 0):
            img = img[:, padding[0]:(img.shape[1] - padding[0]),
                      padding[1]:(img.shape[2] - padding[1])].copy()
    return img


def cropImage3DAt(img, center, size, c_val=None):
    if c_val is None:
        c_val = img.min()

    shape_old = img.shape
    bbox = [[0, 0], [0, 0], [0, 0]]
    bbox[0][0] = int(center[0] - size[0] // 2)
    bbox[0][1] = bbox[0][0] + size[0]
    bbox[1][0] = int(center[1] - size[1] // 2)
    bbox[1][1] = bbox[1][0] + size[1]
    bbox[2][0] = int(center[2] - size[2] // 2)
    bbox[2][1] = bbox[2][0] + size[2]

    pad = [[0, 0], [0, 0], [0, 0]]
    for i in range(3):
        if bbox[i][0] < 0:
            pad[i][0] = -bbox[i][0]
            bbox[i][0] = 0

        if bbox[i][1] > shape_old[i]:
            pad[i][1] = bbox[i][1] - shape_old[i]

    bbox[0][1] = bbox[0][0] + size[0]
    bbox[1][1] = bbox[1][0] + size[1]
    bbox[2][1] = bbox[2][0] + size[2]

    img_new = np.pad(img, pad, 'constant', constant_values=c_val)
    return img_new[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1]]


def cropImage2DAt(img, center, size, c_val=None):
    if c_val is None:
        c_val = img.min()

    shape_old = img.shape[-2:]
    bbox = [[0, 0], [0, 0]]
    bbox[0][0] = int(center[0] - size[0] // 2)
    bbox[0][1] = bbox[0][0] + size[0]
    bbox[1][0] = int(center[1] - size[1] // 2)
    bbox[1][1] = bbox[1][0] + size[1]

    pad = [[0, 0], [0, 0]]
    for i in range(2):
        if bbox[i][0] < 0:
            pad[i][0] = -bbox[i][0]
            bbox[i][0] = 0

        if bbox[i][1] > shape_old[i]:
            pad[i][1] = bbox[i][1] - shape_old[i]

    bbox[0][1] = bbox[0][0] + size[0]
    bbox[1][1] = bbox[1][0] + size[1]

    if len(img.shape) == 2:
        img_new = np.pad(img, pad, 'constant', constant_values=c_val)
        return img_new[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1]]
    else:  # (c, h, w)
        img_new = np.pad(img, [[0, 0]]+pad, 'constant', constant_values=c_val)
        return img_new[:, bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1]]


def flip(img, vertical=False, horizontal=False):
    ## img: 2D/3D data; if 3D, the dims are (channel, height, width)
    ## vertical: up-down (dim 0 for 2D, dim 1 for 3D), horizontal: left-right (dim 1 for 2D, dim 2 for 3D)
    if vertical and horizontal:
        return np.flip(img, (-2, -1)).copy()
    elif horizontal:
        return np.flip(img, -1).copy()
    elif vertical:
        return np.flip(img, -2).copy()
    else:
        return img


def window_image(img, window_center, window_width):
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img = np.clip(img, img_min, img_max)
    return img
