import numpy as np
from metric.hausdorff95 import hausdorff95

from metric.dice import one_cls_dice


def hd_sparse2d(output, target, misc=None):
    """
    Calculate dice for each critical slices, and average.
    :param output:Output dimension: Batch x Channel x X x Y (x Z) float
    :param target:Target dimension: Batch x X x Y (x Z) int:[0, Channel]
    :return:
    """
    assert misc is not None, "The misc should has at least one input tensor"
    z_index = np.unique(np.where(misc['input'].cpu().squeeze().numpy() > 0)[2])
    hd_arr = []
    for z in z_index:
        o = output[:, :, :, :, z]
        t = target[:, :, :, z]
        hd_score = hausdorff95(o,t)
        hd_arr.append(hd_score)

    return np.average(hd_arr)
