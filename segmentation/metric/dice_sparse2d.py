import numpy as np

from metric.dice import one_cls_dice


def dice_sparse2d(output, target, misc=None):
    """
    Calculate dice for each critical slices, and average.
    :param output:Output dimension: Batch x Channel x X x Y (x Z) float
    :param target:Target dimension: Batch x X x Y (x Z) int:[0, Channel]
    :return:
    """
    assert misc is not None, "The misc should has at least one input tensor"
    z_index = np.unique(np.where(misc['input'].cpu().squeeze().numpy() > 0)[2])
    dice_arr = []
    for z in z_index:
        o = output[:, :, :, :, z]
        t = target[:, :, :, z]
        dice = one_cls_dice(o, t,label_idx=1)
        dice_arr.append(dice)


    return np.average(dice_arr)
