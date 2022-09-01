import numpy as np
import torch

from metric.dice import one_cls_dice


def dice_cat3(output, target, misc):
    """
    Calculate dice for all channels.
    :param output:Output dimension: Batch x Channel x X x Y (x Z) float
    :param target:Target dimension: Batch x X x Y (x Z) int:[0, Channel]
    :return:
    """

    return one_cls_dice(output, target, label_idx=3)
