import numpy as np
import torch
from utils.clean_noise import CleanNoise


def one_cls_dice(output, target, label_idx):
    """
    Calculate Dice metrics for one channel
    :param output:Output dimension: Batch x Channel x X x Y (x Z) float
    :param target:Target dimension: Batch x X x Y (x Z) int:[0, Channel]
    :param label_idx:
    :return:
    """
    eps = 0.0001

    pred = np.argmax(output, 1)

    clean = CleanNoise()

    pred_list = []
    for i in range(pred.shape[3]):
        pred_list.append(clean.clean_small_obj(pred[:,:,:,i].squeeze()))

    pred = np.stack(pred_list, axis=-1)[np.newaxis]

    pred_bool = (pred == label_idx)
    target_bool = (target == label_idx)
    intersection = pred_bool * target_bool
    sum_range = tuple(range(1, len(intersection.shape)))
    return 2* intersection.sum(sum_range) / (pred_bool.sum(sum_range) + target_bool.sum(sum_range) + eps)

    # with torch.no_grad():
    #     pred = torch.argmax(output, 1)
    #     pred_bool = (pred == label_idx)
    #     target_bool = (target == label_idx)
    #
    #     intersection = pred_bool * target_bool
    #     return (2 * int(intersection.sum())) / (int(pred_bool.sum()) + int(target_bool.sum()) + eps)


def dice_denoise(output, target, misc=None):
    """
    Calculate dice for all channels.
    :param output:Output dimension: Batch x Channel x X x Y (x Z) float
    :param target:Target dimension: Batch x X x Y (x Z) int:[0, Channel]
    :return:
    """
    channel_num = output.shape[1]
    # assert 1 < channel_num <= (target.max()+1)  # At least should have 1 foreground channel, and should have less than the target max.

    # We found numpy argmax is much faster than pytorch tensor
    output = output.detach().cpu().numpy()
    target = target.detach().cpu().numpy()

    dices = []
    for i in range(1, channel_num):
        dice = one_cls_dice(output, target, label_idx=i)
        dices += dice.tolist()

    return np.average(np.array(dices))
