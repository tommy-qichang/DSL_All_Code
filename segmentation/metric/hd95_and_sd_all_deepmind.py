
import metric.surface_distance.metrics as metrics
# import torch
import numpy as np

def hd95_and_sd_all(output, target, misc):
    channel_num = output.shape[1]
    pred = np.argmax(output, axis=1)
    sds = []
    hds = []
    for i in range(1, channel_num):
        hd, sd = h95_and_sd_calc(pred, target, misc, i)
        hds.append(hd)
        sds.append(sd)

    return hds, sds


def h95_and_sd_calc(pred, target, misc, label_idx):
    if "spacing_mm" in misc:
        spacing_mm = misc['spacing_mm']
    else:
        spacing_mm = [1, 1, 1]

    pred_bool = (pred == label_idx)
    target_bool = (target == label_idx)
    assert pred_bool.shape == target_bool.shape

    if not np.any(pred_bool):
        return pred.shape[-1] * spacing_mm[-1], pred.shape[-1] * spacing_mm[-1]

    sd = metrics.compute_surface_distances(target_bool, pred_bool, spacing_mm)
    hd95= metrics.compute_robust_hausdorff(sd, 95)
    gt_to_pred = sd['distances_gt_to_pred'].mean()
    pred_to_pred = sd['distances_pred_to_gt'].mean()
    assd = np.array([gt_to_pred, pred_to_pred]).mean()

    return hd95, assd
