
import metric.surface_distance.metrics as metrics
# import torch
import numpy as np

def hd95_and_sd_3D(output, target, misc):
    if "spacing_mm" in misc:
        spacing_mm = misc['spacing_mm']
    else:
        spacing_mm = [1, 1, 1]

    pred = np.argmax(output, axis=1)
    pred_bool = (pred > 0)
    target_bool = (target > 0)

    assert pred_bool.shape == target_bool.shape

    if not np.any(pred_bool):
        return pred.shape[-1] * spacing_mm[-1], pred.shape[-1] * spacing_mm[-1]

    sd = metrics.compute_surface_distances(target_bool, pred_bool, spacing_mm)
    hd95 = metrics.compute_robust_hausdorff(sd, 95)
    gt_to_pred = sd['distances_gt_to_pred'].mean()
    pred_to_pred = sd['distances_pred_to_gt'].mean()
    assd = np.array([gt_to_pred, pred_to_pred]).mean()

    return hd95, assd
