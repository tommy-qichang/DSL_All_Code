
import surface_distance
import torch
import numpy as np

def hausdorff_distance95_avg(output, target, misc):
    channel_num = output.shape[1]
    sds = []
    for i in range(1, channel_num):
        sd = hausdorff_distance95_calc(output, target, misc, i)
        sds.append(sd)

    return np.average(np.array(sds))


def hausdorff_distance95_calc(output, target, misc, label_idx):
    if "spacing_mm" in misc:
        spacing_mm = misc['spacing_mm']
    else:
        spacing_mm = 1
    with torch.no_grad():
        pred = torch.argmax(output, 1)
        pred_bool = (pred == label_idx)
        target_bool = (target == label_idx)
        assert pred_bool.shape == target_bool.shape
        pred_bool = pred_bool.cpu().numpy()
        target_bool = target_bool.cpu().numpy()
        sd = surface_distance.compute_surface_distances(target_bool, pred_bool, spacing_mm)
        hd = surface_distance.compute_robust_hausdorff(sd, 95)
    return hd
