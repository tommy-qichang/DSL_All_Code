import torch
from numpy.core.umath_tests import inner1d
from skimage import morphology
import numpy as np

def hausdorff95(output, target, label_idx):
    with torch.no_grad():
        # pred = torch.sigmoid(output) > 0.5
        # pred = output > 0.5
        pred = torch.argmax(output, 1)
        pred = (pred == 1)
        assert pred.size() == target.size()

        pred = pred.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        hd95 = 0.0
        n = 0
        for k in range(pred.shape[0]):
            if np.count_nonzero(pred[k]) > 0:  # need to handle blank prediction
                n += 1
                pred_contours = pred[k] & (~morphology.binary_erosion(pred[k]))
                target_contours = target[k] & (~morphology.binary_erosion(target[k]))
                pred_ind = np.argwhere(pred_contours)
                target_ind = np.argwhere(target_contours)
                hd95 += _haus_dist_95(pred_ind, target_ind)

    return (hd95+0.0001)/(n+0.0001)


def _haus_dist_95(A, B):
    """ compute the 95 percentile hausdorff distance """
    # Find pairwise distance
    D_mat = np.sqrt(inner1d(A, A)[np.newaxis].T + inner1d(B, B) - 2 * (np.dot(A, B.T)))
    dist1 = np.min(D_mat, axis=0)
    dist2 = np.min(D_mat, axis=1)
    hd95 = np.percentile(np.hstack((dist1, dist2)), 95)

    # hd = np.max(np.array([np.max(np.min(D_mat, axis=0)), np.max(np.min(D_mat, axis=1))]))

    return hd95
