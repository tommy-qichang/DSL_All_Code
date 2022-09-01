import numpy as np

def aji(pred, target, istrain):
    """ target should be instance level label"""
    assert pred.shape == target.shape

    aji = 0.0
    for k in range(pred.shape[0]):
        aji += _AJI_fast(pred[k], target[k])
    aji /= pred.shape[0]
    return aji


def _AJI_fast(pred_arr, gt):
    gs, g_areas = np.unique(gt, return_counts=True)
    assert np.all(gs == np.arange(len(gs)))
    ss, s_areas = np.unique(pred_arr, return_counts=True)
    assert np.all(ss == np.arange(len(ss)))

    if len(ss) == 1:
        return 0

    i_idx, i_cnt = np.unique(np.concatenate([gt.reshape(1, -1), pred_arr.reshape(1, -1)]),
                             return_counts=True, axis=1)
    i_arr = np.zeros(shape=(len(gs), len(ss)), dtype=np.int)
    i_arr[i_idx[0], i_idx[1]] += i_cnt
    u_arr = g_areas.reshape(-1, 1) + s_areas.reshape(1, -1) - i_arr
    iou_arr = 1.0 * i_arr / u_arr

    i_arr = i_arr[1:, 1:]
    u_arr = u_arr[1:, 1:]
    iou_arr = iou_arr[1:, 1:]

    j = np.argmax(iou_arr, axis=1)

    c = np.sum(i_arr[np.arange(len(gs) - 1), j])
    u = np.sum(u_arr[np.arange(len(gs) - 1), j])
    used = np.zeros(shape=(len(ss) - 1), dtype=np.int)
    used[j] = 1
    u += (np.sum(s_areas[1:] * (1 - used)))
    return 1.0 * c / u