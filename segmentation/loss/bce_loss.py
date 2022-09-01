import torch
from torch.nn import functional as F


def bce_loss(output, target, misc, weight=None):
    if weight is None:
        return F.binary_cross_entropy(output, target)
    else:
        assert len(weight) == len(
            torch.unique(target)), "The weight array should as the same size as target label type number"
        weight = torch.Tensor(weight)
        return F.binary_cross_entropy(output, target, weight=weight)
