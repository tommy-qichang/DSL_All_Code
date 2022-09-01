import torch
from torch.nn import functional as F


def cross_entropy_loss(output, target, misc, weight=None):
    if weight is None:
        return F.cross_entropy(output, target)
    else:
        weight = torch.Tensor(weight)
        return F.cross_entropy(output, target, weight=weight)
