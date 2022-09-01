from torch.nn import functional as F


def nll_loss(output, target, reduction='mean'):
    return F.nll_loss(output, target, reduction=reduction)