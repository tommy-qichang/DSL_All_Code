from torch.nn import functional as F


def bce_loss_with_logits(output, target, misc):
    return F.binary_cross_entropy_with_logits(output, target)