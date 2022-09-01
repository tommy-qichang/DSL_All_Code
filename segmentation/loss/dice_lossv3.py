import torch

from loss.lib.Losses import DiceLoss


def dice_lossv3(input, target):
    """
    input is a torch variable of size BatchxnclassesxHxW representing log probabilities for each class
    target is a 1-hot representation of the groundtruth, shoud have same size as the input
    """
    target = target.unsqueeze(1)
    target = torch.cat((1 - target.float(), target.float()), 1)
    assert input.size() == target.size(), "Input sizes must be equal."
    assert input.dim() == 5, "Input must be a 5D Tensor."

    loss = DiceLoss()
    return loss(input, target)
