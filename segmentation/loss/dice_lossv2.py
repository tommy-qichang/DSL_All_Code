import torch
from torch.autograd import Function
from torch.nn.functional import softmax
from itertools import repeat
import numpy as np


def dice_lossv2(input, target):
    """
    input is a torch variable of size BatchxnclassesxHxW representing log probabilities for each class
    target is a 1-hot representation of the groundtruth, shoud have same size as the input
    """
    target = target.unsqueeze(1)
    target = torch.cat((1 - target.float(), target.float()), 1)
    assert input.size() == target.size(), "Input sizes must be equal."
    assert input.dim() == 5, "Input must be a 5D Tensor."
    # range = [torch.min(target).data[0], torch.max(target).data[0]]
    # uniques=np.unique(target.numpy())
    # assert set(range) <= set([0, 1]), "target must only contain zeros and ones"
    smooth = 0.00001

    input = input.permute(0, 2, 3, 4, 1).contiguous()
    input = input.view(input.numel() // 2, 2)
    probs = softmax(input, dim=1)

    target = target.permute(0, 2, 3, 4, 1).contiguous()
    target = target.view(target.numel() // 2, 2)

    intersection = (probs * target).sum(dim=0)

    iflat = (probs * probs).sum(dim=0)

    tflat = (target * target).sum(dim=0)

    dice_score = ((2. * intersection + smooth) / (iflat + tflat + smooth))

    dice_total = 1 - dice_score[1]  # we ignore bg dice val, and take the fg

    return dice_total

