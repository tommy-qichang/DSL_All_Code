from loss.bce_loss import bce_loss
from loss.dice_loss import dice_loss


def bce_and_dice_loss(output, target):
    return bce_loss(output, target) + dice_loss(output, target)
