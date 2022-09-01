import torch


def binary_dice_loss(score, target, misc):
    target = target.view(-1).float()
    score = score.view(-1).float()
    # print(f"after sigmoid,max:{score.max()},min:{score.min()}")
    # assert score.max()==1,"score max should be 1"
    # assert score.min() == 0, "score min should be 0"
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss