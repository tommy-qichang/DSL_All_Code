import torch
from torch.nn import functional as F


def make_one_hot(labels, C=2):
    size_list = labels.size()
    new_size = [size_list[0]] + [C] + list(size_list[1:])
    one_hot = torch.zeros(new_size).cuda()
    target = one_hot.scatter_(1, labels.unsqueeze(1), 1)

    return target


def dice_loss(input, target, misc=None):
    C = input.size(1)

    input = input.permute(0, 2, 3, 4, 1).contiguous().view(-1, C)
    input = F.softmax(input, dim=1)

    y_onehot = make_one_hot(target.view(-1, 1), C).view(-1, C)
    smooth = 1

    intersection = (input * y_onehot).sum(0)
    dice = ((2. * intersection + smooth) /
            (input.sum(0) + y_onehot.sum(0) + smooth))
    # print(dice)
    return 1 - dice[1:].mean()

# def dice_loss(output, target, eps=1e-7):
#     """Computes the Sørensen–Dice loss.
#     Note that PyTorch optimizers minimize a loss. In this
#     case, we would like to maximize the dice loss so we
#     return the negated dice loss.
#     Args:
#         target: a tensor of shape [B, 1, H, W].
#         output: a tensor of shape [B, C, H, W]. Corresponds to
#             the raw output or logits of the model.
#         eps: added to the denominator for numerical stability.
#     Returns:
#         dice_loss: the Sørensen–Dice loss.
#     """
#     num_classes = output.shape[1]
#     if num_classes == 1:
#         true_1_hot = torch.eye(num_classes + 1)[target.squeeze(1)]
#         true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
#         true_1_hot_f = true_1_hot[:, 0:1, :, :]
#         true_1_hot_s = true_1_hot[:, 1:2, :, :]
#         true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
#         pos_prob = torch.sigmoid(output)
#         neg_prob = 1 - pos_prob
#         probas = torch.cat([pos_prob, neg_prob], dim=1)
#     else:
#         true_1_hot = torch.eye(num_classes)[target.squeeze(1)]
#         true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
#         probas = F.softmax(output, dim=1)
#     true_1_hot = true_1_hot.type(output.type())
#     dims = (0,) + tuple(range(2, target.ndimension()))
#     intersection = torch.sum(probas * true_1_hot, dims)
#     cardinality = torch.sum(probas + true_1_hot, dims)
#     dice_loss = (2. * intersection / (cardinality + eps)).mean()
#     return (1 - dice_loss)
