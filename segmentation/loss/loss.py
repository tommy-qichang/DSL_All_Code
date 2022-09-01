import torch.nn.functional as F

from loss.bce_loss import bce_loss
from loss.bce_loss_with_logits import bce_loss_with_logits
from loss.cross_entropy_loss import cross_entropy_loss
from loss.dice_loss import dice_loss
from loss.nll_loss import nll_loss

if __name__ == '__main__':
    import torch
    torch.manual_seed(32)
    output = torch.rand([2, 2, 3, 4])
    target = torch.tensor([[[0, 1, 1, 1], [0, 0, 1, 0], [1, 1, 0, 0]],
                           [[1, 1, 0, 0], [1, 0, 0, 1], [0, 0, 0, 0]]])

    print('dice loss: {:.4f}'.format(dice_loss(output[:, 1, :, :], target.float())))
    print('ce loss: {:.4f}'.format(cross_entropy_loss(output, target)))
    print('logsoftmax + nll loss: {:.4f}'.format(nll_loss(F.log_softmax(output, dim=1).float(), target)))
    print('bce loss: {:.4f}'.format(bce_loss(F.softmax(output, dim=1)[:, 1, :, :], target.float())))
    print('bce loss with digits: {:.4f}'.format(bce_loss_with_logits(output[:, 1, :, :], target.float())))