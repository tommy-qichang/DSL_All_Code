
def dice_loss_multilabel(input, target, class_weights=None, n_classes=1):

    if class_weights is None:
        class_weights = [1]
    smooth = 1.
    loss = 0.
    for c in range(n_classes):
        iflat = input[:, c].view(-1)
        tflat = target[:, c].view(-1)
        intersection = (iflat * tflat).sum()

        w = class_weights[c]
        loss += w * (1 - ((2. * intersection + smooth) /
                          (iflat.sum() + tflat.sum() + smooth)))
    return loss
