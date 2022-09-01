import torch.nn.functional as f


def mse(input, target, misc):
    return f.mse_loss(input, target)
