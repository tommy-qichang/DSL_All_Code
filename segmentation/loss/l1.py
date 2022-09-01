import torch.nn.functional as f


def l1(input, target, misc):
    # print(f"{input.size()}, {target.size()}")
    return f.l1_loss(input, target)
