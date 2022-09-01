import torch


def accuracy_binary(output, target):
    with torch.no_grad():
        pred = (output > 0.5)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / target.numel()
