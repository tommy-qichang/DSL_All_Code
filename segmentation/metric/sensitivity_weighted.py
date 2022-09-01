import torch
from imblearn.metrics import sensitivity_score


def sensitivity_weighted(output, target):
    """
    Sensitivity = TP / (TP + FN)
    :param output: Batch x Channel x ....
    :param target: Batch x ....
    :return:
    """
    with torch.no_grad():
        if len(output.shape) == (len(target.shape) + 1):
            # reverse one-hot encode output
            output = torch.argmax(output, 1)

        assert output.shape == target.shape, "The output size should be the same or one dimension more than the shape of target."

        output = output.cpu().detach().numpy()
        target = target.cpu().detach().numpy()
        score = sensitivity_score(target, output, average='weighted')

    return score
