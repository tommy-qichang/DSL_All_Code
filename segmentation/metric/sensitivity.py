import torch
from imblearn.metrics import sensitivity_score
from sklearn.metrics import confusion_matrix


def sensitivity(output, target, misc):
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

        output = output.flatten().cpu().detach().numpy()
        target = target.flatten().cpu().detach().numpy()

        tn, fp, fn, tp = confusion_matrix(target, output).ravel()
        score = tp / (tp+fn)

        misc['tn'] = tn
        misc['fp'] = fp

        # score = sensitivity_score(target, output, average='micro')

    return score
