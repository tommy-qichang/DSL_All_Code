# import torch
# from imblearn.metrics import specificity_score


def specificity(output, target, misc):
    """
    Specificity = TN / (TN + FP)
    :param output: Batch x Channel x ....
    :param target: Batch x ....
    :return:
    """
    tn = misc['tn']
    fp = misc['fp']
    score = tn / (tn + fp)
    del misc['tn']
    del misc['fp']

    # with torch.no_grad():
    #     if len(output.shape) == (len(target.shape) + 1):
    #         # reverse one-hot encode output
    #         output = torch.argmax(output, 1)
    #
    #     assert output.shape == target.shape, "The output size should be the same or one dimension more than the shape of target."
    #
    #     # output = output.flatten().cpu().detach().numpy()
    #     # target = target.flatten().cpu().detach().numpy()
    #     #
    #     # tn, fp, fn, tp = confusion_matrix(target, output).ravel()
    #     score = tn/(tn + fp)
    #

        # score = specificity_score(target, output, average='micro')

    return score
