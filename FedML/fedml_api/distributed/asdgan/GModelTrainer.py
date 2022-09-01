import logging, time

import numpy as np
import torch

try:
    from fedml_core.trainer.model_trainer import ModelTrainer
except ImportError:
    from FedML.fedml_core.trainer.model_trainer import ModelTrainer
    
from .utils import EvaluationMetricsKeeper, float_to_uint_img


def convert_to_img(A, B, fake_B):
    nc = B.shape[0]

    syn_img = float_to_uint_img(fake_B, (240, 240), 1, minv=-1, maxv=1)

    label = A
    # values = np.unique(label)
    # maxv = len(values)-1
    # for v in values[::-1]:
    #     label[label==v] = maxv
    #     maxv -= 1

    if len(label.shape) == 3:
        label = label[0]
    label = float_to_uint_img(label, (240, 240), 0, minv=0, maxv=1)

    realdata = float_to_uint_img(B, (240, 240), 1, minv=-1, maxv=1)

    return label, realdata, syn_img


class GModelTrainer(ModelTrainer):
    def __init__(self, model, id, args=None):
        super().__init__(model, args)
        self.id = id
        self.node_name = 'Server'

        self.model.setup(args)

    def get_model_params(self):
        logging.info('[{}] Obtain model parameters'.format(self.node_name))
        return self.model.get_weights()

    def set_model_params(self, model_parameters):
        logging.info('[{}] Updating model'.format(self.node_name))
        self.model.load_weights(model_parameters)

    def train(self, train_data, device, args=None):
        pass

    def train_forward_one_iter(self, A):
        input = {'A': A}
        self.model.set_input(input)
        fake_B = self.model.forward()
        return fake_B

    def train_optimize_one_iter(self, grad_fake_B):
        grad_fake_B = torch.tensor(grad_fake_B)
        self.model.backward_G(grad_fake_B)  # calculate graidents for G

    def test(self, test_data, device, args=None):

        logging.info('[{name}] Testing on Test Dataset'.format(name=self.node_name))
        test_evaluation_metrics = self._infer(test_data, device)
        # logging.info("Testing Complete for client {}".format(self.id))
        return test_evaluation_metrics

    def test_train(self, train_data, device):
        logging.info('[{name}] Testing on Train Dataset'.format(name=self.node_name))
        train_evaluation_metrics = self._infer(train_data, device)
        return train_evaluation_metrics

    def _infer(self, test_data, device):

        evaluation_results = {'batch_label': [], 'batch_img': [], 'batch_syn_img': [], 'batch_key': []}

        batch = next(iter(test_data))

        syn_img = self.model.evaluate(batch)
        A = batch['A'].detach().cpu().numpy()
        B = batch['B'].detach().cpu().numpy()
        for j in range(syn_img.shape[0]):
            label, img, syn = convert_to_img(A[j], B[j], syn_img[j])
            evaluation_results['batch_label'].append(label)
            evaluation_results['batch_img'].append(img)
            evaluation_results['batch_syn_img'].append(syn)
            evaluation_results['batch_key'].append(batch['key'][j])

        return evaluation_results

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False
