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

    syn_img = float_to_uint_img(fake_B, None, 1)  #, minv=-1, maxv=1)

    label = A
    # values = np.unique(label)
    # maxv = len(values)-1
    # for v in values[::-1]:
    #     label[label==v] = maxv
    #     maxv -= 1

    if len(label.shape) == 3:
        label = label[0]
    label = float_to_uint_img(label, None, 0, minv=0, maxv=1)

    realdata = float_to_uint_img(B, None, 1, minv=-1, maxv=1)

    return label, realdata, syn_img


class MyModelTrainer(ModelTrainer):
    def __init__(self, model, id, args=None):
        super().__init__(model, args)
        self.id = id
        self.node_name = 'Client {}'.format(id) if id > -1 else 'Server'

        self.model.setup(args)

    def get_model_params(self):
        logging.info('[{}] Obtain model parameters'.format(self.node_name))
        return self.model.get_weights()

    def set_model_params(self, model_parameters):
        logging.info('[{}] Updating model'.format(self.node_name))
        self.model.load_weights(model_parameters)

    def train(self, train_data, device, args=None):
        self._train(train_data)
        logging.info('[{}] Train done'.format(self.node_name))

    def _train(self, train_data):
        model = self.model
        args = self.args

        if self.args.backbone_freezed:
            logging.info('[{0}] Training (Backbone Freezed) for {1} Epochs'.format(self.node_name, self.args.epochs))
        else:
            logging.info('[{0}] Training for {1} Epochs'.format(self.node_name, self.args.epochs))

        epoch_loss_D = []
        epoch_loss_G = []

        for epoch in range(args.epochs):
            t = time.time()
            batch_loss_D = []
            batch_loss_G = []
            logging.info('[{0}] Train Epoch: {1}'.format(self.node_name, epoch))

            for (batch_idx, batch) in enumerate(train_data):

                model.set_input(batch)  # unpack data from dataset and apply preprocessing
                loss_D, loss_G = model.optimize_parameters()  # calculate loss functions, get gradients, update network weights

                batch_loss_D.append(loss_D)
                batch_loss_G.append(loss_G)
                if (batch_idx % 100 == 0):
                    logging.info('[{0}] Train Iteration: {1}, LossD: {2}, lossG: {3}, Time Elapsed: {4}'.format(self.node_name, batch_idx, loss_D, loss_G, (time.time()-t)/60))

            if len(batch_loss_D) > 0:
                epoch_loss_D.append(sum(batch_loss_D) / len(batch_loss_D))
                epoch_loss_G.append(sum(batch_loss_G) / len(batch_loss_G))
                logging.info('[{}]. Local Training Epoch: {} \tLossD: {:.6f}\tLossG: {:.6f}'.format(self.node_name,
                                                                epoch, sum(epoch_loss_D) / len(epoch_loss_D), sum(epoch_loss_G) / len(epoch_loss_G)))

            model.update_learning_rate()  # update learning rates at the end of every epoch.

    def test(self, test_data, device, args=None):

        logging.info('[{name}] Testing on Test Dataset'.format(name=self.node_name))
        test_evaluation_metrics = self._infer(test_data, device)
        # logging.info("Testing Complete for client {}".format(self.id))
        return test_evaluation_metrics

    def test_train(self, train_data, device):
        logging.info('[{name}] Testing on Train Dataset'.format(name=self.node_name))
        train_evaluation_metrics = self._infer(train_data, device)
        return train_evaluation_metrics

    def test_visual(self, test_data, device):
        evaluation_results = {'batch_label': [], 'batch_img': [], 'batch_syn_img': [], 'batch_key': []}

        with torch.no_grad():
            batch = next(iter(test_data))

            self.model.set_input(batch)
            syn_img = self.model.forward()

            A = batch['A'].detach().cpu().numpy()
            B = batch['B'].detach().cpu().numpy()
            for j in range(syn_img.shape[0]):
                label, img, syn = convert_to_img(A[j], B[j], syn_img[j])
                evaluation_results['batch_label'].append(label)
                evaluation_results['batch_img'].append(img)
                evaluation_results['batch_syn_img'].append(syn)
                evaluation_results['batch_key'].append(batch['key'][j])

        return evaluation_results

    def _infer(self, test_data, device):
        t = time.time()
        loss_D = loss_D_fake = loss_D_real = loss_G = loss_G_GAN = loss_G_L1 = loss_G_perceptual = 0
        test_total = 0.

        with torch.no_grad():
            for (batch_idx, batch) in enumerate(test_data):
                loss_dict = self.model.evaluate(batch)

                loss_D += loss_dict['loss_D']
                loss_D_fake += loss_dict['loss_D_fake']
                loss_D_real += loss_dict['loss_D_real']
                loss_G += loss_dict['loss_G']
                loss_G_GAN += loss_dict['loss_G_GAN']
                loss_G_L1 += loss_dict['loss_G_L1']
                loss_G_perceptual += loss_dict['loss_G_perceptual']

                test_total += 1  #batch['A'].size(0)

                if batch_idx % 100 == 0:
                    logging.info('[{0}] Test Iteration: {1}, Loss_D: {2}, Loss_G: {3}, Time Elapsed: {4}'.format(self.node_name, batch_idx,
                                                                                                                     loss_dict['loss_D'],
                                                                                                                     loss_dict['loss_G'],
                                                                                                                     (time.time() - t) / 60))

        # Evaluation Metrics (Averaged over number of samples)
        loss_D = loss_D / test_total
        loss_D_fake = loss_D_fake / test_total
        loss_D_real = loss_D_real / test_total
        loss_G = loss_G / test_total
        loss_G_GAN = loss_G_GAN / test_total
        loss_G_L1 = loss_G_L1 / test_total
        loss_G_perceptual = loss_G_perceptual / test_total

        # logging.info("Client={0}, loss_D={1}, loss_D_fake={2}, loss_D_real={3}, loss_G={4}, loss_G_GAN={5}, loss_G_L1={6}, loss_G_perceptual={7}".format(
        #         self.id, loss_D, loss_D_fake, loss_D_real, loss_G, loss_G_GAN, loss_G_L1, loss_G_perceptual))
        eval_metrics = EvaluationMetricsKeeper(loss_D, loss_G, loss_D_fake, loss_D_real, loss_G_GAN, loss_G_L1, loss_G_perceptual)
        return eval_metrics

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False
