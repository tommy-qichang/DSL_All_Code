import matplotlib
matplotlib.use('agg')
import copy
import logging
import time
import torch
import wandb
import numpy as np
from torch import nn
import matplotlib.pyplot as plt

from .utils import transform_list_to_tensor, Saver, EvaluationMetricsKeeper


class FedGanAggregator(object):
    def __init__(self, worker_num, device, model, args, model_trainer, train_data_global, test_data_global):
        self.trainer = model_trainer
        self.worker_num = worker_num
        self.device = device
        self.args = args
        self.train_data_global = train_data_global
        self.test_data_global = test_data_global
        self.model_dict = dict()
        self.sample_num_dict = dict()
        self.flag_client_model_uploaded_dict = dict()

        for idx in range(self.worker_num):
            self.flag_client_model_uploaded_dict[idx] = False
        #self.model = model

        self.train_loss_D_client_dict = dict()
        self.train_loss_G_client_dict = dict()
        self.train_loss_D_fake_client_dict = dict()
        self.train_loss_D_real_client_dict = dict()
        self.train_loss_G_GAN_client_dict = dict()
        self.train_loss_G_L1_client_dict = dict()
        self.train_loss_G_perceptual_client_dict = dict()

        self.test_loss_D_client_dict = dict()
        self.test_loss_G_client_dict = dict()
        self.test_loss_D_fake_client_dict = dict()
        self.test_loss_D_real_client_dict = dict()
        self.test_loss_G_GAN_client_dict = dict()
        self.test_loss_G_L1_client_dict = dict()
        self.test_loss_G_perceptual_client_dict = dict()

        self.best_lossG = 10000.
        self.best_lossG_clients = dict()

        self.saver = Saver(args)
        self.saver.save_experiment_config()
        logging.info('[Server] experiment dir: {0}'.format(self.saver.experiment_dir))

        logging.info('[Server] Initializing FedGanAggregator with workers: {0}'.format(worker_num))

    def get_global_model_params(self):
        return self.trainer.get_model_params()

    def set_global_model_params(self, model_parameters):
        self.trainer.set_model_params(model_parameters)

    def add_local_trained_result(self, index, model_params, sample_num):
        logging.info("[Server] Add model index: {}".format(index))
        self.model_dict[index] = model_params
        self.sample_num_dict[index] = sample_num
        self.flag_client_model_uploaded_dict[index] = True

    def check_whether_all_receive(self):
        for idx in range(self.worker_num):
            if not self.flag_client_model_uploaded_dict[idx]:
                return False
        for idx in range(self.worker_num):
            self.flag_client_model_uploaded_dict[idx] = False
        return True

    def aggregate(self):
        start_time = time.time()
        model_list = []
        training_num = 0

        for idx in range(self.worker_num):
            # if self.args.is_mobile == 1:
            #     self.model_dict[idx] = transform_list_to_tensor(self.model_dict[idx])
            model_list.append((self.sample_num_dict[idx], self.model_dict[idx]))
            training_num += self.sample_num_dict[idx]

        logging.info("[Server] Aggregating...... {0}, {1}".format(len(self.model_dict),len(model_list)))

        (num0, averaged_params) = model_list[0]
        for net in averaged_params.keys():
            if isinstance(averaged_params[net], list):
                for inet in range(len(averaged_params[net])):
                    for k in averaged_params[net][inet].keys():
                        for i in range(0, len(model_list)):
                            local_sample_number, local_model_params = model_list[i]
                            w = local_sample_number / training_num
                            if i == 0:
                                averaged_params[net][inet][k] = local_model_params[net][inet][k] * w
                            else:
                                averaged_params[net][inet][k] += local_model_params[net][inet][k] * w
            else:
                for k in averaged_params[net].keys():
                    for i in range(0, len(model_list)):
                        local_sample_number, local_model_params = model_list[i]
                        w = local_sample_number / training_num
                        if i == 0:
                            averaged_params[net][k] = local_model_params[net][k] * w
                        else:
                            averaged_params[net][k] += local_model_params[net][k] * w

        # update the global model which is cached at the server side
        self.set_global_model_params(averaged_params)

        end_time = time.time()
        logging.info("[Server] Aggregate time cost: %d" % (end_time - start_time))
        return averaged_params

    def client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        logging.info("[Server] Sample clients: {}".format(client_indexes))
        return client_indexes

    def add_client_test_result(self, round_idx, client_idx, train_eval_metrics:EvaluationMetricsKeeper, test_eval_metrics:EvaluationMetricsKeeper):
        logging.info("[Server] Adding client test result : {}".format(client_idx))

        # Populating Testing Dictionary
        if test_eval_metrics:
            self.test_loss_D_client_dict[client_idx] = test_eval_metrics.loss_D
            self.test_loss_G_client_dict[client_idx] = test_eval_metrics.loss_G
            self.test_loss_D_fake_client_dict[client_idx] = test_eval_metrics.loss_D_fake
            self.test_loss_D_real_client_dict[client_idx] = test_eval_metrics.loss_D_real
            self.test_loss_G_GAN_client_dict[client_idx] = test_eval_metrics.loss_G_GAN
            self.test_loss_G_L1_client_dict[client_idx] = test_eval_metrics.loss_G_L1
            self.test_loss_G_perceptual_client_dict[client_idx] = test_eval_metrics.loss_G_perceptual

            test_eval_metrics_dict = {
                'loss_G': self.test_loss_G_client_dict[client_idx],
                'loss_D': self.test_loss_D_client_dict[client_idx],
                'loss_D_fake': self.test_loss_D_fake_client_dict[client_idx],
                'loss_D_real': self.test_loss_D_real_client_dict[client_idx],
                'loss_G_GAN': self.test_loss_G_GAN_client_dict[client_idx],
                'loss_G_L1': self.test_loss_G_L1_client_dict[client_idx],
                'loss_G_perceptual': self.test_loss_G_perceptual_client_dict[client_idx]
            }
            logging.info("[Server] Testing statistics of client {0}: {1}".format(client_idx, test_eval_metrics_dict))

        # Populating Training Dictionary
        if (round_idx+1) % self.args.evaluation_frequency == 0:
            self.train_loss_D_client_dict[client_idx] = train_eval_metrics.loss_D
            self.train_loss_G_client_dict[client_idx] = train_eval_metrics.loss_G
            self.train_loss_D_fake_client_dict[client_idx] = train_eval_metrics.loss_D_fake
            self.train_loss_D_real_client_dict[client_idx] = train_eval_metrics.loss_D_real
            self.train_loss_G_GAN_client_dict[client_idx] = train_eval_metrics.loss_G_GAN
            self.train_loss_G_L1_client_dict[client_idx] = train_eval_metrics.loss_G_L1
            self.train_loss_G_perceptual_client_dict[client_idx] = train_eval_metrics.loss_G_perceptual

            train_eval_metrics_dict = {
                'loss_G': self.train_loss_G_client_dict[client_idx],
                'loss_D': self.train_loss_D_client_dict[client_idx],
                'loss_D_fake': self.train_loss_D_fake_client_dict[client_idx],
                'loss_D_real': self.train_loss_D_real_client_dict[client_idx],
                'loss_G_GAN': self.train_loss_G_GAN_client_dict[client_idx],
                'loss_G_L1': self.train_loss_G_L1_client_dict[client_idx],
                'loss_G_perceptual': self.train_loss_G_perceptual_client_dict[client_idx]
            }
            logging.info("Training statistics of client {0}: {1}".format(client_idx, train_eval_metrics_dict))

            if self.args.save_client_model:
                best_lossG = self.best_lossG_clients.setdefault(client_idx, 10000.)
                if test_eval_metrics:
                    test_lossG = self.test_loss_G_client_dict[client_idx]
                else:
                    test_lossG = self.train_loss_G_client_dict[client_idx]

                if test_lossG < best_lossG:
                    self.best_lossG_clients[client_idx] = test_lossG
                    logging.info('[Server] Saving Model Checkpoint for Client: {0} --> Previous lossG:{1}; Improved lossG:{2}'.format(client_idx, best_lossG, test_lossG))
                    is_best = False
                    filename = "client" + str(client_idx) + "_checkpoint.pth.tar"
                    saver_state = {'best_lossG': test_lossG, 'round': round_idx + 1, 'state_dict': self.model_dict[client_idx],
                                   'train_data_evaluation_metrics': train_eval_metrics_dict}

                    if test_eval_metrics:
                        saver_state['test_data_evaluation_metrics'] = test_eval_metrics_dict

                    self.saver.save_checkpoint(saver_state, is_best, filename)

    def output_global_acc_and_loss(self, round_idx):
        logging.info("[Server] ################## Output global accuracy and loss for round {} :".format(round_idx))

        stats_train = None
        if (round_idx + 1) % self.args.evaluation_frequency == 0:
            train_loss_D = np.array([self.train_loss_D_client_dict[k] for k in self.train_loss_D_client_dict.keys()]).mean()
            train_loss_G = np.array([self.train_loss_G_client_dict[k] for k in self.train_loss_G_client_dict.keys()]).mean()
            train_loss_D_fake = np.array([self.train_loss_D_fake_client_dict[k] for k in self.train_loss_D_fake_client_dict.keys()]).mean()
            train_loss_D_real = np.array([self.train_loss_D_real_client_dict[k] for k in self.train_loss_D_real_client_dict.keys()]).mean()
            train_loss_G_GAN = np.array([self.train_loss_G_GAN_client_dict[k] for k in self.train_loss_G_GAN_client_dict.keys()]).mean()
            train_loss_G_L1 = np.array([self.train_loss_G_L1_client_dict[k] for k in self.train_loss_G_L1_client_dict.keys()]).mean()
            train_loss_G_perceptual = np.array([self.train_loss_G_perceptual_client_dict[k] for k in
                                                self.train_loss_G_perceptual_client_dict.keys()]).mean()

            # Train Logs
            wandb.log({"Train/Loss_G": train_loss_G, "round": round_idx})
            wandb.log({"Train/Loss_D": train_loss_D, "round": round_idx})
            wandb.log({"Train/Loss_D_fake": train_loss_D_fake, "round": round_idx})
            wandb.log({"Train/Loss_D_real": train_loss_D_real, "round": round_idx})
            wandb.log({"Train/Loss_G_GAN": train_loss_G_GAN, "round": round_idx})
            wandb.log({"Train/Loss_G_L1": train_loss_G_L1, "round": round_idx})
            wandb.log({"Train/Loss_G_perceptual": train_loss_G_perceptual, "round": round_idx})
            stats_train = {'LossG': train_loss_G,
                           'LossD': train_loss_D,
                           'LossD_fake': train_loss_D_fake,
                           'LossD_real': train_loss_D_real,
                           'LossG_GAN': train_loss_G_GAN,
                           'LossG_L1': train_loss_G_L1,
                           'LossG_perceptual': train_loss_G_perceptual}
            logging.info("[Server] Training statistics: {}".format(stats_train))

        # Test on global testing set
        test_eval_metrics = self.trainer.test(self.test_data_global, self.device)

        visual_results = self.trainer.test_visual(self.test_data_global, self.device)
        # add visual in wandb log
        nc = visual_results['batch_img'][0].shape[0]
        num_r = min(3, len(visual_results['batch_label']))

        if self.args.dataset.lower() == 'path':
            num_c = 1 + 2
            show_RGB = True
        elif self.args.dataset[:5].lower() == "brats":
            num_c = 1 + 2 * nc
            show_RGB = False
            if nc == 4:
                mod_names = ['T1', 'T2', 'Flair', 'T1c']
            elif nc == 3:
                mod_names = ['T1c', 'T2', 'Flair']
        else:
            num_c = 1 + 2
            show_RGB = False
            mod_names = ['img']
        ctr = 0
        plt.figure(figsize=(3*num_c, 9))
        sample_idx = np.random.choice(len(visual_results['batch_label']), num_r, replace=False)
        for i in range(num_r):
            label = visual_results['batch_label'][sample_idx[i]]
            real_img = visual_results['batch_img'][sample_idx[i]]
            syn_img = visual_results['batch_syn_img'][sample_idx[i]]
            # key = visual_results['batch_key'][sample_idx[i]]

            ctr += 1
            plt.subplot(num_r, num_c, ctr)
            plt.imshow(label, cmap="gray")
            if i == 0:
                plt.title("Label")
            plt.axis('off')

            if show_RGB:
                ctr += 1
                plt.subplot(num_r, num_c, ctr)
                plt.imshow(np.moveaxis(syn_img, 0, -1))
                if i == 0:
                    plt.title('syn')
                plt.axis('off')

                ctr += 1
                plt.subplot(num_r, num_c, ctr)
                plt.imshow(np.moveaxis(real_img, 0, -1))
                if i == 0:
                    plt.title('real')
                plt.axis('off')
            else:
                for k in range(nc):
                    ctr += 1
                    plt.subplot(num_r, num_c, ctr)
                    plt.imshow(syn_img[k], cmap="gray")
                    if i == 0:
                        plt.title(mod_names[k])
                    plt.axis('off')

                for k in range(nc):
                    ctr += 1
                    plt.subplot(num_r, num_c, ctr)
                    plt.imshow(real_img[k], cmap="gray")
                    if i == 0:
                        plt.title('real_' + mod_names[k])
                    plt.axis('off')

        # Test Logs
        plt.tight_layout()
        wandb.log({"Test/samples": [wandb.Image(plt, caption="syn vs real")]})
        plt.close()

        test_lossG = test_eval_metrics.loss_G
        test_lossD = test_eval_metrics.loss_D
        test_lossD_fake = test_eval_metrics.loss_D_fake
        test_lossD_real = test_eval_metrics.loss_D_real
        test_lossG_GAN = test_eval_metrics.loss_G_GAN
        test_lossG_L1 = test_eval_metrics.loss_G_L1
        test_lossG_perceptual = test_eval_metrics.loss_G_perceptual

        # Test Logs
        wandb.log({"Test/Loss_G": test_lossG, "round": round_idx})
        wandb.log({"Test/Loss_D": test_lossD, "round": round_idx})
        wandb.log({"Test/Loss_D_fake": test_lossD_fake, "round": round_idx})
        wandb.log({"Test/Loss_D_real": test_lossD_real, "round": round_idx})
        wandb.log({"Test/Loss_G_GAN": test_lossG_GAN, "round": round_idx})
        wandb.log({"Test/Loss_G_L1": test_lossG_L1, "round": round_idx})
        wandb.log({"Test/Loss_G_perceptual": test_lossG_perceptual, "round": round_idx})
        stats = {'LossG': test_lossG,
                 'LossD': test_lossD,
                 'LossD_fake': test_lossD_fake,
                 'LossD_real': test_lossD_real,
                 'LossG_GAN': test_lossG_GAN,
                 'LossG_L1': test_lossG_L1,
                 'LossG_perceptual': test_lossG_perceptual}

        logging.info("[Server] Testing statistics: {}".format(stats))

        if test_lossG < self.best_lossG:
            logging.info('[Server] Saving Model Checkpoint --> Previous lossG:{0}; Improved lossG:{1}'.format(self.best_lossG, test_lossG))
            is_best = True
            self.best_lossG = test_lossG
            saver_state = {'best_lossG': self.best_lossG, 'round': round_idx + 1, 'state_dict': self.trainer.model.get_weights(),
                           'test_data_evaluation_metrics': stats}

            if stats_train is not None:
                saver_state['train_data_evaluation_metrics'] = stats_train

            self.saver.save_checkpoint(saver_state, is_best)

        if (round_idx+1) % self.args.evaluation_frequency == 0:
            filename = "aggregated_checkpoint_%d.pth.tar" % (round_idx+1)
            saver_state = {'best_lossG': test_lossG, 'round': round_idx + 1, 'state_dict': self.trainer.model.get_weights(),
                           'test_data_evaluation_metrics': stats, 'train_data_evaluation_metrics': stats_train}
            self.saver.save_checkpoint(saver_state, False, filename)
