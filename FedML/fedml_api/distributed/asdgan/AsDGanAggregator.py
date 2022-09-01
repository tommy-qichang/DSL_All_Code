import matplotlib
matplotlib.use('agg')
# import copy
import logging
import time
# import torch
import torch.utils.data as data
import wandb
import numpy as np
# from torch import nn
import matplotlib.pyplot as plt

from .utils import Saver, EvaluationMetricsKeeper


class AsDGanAggregator(object):
    def __init__(self, worker_num, device, model, args, model_trainer, train_data_global, test_data_global):
        self.trainer = model_trainer
        self.worker_num = worker_num
        self.device = device
        self.args = args
        self.train_data_global = train_data_global
        self.train_dataloader = None
        self.test_data_global = test_data_global
        self.model_dict = dict()
        self.sample_num_dict = dict()
        self.flag_client_grad_uploaded_dict = dict()
        self.flag_client_label_uploaded_dict = dict()

        for idx in range(self.worker_num):
            self.flag_client_grad_uploaded_dict[idx] = False
            self.flag_client_label_uploaded_dict[idx] = False
        #self.model = model

        self.train_loss_D_client_dict = dict()
        self.train_loss_G_client_dict = dict()
        self.train_loss_D_fake_client_dict = dict()
        self.train_loss_D_real_client_dict = dict()
        self.train_loss_G_GAN_client_dict = dict()
        self.train_loss_G_L1_client_dict = dict()
        self.train_loss_G_perceptual_client_dict = dict()

        self.train_batch_iter_cnt = dict()

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

        self.epoch_idx = 0
        self.batch_idx = 0
        self.n_iter_epoch = 0
        self.n_iter_display = 100
        self.batch_size = 0
        self.sample_client_indexes = []
        self.grad_dict = dict()
        self.finished = False

        logging.info('[Server] Initializing FedGanAggregator with workers: {0}'.format(worker_num))

    def get_global_model_params(self):
        return self.trainer.get_model_params()

    def set_global_model_params(self, model_parameters):
        self.trainer.set_model_params(model_parameters)

    def add_client_label_data(self, client_id, label_data, key_data):
        self.train_data_global.add_client_labels(client_id, label_data, key_data)
        self.sample_num_dict[client_id] = len(label_data)
        self.flag_client_label_uploaded_dict[client_id] = True

    def check_all_label_receive(self):
        for idx in range(self.worker_num):
            if not self.flag_client_label_uploaded_dict[idx]:
                return False
        for idx in range(self.worker_num):
            self.flag_client_label_uploaded_dict[idx] = False
        return True

    def init_train_dataloader(self):
        logging.info('[Server] Init dataloader')
        self.train_dataloader = data.DataLoader(dataset=self.train_data_global, batch_size=self.args.batch_size, shuffle=True, drop_last=True,
                                                num_workers=self.args.dl_num_workers, pin_memory=True)
        if self.args.sample_method == 'balance':
            self.batch_size = self.args.batch_size * self.worker_num
        else:  # self.args.sample_method == 'uniform'
            self.batch_size = self.args.batch_size
        self.n_iter_epoch = len(self.train_dataloader)
        self.n_iter_display = min(int(self.n_iter_epoch / 2 + 0.5), self.n_iter_display)

    def forward_G(self):
        if (self.batch_idx+1) % self.n_iter_display == 0:
            t = time.localtime()
            current_time = time.strftime("%H:%M:%S", t)
            logging.info("[Server] Train at epoch {0}: {1} / {2}. {3}".format(self.epoch_idx, self.batch_idx, self.n_iter_epoch, current_time))
        batch = next(iter(self.train_dataloader))
        client_ids, A, key_ids, trans_para = batch
        # logging.info('[Server] forward_G client_ids: {0}, A: {1}, keys: {2}'.format(client_ids.size(), A.size(), key_ids.size()))
        asize = A.size()
        A = A.view(-1, asize[2], asize[3], asize[4])
        fake_B = self.trainer.train_forward_one_iter(A)
        client_ids = client_ids.view(-1).detach().cpu().numpy()
        key_ids = key_ids.view(-1).detach().cpu().numpy()
        trans_para = trans_para.view(-1, *trans_para.shape[2:]).detach().cpu().numpy()

        forward_data = {}
        sample_idx_dict = {}
        for sample_id, (id, key, fb, tran) in enumerate(zip(client_ids, key_ids, fake_B, trans_para)):
            if id not in forward_data.keys():
                forward_data[id] = {}
                forward_data[id]['key'] = []
                forward_data[id]['fake'] = []
                forward_data[id]['trans_para'] = []
                sample_idx_dict[id] = []
            forward_data[id]['key'].append(self.train_data_global.get_key_str(key))
            forward_data[id]['fake'].append(fb)
            forward_data[id]['trans_para'].append(tran)
            sample_idx_dict[id].append(sample_id)
        # logging.info('[Server] forward_G forward_data: {0}'.format(forward_data.keys()))
        self.sample_client_indexes = list(forward_data.keys())
        return forward_data, sample_idx_dict, (self.n_iter_epoch - self.batch_idx)

    def backward_G(self, sample_idx_dict):
        grad_fake_B_list = []
        training_num = 0
        sample_idx = []
        for idx in self.sample_client_indexes:
            grad_fake_B_list.append((self.sample_num_dict[idx], self.grad_dict[idx]))
            training_num += self.sample_num_dict[idx]
            sample_idx.append(sample_idx_dict[idx])

        # logging.info("[Server] Aggregating grads...... {0}, {1}".format(len(self.grad_dict), len(grad_fake_B_list)))
        grad_fake_B = np.zeros([self.batch_size]+list(grad_fake_B_list[0][1].shape)[1:], dtype='float32')

        for i in range(0, len(grad_fake_B_list)):
            local_sample_number, local_grad = grad_fake_B_list[i]
            if self.args.sample_method == 'balance':
                w = local_sample_number / training_num
            else:
                w = 1.0
            grad_fake_B[sample_idx[i]] = local_grad * w
        # logging.info("[Server] shape of grad_fake_B: " + str(grad_fake_B.shape))
        self.trainer.train_optimize_one_iter(grad_fake_B)

        self.batch_idx += 1

        if self.batch_idx == self.n_iter_epoch:
            self.epoch_idx += 1
            if self.epoch_idx == self.args.epochs:
                self.finished = True
            self.batch_idx = 0
            self.trainer.model.update_learning_rate()  # update learning rates at the end of every epoch.
        return

    def add_local_grad(self, index, grad_fake_B, model):
        # logging.info("[Server] Add grad index: {}".format(index))
        self.grad_dict[index] = grad_fake_B
        self.model_dict[index] = model
        self.flag_client_grad_uploaded_dict[index] = True

    def check_whether_all_receive(self):
        # for idx in range(self.worker_num):
        for idx in self.sample_client_indexes:
            if not self.flag_client_grad_uploaded_dict[idx]:
                return False
        for idx in range(self.worker_num):
            self.flag_client_grad_uploaded_dict[idx] = False
        return True

    def client_sampling(self, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            self.sample_client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(self.epoch_idx)  # make sure for each comparison, we are selecting the same clients each round
            self.sample_client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        logging.info("[Server] Sample clients: {}".format(self.sample_client_indexes))
        return self.sample_client_indexes

    def add_client_test_result(self, client_idx, train_eval_metrics:EvaluationMetricsKeeper, test_eval_metrics:EvaluationMetricsKeeper):
        # logging.info("[Server] Adding client test result : {}".format(client_idx))

        # Populating Training Dictionary
        if (self.epoch_idx+1) % self.args.evaluation_frequency == 0:
            self.train_loss_D_client_dict[client_idx] = self.train_loss_D_client_dict.setdefault(client_idx, 0) + train_eval_metrics.loss_D
            self.train_loss_G_client_dict[client_idx] = self.train_loss_G_client_dict.setdefault(client_idx, 0) + train_eval_metrics.loss_G
            self.train_loss_D_fake_client_dict[client_idx] = self.train_loss_D_fake_client_dict.setdefault(client_idx, 0) + train_eval_metrics.loss_D_fake
            self.train_loss_D_real_client_dict[client_idx] = self.train_loss_D_real_client_dict.setdefault(client_idx, 0) + train_eval_metrics.loss_D_real
            self.train_loss_G_GAN_client_dict[client_idx] = self.train_loss_G_GAN_client_dict.setdefault(client_idx, 0) + train_eval_metrics.loss_G_GAN
            self.train_loss_G_L1_client_dict[client_idx] = self.train_loss_G_L1_client_dict.setdefault(client_idx, 0) + train_eval_metrics.loss_G_L1
            self.train_loss_G_perceptual_client_dict[client_idx] = self.train_loss_G_perceptual_client_dict.setdefault(client_idx, 0) + train_eval_metrics.loss_G_perceptual
            self.train_batch_iter_cnt[client_idx] = self.train_batch_iter_cnt.setdefault(client_idx, 0) + 1

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

    def output_global_acc_and_loss(self):
        if self.batch_idx > 0:
            return
        round_idx = self.epoch_idx - 1
        logging.info("[Server] ################## Output global accuracy and loss for round {} :".format(round_idx))

        stats_train = None
        if (round_idx + 1) % self.args.evaluation_frequency == 0:
            train_loss_D = np.array([self.train_loss_D_client_dict[k] / self.train_batch_iter_cnt[k] for k in self.train_loss_D_client_dict.keys()]).mean()
            train_loss_G = np.array([self.train_loss_G_client_dict[k] / self.train_batch_iter_cnt[k] for k in self.train_loss_G_client_dict.keys()]).mean()
            train_loss_D_fake = np.array([self.train_loss_D_fake_client_dict[k] / self.train_batch_iter_cnt[k] for k in self.train_loss_D_fake_client_dict.keys()]).mean()
            train_loss_D_real = np.array([self.train_loss_D_real_client_dict[k] / self.train_batch_iter_cnt[k] for k in self.train_loss_D_real_client_dict.keys()]).mean()
            train_loss_G_GAN = np.array([self.train_loss_G_GAN_client_dict[k] / self.train_batch_iter_cnt[k] for k in self.train_loss_G_GAN_client_dict.keys()]).mean()
            train_loss_G_L1 = np.array([self.train_loss_G_L1_client_dict[k] / self.train_batch_iter_cnt[k] for k in self.train_loss_G_L1_client_dict.keys()]).mean()
            train_loss_G_perceptual = np.array([self.train_loss_G_perceptual_client_dict[k] / self.train_batch_iter_cnt[k] for k in self.train_loss_G_perceptual_client_dict.keys()]).mean()

            self.train_loss_D_client_dict = dict()
            self.train_loss_G_client_dict = dict()
            self.train_loss_D_fake_client_dict = dict()
            self.train_loss_D_real_client_dict = dict()
            self.train_loss_G_GAN_client_dict = dict()
            self.train_loss_G_L1_client_dict = dict()
            self.train_loss_G_perceptual_client_dict = dict()
            self.train_batch_iter_cnt = dict()

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

            logging.info('[Server] Saving G Model Checkpoint')
            filename = "G_aggregated_checkpoint_ep%d.pth.tar" % (round_idx + 1)
            saver_state = {'best_lossG': train_loss_G, 'round': round_idx + 1, 'state_dict': self.trainer.model.get_weights(),
                           'train_data_evaluation_metrics': stats_train}
            self.saver.save_checkpoint(saver_state, False, filename)

            if self.args.save_client_model:
                for client_idx in range(self.worker_num):
                    if self.model_dict[client_idx] is None:
                        continue
                    # self.best_lossG_clients[client_idx] = test_lossG
                    logging.info('[Server] Saving D Model Checkpoint of client {0}'.format(client_idx))
                    filename = "client_{0}_D_checkpoint_ep{1}.pth.tar".format(client_idx, round_idx + 1)
                    saver_state = {'round': round_idx + 1, 'state_dict': self.model_dict[client_idx]}
                    self.saver.save_checkpoint(saver_state, False, filename)

        if self.test_data_global:
            evaluation_results = self.trainer.test(self.test_data_global, self.device)

            nc = evaluation_results['batch_img'][0].shape[0]
            num_r = min(3, len(evaluation_results['batch_label']))

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
                    if 't2' in self.args.dataset:
                        mod_names = ['T2']
                    elif 't1' in self.args.dataset:
                        mod_names = ['T1c']
                    elif 'flair' in self.args.dataset:
                        mod_names = ['Flair']
            else:
                num_c = 1 + 2
                show_RGB = False
                mod_names = ['img']

            ctr = 0
            plt.figure(figsize=(num_c*3, 9))
            sample_idx = np.random.choice(len(evaluation_results['batch_label']), num_r, replace=False)
            for i in range(num_r):
                label = evaluation_results['batch_label'][sample_idx[i]]
                real_img = evaluation_results['batch_img'][sample_idx[i]]
                syn_img = evaluation_results['batch_syn_img'][sample_idx[i]]
                key = evaluation_results['batch_key'][sample_idx[i]]

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
                            plt.title('real_'+mod_names[k])
                        plt.axis('off')

            # Test Logs
            plt.tight_layout()
            wandb.log({"Test/samples": [wandb.Image(plt, caption="syn vs real")]})
            plt.close()

            # if test_lossG < self.best_lossG:
            #     logging.info('[Server] Saving G Model Checkpoint --> Previous lossG:{0}; Improved lossG:{1}'.format(self.best_lossG, test_lossG))
            #     is_best = True
            #     self.best_lossG = test_lossG
            #     saver_state = {'best_lossG': self.best_lossG, 'round': round_idx + 1, 'state_dict': self.trainer.model.get_weights(),
            #                    'test_data_evaluation_metrics': stats}
            #
            #     if stats_train is not None:
            #         saver_state['train_data_evaluation_metrics'] = stats_train
            #
            #     self.saver.save_checkpoint(saver_state, is_best)
