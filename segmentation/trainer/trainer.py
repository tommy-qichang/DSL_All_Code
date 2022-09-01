import random

import numpy as np
import torch
from torchvision.utils import make_grid

from base import BaseTrainer
from utils import inf_loop, MetricTracker


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        # for batch_idx, (data, target, misc) in enumerate(self.data_loader):
        for batch_idx, enumerate_result in enumerate(self.data_loader):
            if type(self.data_loader).__name__ == "GeneralDataLoader" or type(self.data_loader).__name__ == "FasterGeneralDataLoader":
                data, target, misc = enumerate_result
            else:
                data, target = enumerate_result
                misc = None
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            each_loss = []
            loss = self.criterion[0](output, target, misc)
            each_loss.append(float(loss))
            if len(self.criterion) > 1:
                for idx in range(1, len(self.criterion)):
                    loss2 = self.criterion[idx](output, target, misc)
                    loss = loss + loss2
                    each_loss.append(float(loss2))
            # loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target, misc))

            if batch_idx % self.log_step == 0:
                self.logger.debug(
                    f'Train Epoch: {epoch} {self._progress(batch_idx)} Loss:{loss.item():.6f}({each_loss})')
                vis_data = data.squeeze().clone()
                if type(output) is tuple:
                    pred_data = output[0].clone()
                    pred_data = torch.argmax(pred_data, 1).float()
                else:
                    pred_data = output.clone()
                    pred_data = torch.argmax(pred_data, 1).float()
                target_data = target.clone()
                if 'output_processed' in misc:
                    pred_data = misc['output_processed']
                if len(data.shape) >4 and data.shape[1] != 3 and data.shape[1] != 1:
                    vis_data = data[:, 0]
                if len(pred_data.shape) > 4 and pred_data.shape[1] != 3 and pred_data.shape[1] != 1:
                    pred_data = pred_data[:, 0]
                if len(target_data.shape) > 4 and target_data.shape[1] != 3 and target_data.shape[1] != 1:
                    target_data = target_data[:, 0]

                # classification task should not vis the prediction.
                if len(pred_data.shape)>3:

                    vis_arr = torch.stack([item.cpu() for i in zip(vis_data, pred_data, target_data.to(torch.float)) for item in i])
                    if len(vis_arr.shape) == 4 and vis_arr.shape[1] != 3:
                        #The vis should convert 3D to 2D images
                        idx = random.randint(1,12)
                        vis_arr = vis_arr[:,:,:,idx].unsqueeze(1)
                    self.writer.add_image('input', make_grid(vis_arr, nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            self.logger.debug(f"Learning Rate:{self.lr_scheduler.get_lr()}")
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, enumerate_result in enumerate(self.valid_data_loader):
                if type(self.data_loader).__name__ == "GeneralDataLoader" or type(self.data_loader).__name__ == "FasterGeneralDataLoader":
                    data, target, misc = enumerate_result
                else:
                    data, target = enumerate_result
                    misc = None
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)

                loss = self.criterion[0](output, target, misc)
                each_loss = []
                each_loss.append(float(loss))
                if len(self.criterion) > 1:
                    for idx in range(1, len(self.criterion)):
                        loss2 = self.criterion[idx](output, target, misc)
                        loss = loss + loss2
                        each_loss.append(float(loss2))

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target, misc))

                if batch_idx % self.log_step == 0:
                    self.logger.debug(
                        f'Validation Epoch: {epoch} {self._progress(batch_idx, mode="val")} Loss: {loss.item():.6f}({each_loss})')
                    # vis_data = data
                    # if data.shape[1] != 3 and data.shape[1] != 1:
                    #     vis_data = data[:, 0]
                    # self.writer.add_image('input', make_grid(vis_data.cpu(), nrow=8, normalize=True))
                    vis_data = data.squeeze().clone()
                    if type(output) is tuple:
                        pred_data = output[0].clone()
                        pred_data = torch.argmax(pred_data, 1).float()
                    else:
                        pred_data = output.clone()
                        pred_data = torch.argmax(pred_data, 1).float()
                    target_data = target.clone()
                    if 'output_processed' in misc:
                        pred_data = misc['output_processed']
                    if len(data.shape) >4 and data.shape[1] != 3 and data.shape[1] != 1:
                        vis_data = data[:, 0]
                    if len(pred_data.shape) > 4 and pred_data.shape[1] != 3 and pred_data.shape[1] != 1:
                        pred_data = pred_data[:, 0]
                    if len(target_data.shape) > 4 and target_data.shape[1] != 3 and target_data.shape[1] != 1:
                        target_data = target_data[:, 0]

                    # classification task should not vis the prediction.
                    if len(pred_data.shape)>3:
                        vis_arr = torch.stack([item.cpu() for i in zip(vis_data, pred_data, target_data.to(torch.float)) for item in i])
                        if len(vis_arr.shape) == 4 and vis_arr.shape[1] != 3:
                            # The vis should convert 3D to 2D images
                            idx = random.randint(1, 12)
                            vis_arr = vis_arr[:, :, :, idx].unsqueeze(1)
                        self.writer.add_image('val_input', make_grid(vis_arr, nrow=8,normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx, mode='train'):
        base = '[{}/{} ({:.0f}%)]'
        if mode == 'train':
            if hasattr(self.data_loader, 'n_samples'):
                current = batch_idx * self.data_loader.batch_size
                total = self.data_loader.n_samples
            else:
                current = batch_idx
                total = self.len_epoch
        else:  # validation
            if hasattr(self.valid_data_loader, 'n_samples'):
                current = batch_idx * self.valid_data_loader.batch_size
                total = self.valid_data_loader.n_samples
            else:
                current = batch_idx
                total = len(self.valid_data_loader)
        return base.format(current, total, 100.0 * current / total)
