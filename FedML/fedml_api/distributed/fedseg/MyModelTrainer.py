import logging, time
import os
import numpy as np
import torch
import shelve

try:
    from fedml_core.trainer.model_trainer import ModelTrainer
except ImportError:
    from FedML.fedml_core.trainer.model_trainer import ModelTrainer
    
from .utils import SegmentationLosses, Evaluator, LR_Scheduler, EvaluationMetricsKeeper


class MyModelTrainer(ModelTrainer):
    def __init__(self, model, id, args=None):
        super().__init__(model, args)
        self.id = id
        self.node_name = 'Client {}'.format(id) if id > -1 else 'Server'
        self.criterion = SegmentationLosses(reduction=args.loss_reduction).build_loss(mode=args.loss_type)
        self.evaluator = Evaluator(model.n_classes)

        if id > -1:  # only train client trainer
            if self.args.client_optimizer == "sgd":

                if self.args.backbone_freezed:
                    self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.lr * 10,
                                                     momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
                else:
                    if args.model == 'deeplabV3_plus':
                        train_params = [{'params': self.model.get_1x_lr_params(), 'lr': args.lr},
                                        {'params': self.model.get_10x_lr_params(), 'lr': args.lr * 10}]
                    else:
                        train_params = [{'params': self.model.parameters(), 'lr': args.lr}]

                    self.optimizer = torch.optim.SGD(train_params, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
            else:
                self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                                  lr=self.args.lr, weight_decay=self.args.weight_decay, amsgrad=True)

            self.train_data_extracted_features_path = None

    def get_model_params(self):
        if self.args.backbone_freezed:
            logging.info('[{}] retrieve model (Backbone Freezed)'.format(self.node_name))
            return self.model.encoder_decoder.cpu().state_dict()
        else:
            logging.info('[{}] retrieve end-to-end model'.format(self.node_name))
            return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        if self.args.backbone_freezed:
            logging.info('[{}] Updating model (Backbone Freezed)'.format(self.node_name))
            self.model.encoder_decoder.load_state_dict(model_parameters)
        else:
            logging.info('[{}] Updating model'.format(self.node_name))
            self.model.load_state_dict(model_parameters)

    def _extract_features(self, dataset_loader, file_name, device):
        self.model.eval()
        self.model.to(device)

        if self.args.partition_method == "hetero":
            directory = "./extracted_features/" + self.args.dataset + "/hetero/"
            file_path = directory + str(self.id) + '-' + file_name

        else:
            directory = "./extracted_features/" + self.args.dataset + "/homo/"
            file_path = directory + str(self.id) + '-' + file_name

        if not os.path.exists(directory):
            os.makedirs(directory)

        if not os.path.exists(file_path):
            logging.info('Extracting Features')
            features_db = shelve.open(file_path)
            with torch.no_grad():
                for (batch_idx, batch) in enumerate(dataset_loader):
                    x, labels = batch['image'], batch['label']
                    x = x.to(device)
                    extracted_inputs, extracted_features = self.model.feature_extractor(x)
                    features_db[str(batch_idx)] = (extracted_inputs.detach().cpu(), extracted_features.detach().cpu(), labels)
            features_db.close()
        logging.info('[{0}] Returning extracted features database'.format(self.node_name))
        return file_path

    def train(self, train_data, device, args=None):
        if self.id == -1:
            return

        if self.args.backbone_freezed and self.args.extract_feat:
            self._train_extracted_feat(train_data, device)
        else:
            self._train(train_data, device)
        logging.info('[{}] Train done'.format(self.node_name))

    def _train(self, train_data, device):
        model = self.model
        args = self.args

        model.to(device)
        model.train()

        if self.args.backbone_freezed:
            logging.info('[{0}] Training (Backbone Freezed) for {1} Epochs'.format(self.node_name, self.args.epochs))
        else:
            logging.info('[{0}] Training for {1} Epochs'.format(self.node_name, self.args.epochs))

        criterion = self.criterion
        optimizer = self.optimizer
        scheduler = LR_Scheduler(args.lr_scheduler, args.lr, args.epochs, iters_per_epoch=len(train_data), lr_step=args.lr_step)

        epoch_loss = []

        for epoch in range(args.epochs):
            t = time.time()
            batch_loss = []
            logging.info('[{0}] Train Epoch: {1}'.format(self.node_name, epoch))

            for (batch_idx, batch) in enumerate(train_data):
                x, labels = batch['image'], batch['label']
                x, labels = x.to(device), labels.to(device)

                scheduler(optimizer, batch_idx, epoch)
                optimizer.zero_grad()
                log_probs = model(x)

                # for Nuclei seg
                if 'weight_map' in batch:
                    weight_map = batch['weight_map']
                    loss = criterion(log_probs, labels).to(device)
                    loss *= weight_map.to(device)
                    loss = loss.mean()
                else:
                    loss = criterion(log_probs, labels).to(device)

                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
                if (batch_idx % 100 == 0):
                    logging.info('[{0}] Train Iteration: {1}, Loss: {2}, Time Elapsed: {3}'.format(self.node_name, batch_idx, loss, (time.time()-t)/60))

            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                logging.info('[{}] Local Training Epoch: {} \tLoss: {:.6f}'.format(self.node_name, epoch, sum(epoch_loss) / len(epoch_loss)))

    def _train_extracted_feat(self, train_data, device):
        args = self.args
        if self.train_data_extracted_features_path is None and self.args.backbone_freezed and self.args.extract_feat:
            logging.info('[{}] Generating Feature Maps for Training Dataset'.format(self.node_name))
            self.train_data_extracted_features_path = self._extract_features(train_data, 'train_features', device)

        self.model.to(device)
        # change to train mode
        self.model.train()

        logging.info('[{0}} Training (Backbone Freezed) for {1} Epochs; Feature Map already extracted'.format(self.node_name, self.args.epochs))
        epoch_loss = []

        criterion = self.criterion
        optimizer = self.optimizer
        scheduler = LR_Scheduler(args.lr_scheduler, args.lr, args.epochs, iters_per_epoch=len(train_data), lr_step=args.lr_step)

        for epoch in range(self.args.epochs):
            t = time.time()
            batch_loss = []

            logging.info('[{0}] Train Epoch: {1}'.format(self.node_name, epoch))

            with shelve.open(self.train_data_extracted_features_path, 'r') as features_db:
                for batch_idx in features_db.keys():
                    (x, low_level_feat, labels) = features_db[batch_idx]
                    x, low_level_feat, labels = x.to(device), low_level_feat.to(device), labels.to(device)

                    scheduler(optimizer, batch_idx, epoch)
                    optimizer.zero_grad()

                    log_probs = self.model.encoder_decoder(x, low_level_feat)
                    loss = criterion(log_probs, labels).to(device)
                    loss.backward()
                    optimizer.step()
                    batch_loss.append(loss.item())
                    logging.info('[{0}] Train Iteration: {1}, Loss: {2}, Time Elapsed: {3}'.format(self.node_name, batch_idx, loss,
                                                                                                      (time.time() - t) / 60))
                # if (batch_idx % 500 == 0):
                # logging.info('Client Id: {0} Iteration: {1}, Loss: {2}, Time Elapsed: {3}'.format(self.id, batch_idx, loss, (time.time()-t)/60))
            features_db.close()
            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                logging.info('[{}] Local Training Epoch: {} \tLoss: {:.6f}'.format(self.node_name, epoch, sum(epoch_loss) / len(epoch_loss)))

    def test(self, test_data, device, args=None):

        logging.info('[{name}] Testing on Test Dataset'.format(name=self.node_name))
        test_evaluation_metrics = self._infer(test_data, device)
        # logging.info("Testing Complete for client {}".format(self.id))

        return test_evaluation_metrics

    def test_train(self, train_data, device):
        logging.info('[{name}] Testing on Train Dataset'.format(name=self.node_name))
        train_evaluation_metrics = None

        if self.args.backbone_freezed and self.args.extract_feat:
            logging.info('[{}] Evaluating model on Train dataset with extracted feature maps (Backbone Freezed)'.format(self.node_name))
            train_evaluation_metrics = self._infer_extracted_feat(train_data, device)
        else:
            logging.info('[{name}] Evaluating model on Train dataset {backbone}'.format(
                backbone="(Backbone Freezed)" if self.args.backbone_freezed else "",
                name=self.node_name))
            train_evaluation_metrics = self._infer(train_data, device)
        return train_evaluation_metrics

    def _infer(self, test_data, device):
        self.model.eval()
        self.model.to(device)
        t = time.time()
        self.evaluator.reset()
        test_acc = test_acc_class = test_mIoU = test_FWIoU = test_dice = test_loss = test_total = 0.
        criterion = self.criterion

        with torch.no_grad():
            for (batch_idx, batch) in enumerate(test_data):
                x, target = batch['image'], batch['label']
                x, target = x.to(device), target.to(device)

                # for Nuclei seg
                if 'weight_map' in batch:
                    output = self.split_forward(device, x, 224, 80)

                    loss = criterion(output, target).to(device)
                    weight_map = batch['weight_map']
                    loss *= weight_map.to(device)
                    loss = loss.mean()

                else:
                    output = self.model(x)
                    loss = criterion(output, target).to(device)

                test_loss += loss.item()
                test_total += target.size(0)
                pred = output.detach().cpu().numpy()
                target = target.detach().cpu().numpy()
                pred = np.argmax(pred, axis=1)
                self.evaluator.add_batch(target, pred)  # todo: need a specific evaluator for nuclei seg, current dice metric for MoNuSeg is wrong
                if (batch_idx % 100 == 0):
                    logging.info('[{0}] Test Iteration: {1}, Loss: {2}, Time Elapsed: {3}'.format(self.node_name, batch_idx, loss,
                                                                                           (time.time() - t) / 60))

                # time_end_test_per_batch = time.time()
                # logging.info("time per batch = " + str(time_end_test_per_batch - time_start_test_per_batch))
                # logging.info("Client = {0} Batch = {1}".format(self.id, batch_idx)

        # Evaluation Metrics (Averaged over number of samples)
        test_acc = self.evaluator.Pixel_Accuracy()
        test_acc_class = self.evaluator.Pixel_Accuracy_Class()
        test_mIoU = self.evaluator.Mean_Intersection_over_Union()
        test_FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        test_loss = test_loss / test_total

        if 'dice' in self.args.loss_type:
            test_dice = self.evaluator.Dice_over_batch()
            eval_metrics = EvaluationMetricsKeeper(test_acc, test_acc_class, test_mIoU, test_FWIoU, test_loss, test_dice)
        else:
            eval_metrics = EvaluationMetricsKeeper(test_acc, test_acc_class, test_mIoU, test_FWIoU, test_loss)
        return eval_metrics

    def _infer_extracted_feat(self, test_data, device):
        self.model.eval()
        self.model.to(device)
        self.evaluator.reset()

        if test_data == self.train_local:
            test_data_extracted_features = self.train_data_extracted_features_path
        else:
            test_data_extracted_features = self.test_data_extracted_features_path

        test_acc = test_acc_class = test_mIoU = test_FWIoU = test_loss = test_total = 0.
        criterion = self.criterion

        with torch.no_grad():
            with shelve.open(test_data_extracted_features, 'r') as features_db:
                for batch_idx in features_db.keys():
                    (x, low_level_feat, target) = features_db[batch_idx]
                    x, low_level_feat, target = x.to(device), low_level_feat.to(device), target.to(device)
                    output = self.model.encoder_decoder(x, low_level_feat)
                    loss = criterion(output, target).to(device)
                    test_loss += loss.item()
                    test_total += target.size(0)
                    pred = output.detach().cpu().numpy()
                    target = target.detach().cpu().numpy()
                    pred = np.argmax(pred, axis=1)
                    self.evaluator.add_batch(target, pred)
                # logging.info("time per batch = " + str(time_end_test_per_batch - time_start_test_per_batch))
            features_db.close()
        # Evaluation Metrics (Averaged over number of samples)
        test_acc = self.evaluator.Pixel_Accuracy()
        test_acc_class = self.evaluator.Pixel_Accuracy_Class()
        test_mIoU = self.evaluator.Mean_Intersection_over_Union()
        test_FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        test_loss = test_loss / test_total

        eval_metrics = EvaluationMetricsKeeper(test_acc, test_acc_class, test_mIoU, test_FWIoU, test_loss)

        return eval_metrics

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False

    def split_forward(self, device, input, size, overlap, outchannel=3):
        '''
        split the input image for forward process
        '''

        b, c, h0, w0 = input.size()

        # zero pad for border patches
        pad_h = 0
        if h0 - size > 0:
            pad_h = (size - overlap) - (h0 - size) % (size - overlap)
            tmp = torch.zeros((b, c, pad_h, w0)).to(device)
            input = torch.cat((input, tmp), dim=2)

        if w0 - size > 0:
            pad_w = (size - overlap) - (w0 - size) % (size - overlap)
            tmp = torch.zeros((b, c, h0 + pad_h, pad_w)).to(device)
            input = torch.cat((input, tmp), dim=3)

        _, c, h, w = input.size()

        output = torch.zeros((input.size(0), outchannel, h, w)).to(device)
        for i in range(0, h - overlap, size - overlap):
            r_end = i + size if i + size < h else h
            ind1_s = i + overlap // 2 if i > 0 else 0
            ind1_e = i + size - overlap // 2 if i + size < h else h
            for j in range(0, w - overlap, size - overlap):
                c_end = j + size if j + size < w else w

                input_patch = input[:, :, i:r_end, j:c_end]
                input_var = input_patch
                with torch.no_grad():
                    output_patch = self.model(input_var)

                ind2_s = j + overlap // 2 if j > 0 else 0
                ind2_e = j + size - overlap // 2 if j + size < w else w
                output[:, :, ind1_s:ind1_e, ind2_s:ind2_e] = output_patch[:, :, ind1_s - i:ind1_e - i,
                                                             ind2_s - j:ind2_e - j]

        output = output[:, :, :h0, :w0]

        return output
