from .utils import transform_tensor_to_list


class AsDGanTrainer(object):
    def __init__(self, client_index, train_data_local_dict, train_data_local_num_dict,
                 test_data_local_dict, device, args, model_trainer):

        self.trainer = model_trainer

        self.client_index = client_index
        self.train_data_local_dict = train_data_local_dict
        self.train_data_local_num_dict = train_data_local_num_dict
        self.test_data_local_dict = test_data_local_dict
        # self.all_train_data_num = train_data_num
        self.train_dataset_local = self.train_data_local_dict[client_index]
        self.local_sample_number = self.train_data_local_num_dict[client_index]
        self.test_local = self.test_data_local_dict[client_index]

        self.device = device
        self.args = args

    def update_dataset(self, client_index):
        self.train_dataset_local = self.train_data_local_dict[client_index]
        self.local_sample_number = self.train_data_local_num_dict[client_index]
        self.test_local = self.test_data_local_dict[client_index]

    def get_model(self):
        return self.trainer.get_model_params()

    def update_lr(self):
        self.trainer.model.update_learning_rate()

    def upload_labels(self):
        keys, labels = self.train_dataset_local.collect_label()
        return keys, labels

    def upload_statistics(self):
        mu, sigma = self.train_dataset_local.get_data_statistics(device=self.device, num_workers=1)
        return mu, sigma

    def train(self, key_samples, fake_samples, trans_paras):
        data_batch, label_batch = self.train_dataset_local.get_data(key_samples, trans_paras)

        train_metrics, grad_fake_B = self.trainer.train_one_iter(label_batch, data_batch, fake_samples)

        return grad_fake_B, train_metrics

    def test(self, round_idx):
        train_evaluation_metrics = test_evaluation_metrics = None

        if (round_idx+1) % self.args.evaluation_frequency == 0:
            train_evaluation_metrics = self.trainer.test_train(self.train_local, self.device)

        if self.test_local:
            test_evaluation_metrics = self.trainer.test(self.test_local, self.device)

        return train_evaluation_metrics, test_evaluation_metrics
