import logging
import traceback
import time
from fedml_api.distributed.asdgan.message_define import MyMessage
from fedml_core.distributed.communication.message import Message
from fedml_core.distributed.server.server_manager import ServerManager


class AsDGanServerManager(ServerManager):
    def __init__(self, args, aggregator, comm=None, rank=0, size=0, backend="MPI"):
        super().__init__(args, comm, rank, size, backend)
        self.args = args
        self.aggregator = aggregator
        self.round_num = args.epochs
        # self.epoch_idx = 0

        self.batch_sample_idx_dict = None
        logging.info('[Server] Initializing Server Manager')

        self.active_clients = [True] * self.args.client_num_in_total

    def run(self):
        try:
            super().run()
        except Exception:
            traceback.print_exc()
            self.terminate()

    def send_init_msg(self):
        # sampling clients
        client_indexes = self.aggregator.client_sampling(self.args.client_num_in_total,
                                                         self.args.client_num_in_total)

        for client_id in client_indexes:
            self.send_message_init_config(client_id+1, client_id)

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_SEND_LABEL_TO_SERVER,
                                              self.handle_message_receive_label_from_client)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_SEND_GRAD_TO_SERVER,
                                              self.handle_message_receive_grad_from_client)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_STOP,
                                              self.handle_message_stop_from_client)

    def handle_message_receive_grad_from_client(self, msg_params):
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        grad_fake_sample = msg_params.get(MyMessage.MSG_ARG_KEY_GRAD_SAMPLES)
        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        train_eval_metrics = msg_params.get(MyMessage.MSG_ARG_KEY_TRAIN_EVALUATION_METRICS)
        test_eval_metrics = msg_params.get(MyMessage.MSG_ARG_KEY_TEST_EVALUATION_METRICS)

        # logging.info('[Server] Received grad from client {0}'.format(sender_id - 1))

        self.aggregator.add_local_grad(sender_id - 1, grad_fake_sample, model_params)
        self.aggregator.add_client_test_result(sender_id - 1, train_eval_metrics, test_eval_metrics)

        b_all_received = self.aggregator.check_whether_all_receive()

        if b_all_received:
            # logging.info("[Server] b_all_received = " + str(b_all_received))
            self.aggregator.backward_G(self.batch_sample_idx_dict)
            self.aggregator.output_global_acc_and_loss()

            # start the next iteration
            if self.aggregator.finished:
                self.terminate()
                return

            forward_data, self.batch_sample_idx_dict, n_rest_iter = self.aggregator.forward_G()

            for client_id, data in forward_data.items():
                self.send_message_fake_sample_to_client(client_id + 1, data, client_id, n_rest_iter)

    def handle_message_receive_label_from_client(self, msg_params):
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        label_data = msg_params.get(MyMessage.MSG_ARG_KEY_LABEL_SAMPLES)
        key_data = msg_params.get(MyMessage.MSG_ARG_KEY_KEY_SAMPLES)

        logging.info('[Server] Received label data from client {0}'.format(sender_id - 1))

        self.aggregator.add_client_label_data(sender_id - 1, label_data, key_data)

        b_all_label_received = self.aggregator.check_all_label_receive()

        if b_all_label_received:
            logging.info("[Server] b_all_label_received = " + str(b_all_label_received))
            # first training iteration start!
            self.aggregator.init_train_dataloader()

            # send fake_B
            forward_data, self.batch_sample_idx_dict, n_rest_iter = self.aggregator.forward_G()

            for client_id, data in forward_data.items():
                self.send_message_fake_sample_to_client(client_id + 1, data, client_id, n_rest_iter)

    def handle_message_stop_from_client(self, msg_params):
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        logging.info('[Server] Received stop signal from Client {0}'.format(sender_id - 1))
        self.active_clients[sender_id - 1] = False
        for process_id in range(1, self.size):
            if self.active_clients[process_id - 1]:
                self.send_message_stop_client(process_id, process_id - 1)
        time.sleep(0.3)
        self.finish()

    def send_message_init_config(self, receive_id, client_index):
        logging.info('[Server] Initial Configurations sent to client {0}'.format(client_index))
        message = Message(MyMessage.MSG_TYPE_S2C_INIT_CONFIG, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        self.send_message(message)

    def send_message_fake_sample_to_client(self, receive_id, data, client_index, is_next_epoch):
        # logging.info('[Server] send fake sample to client. receive_id {0}'.format(receive_id))
        message = Message(MyMessage.MSG_TYPE_S2C_SEND_FAKE_DATA_TO_CLIENT, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_KEY_SAMPLES, data['key'])
        message.add_params(MyMessage.MSG_ARG_KEY_FAKE_SAMPLES, data['fake'])
        message.add_params(MyMessage.MSG_ARG_KEY_TRANS_SAMPLES, data['trans_para'])
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        message.add_params(MyMessage.MSG_ARG_KEY_EPOCH_INDEX, is_next_epoch)
        self.send_message(message)

    def send_message_stop_client(self, receive_id, client_index):
        logging.info('[Server] send stop message to client. receive_id {0}'.format(receive_id))
        message = Message(MyMessage.MSG_TYPE_S2C_STOP, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        self.send_message(message)

    def terminate(self):
        for process_id in range(1, self.size):
            self.send_message_stop_client(process_id, process_id - 1)
        time.sleep(0.3)
        self.finish()
