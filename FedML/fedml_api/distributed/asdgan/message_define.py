class MyMessage(object):
    """
        message type definition
    """
    # server to client
    MSG_TYPE_S2C_INIT_CONFIG = 1
    MSG_TYPE_S2C_SEND_FAKE_DATA_TO_CLIENT = 2
    MSG_TYPE_S2C_STOP = 3

    # client to server
    MSG_TYPE_C2S_SEND_LABEL_TO_SERVER = 4
    MSG_TYPE_C2S_SEND_GRAD_TO_SERVER = 5
    MSG_TYPE_C2S_SEND_MODEL_TO_SERVER = 6
    MSG_TYPE_C2S_STOP = 7
    MSG_TYPE_C2S_SEND_STATS_TO_SERVER = 8

    MSG_ARG_KEY_TYPE = "msg_type"
    MSG_ARG_KEY_SENDER = "sender"
    MSG_ARG_KEY_RECEIVER = "receiver"

    """
        message payload keywords definition
    """
    MSG_ARG_KEY_NUM_SAMPLES = "num_samples"
    MSG_ARG_KEY_FAKE_SAMPLES = "fake_samples"
    MSG_ARG_KEY_LABEL_SAMPLES = "label_samples"
    MSG_ARG_KEY_KEY_SAMPLES = "key_samples"
    MSG_ARG_KEY_TRANS_SAMPLES = 'trans_paras'
    MSG_ARG_KEY_GRAD_SAMPLES = "grad_fake_samples"
    MSG_ARG_KEY_MODEL_PARAMS = "model_params"
    MSG_ARG_KEY_CLIENT_INDEX = "client_idx"
    MSG_ARG_KEY_EPOCH_INDEX = 'epoch_idx'
    MSG_ARG_KEY_DATA_MU = 'data_mu'
    MSG_ARG_KEY_DATA_SIGMA = 'data_sigma'
    MSG_ARG_KEY_TRAIN_EVALUATION_METRICS = "train_evaluation_metrics"
    MSG_ARG_KEY_TEST_EVALUATION_METRICS = "test_evaluation_metrics"

