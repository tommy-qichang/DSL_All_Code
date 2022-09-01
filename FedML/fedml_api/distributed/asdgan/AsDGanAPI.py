import logging

from mpi4py import MPI

from .AsDGanAggregator import AsDGanAggregator
from .AsDGanTrainer import AsDGanTrainer
from .AsDGanClientManager import AsDGanClientManager
from .AsDGanServerManager import AsDGanServerManager
from .DModelTrainer import DModelTrainer
from .GModelTrainer import GModelTrainer


def FedML_init():
    comm = MPI.COMM_WORLD
    process_id = comm.Get_rank()
    worker_number = comm.Get_size()
    return comm, process_id, worker_number


def FedML_AsDGan_distributed(process_id, worker_number, device, comm, model, train_data_global, test_data_global,
                             train_data_local_num_dict, train_data_local_dict, test_data_local_dict, args, model_trainer=None):

    if process_id == 0:
        init_server(args, device, comm, process_id, worker_number, model, model_trainer, train_data_global, test_data_global)
    else:
        init_client(args, device, comm, process_id, worker_number, model, train_data_local_num_dict,
                    train_data_local_dict, test_data_local_dict, model_trainer)


def init_server(args, device, comm, rank, size, model, model_trainer, train_data_global, test_data_global):
    logging.info('Initializing Server')

    if model_trainer is None:
        model_trainer = GModelTrainer(model, -1, args)

    # aggregator
    worker_num = size - 1
    aggregator = AsDGanAggregator(worker_num, device, model, args, model_trainer, train_data_global, test_data_global)

    # start the distributed training
    server_manager = AsDGanServerManager(args, aggregator, comm, rank, size)
    server_manager.send_init_msg()
    server_manager.run()


def init_client(args, device, comm, process_id, size, model, train_data_local_num_dict,
                train_data_local_dict, test_data_local_dict, model_trainer):
    
    client_index = process_id - 1
    logging.info('Initializing Client: {0}'.format(client_index))

    if model_trainer is None:
        model_trainer = DModelTrainer(model, client_index, args)

    # trainer
    trainer = AsDGanTrainer(client_index, train_data_local_dict, train_data_local_num_dict, test_data_local_dict, device,
                            args, model_trainer)
    client_manager = AsDGanClientManager(args, trainer, comm, process_id, size)
    client_manager.run()
