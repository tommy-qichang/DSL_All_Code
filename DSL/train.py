"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import sys
import time
import os
import torch
from shutil import rmtree
import numpy as np
import torchvision
from tqdm import tqdm
from pytorch_fid import fid_score
from data import create_dataset
from models import create_model
from options.train_options import TrainOptions
from util.visualizer import Visualizer


@torch.no_grad()
def calculate_fid(opt, num_batches, dataloader, model, refresh_real=False):
    torch.cuda.empty_cache()
    real_path = os.path.join(opt.checkpoints_dir , opt.name, 'real')
    fake_path = os.path.join(opt.checkpoints_dir , opt.name, 'fake')

    # if not os.path.isdir(real_path) or refresh_real:
    rmtree(real_path, ignore_errors=True)
    os.mkdir(real_path)
    # save fake data
    rmtree(fake_path, ignore_errors=True)
    os.makedirs(fake_path)

    iter_dataloader = iter(dataloader)
    for batch_num in tqdm(range(num_batches), desc="Calculting FID -saving results"):
        real_batch = next(iter_dataloader)
        model.eval()
        model.set_input(real_batch)  # unpack data from data loader
        model.test()  # run inference
        visuals = model.get_current_visuals()  # get image results

        model.save_imgs(real_batch, visuals, opt, batch_num, real_path, fake_path)
        # gt = real_batch['Seg_0']
        # for index in range(gt.shape[0]):
        #     for d_id in range(1,4):
        #         real_img = visuals[f'real_B_{d_id}'][index]
        #         syn_img = visuals[f'fake_B_{d_id}'][index]
        #         filename = f"{batch_num}_{opt.batch_size}_{index}_d{d_id}"
        #         real_img_path = os.path.join(real_path, filename + ".png")
        #         if not os.path.isfile(real_img_path):
        #             torchvision.utils.save_image(real_img, real_img_path)
        #         torchvision.utils.save_image(syn_img, os.path.join(fake_path, filename + ".png"))

    return fid_score.calculate_fid_given_paths([str(real_path), str(fake_path)], 256, 0, 2048)


if __name__ == '__main__':
    opt = TrainOptions().parse()  # get training options
    # print(opt)
    dataloader = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataloader)  # get the max number of images in the datasets.
    print('The max number of training images = %d' % dataset_size)
    # opt.dbsizes = dataloader.dataset.get_center_sizes()
    # print('The number of training images in each center: ', opt.dbsizes)

    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)  # create a visualizer that display/save images and plots
    total_iters = 0  # the total number of training iterations

    for epoch in range(opt.epoch_count,
                       opt.niter + opt.niter_decay + 1):  # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()  # timer for data loading per iteration
        epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch

        for i, data in enumerate(dataloader):  # inner loop within one epoch
            sys.stdout.flush()
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters()  # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:  # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:  # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:  # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        fid = 0
        # if epoch%2 ==0:
        #     #fid10k
        #     # total_calc_img = 30
        #     total_calc_img = 256
        #     fid = calculate_fid(opt, total_calc_img//opt.batch_size ,dataloader, model, refresh_real=True)

        print('End of epoch %d / %d \t Time Taken: %d sec, with Fid10k:%d ' % (
        epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time, fid))
        model.update_learning_rate()  # update learning rates at the end of every epoch.
