import os
import shutil
import glob
import collections
import random
import pickle
import logging
from skimage.transform import resize
from collections import OrderedDict

import numpy as np 
import torch


def transform_list_to_tensor(model_params_list):
    for net in model_params_list.keys():
        for k in model_params_list[net].keys():
            model_params_list[net][k] = torch.from_numpy(np.asarray(model_params_list[net][k])).float()
    return model_params_list


def transform_tensor_to_list(model_params):
    for net in model_params.keys():
        for k in model_params[net].keys():
            model_params[net][k] = model_params[net][k].detach().numpy().tolist()
    return model_params


def save_as_pickle_file(path, data):
    with open(path, "wb") as f:
        pickle.dump(data, f)
        f.close()


def load_from_pickle_file(path):
    return pickle.load(open(path, "rb"))  


def count_parameters(model):
    # params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    paramsG = sum(p.numel() for p in model.netG.parameters() if p.requires_grad)
    if isinstance(model.netD, list):
        paramsD = 0
        for subD in model.netD:
            paramsD += sum(p.numel() for p in subD.parameters() if p.requires_grad)
    else:
        paramsD = sum(p.numel() for p in model.netD.parameters() if p.requires_grad)
    return (paramsG+paramsD) / 1000000


class EvaluationMetricsKeeper:
    def __init__(self, loss_D, loss_G, loss_D_fake, loss_D_real, loss_G_GAN, loss_G_L1, loss_G_perceptual):
        self.loss_D = loss_D
        self.loss_G = loss_G
        self.loss_D_fake = loss_D_fake
        self.loss_D_real = loss_D_real
        self.loss_G_GAN = loss_G_GAN
        self.loss_G_L1 = loss_G_L1
        self.loss_G_perceptual = loss_G_perceptual


# save model checkpoints (centralized)
class Saver(object):

    def __init__(self, args):
        self.args = args
        self.directory = os.path.join('run', args.dataset, args.checkname)
        exp_list = glob.glob(os.path.join(self.directory, 'experiment_*'))
        self.runs = sorted(exp_list, key=lambda exp: int(exp.split('_')[-1]))
        run_id = int(self.runs[-1].split('_')[-1]) + 1 if self.runs else 0

        self.experiment_dir = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)))
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        """Saves checkpoint to disk"""
        filename = os.path.join(self.experiment_dir, filename)
        torch.save(state, filename)
        if is_best:
            best_G = state['best_lossG']
            with open(os.path.join(self.experiment_dir, 'best_G.txt'), 'w') as f:
                f.write(str(best_G))
            # if self.runs:
            #     previous_G = [10000]
            #     for run in self.runs:
            #         run_id = run.split('_')[-1]
            #         path = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)), 'best_G.txt')
            #         if os.path.exists(path):
            #             with open(path, 'r') as f:
            #                 lossG = float(f.readline())
            #                 previous_G.append(lossG)
            #         else:
            #             continue
            #     minG = min(previous_G)
            #     if best_G < minG:
            #         shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth.tar'))
            # else:
            #     shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth.tar'))

    def save_experiment_config(self):
        
        logfile = os.path.join(self.experiment_dir, 'parameters.txt')
        log_file = open(logfile, 'w')

        for opt in vars(self.args):
            log_file.write(opt + ':' + str(getattr(self.args, opt)) + '\n')

        log_file.close()


def float_to_uint_img(img, new_size=None, order=1, minv=None, maxv=None):

    if minv is None:
        minv = img.min()
    if maxv is None:
        maxv = img.max()
    img = (img - minv) * (255 / (maxv - minv))
    img[img > 255] = 255
    img[img < 0] = 0

    if new_size:
        if len(img.shape) == 3:
            img = np.moveaxis(img, 0, -1)
        img = resize(img, new_size, order=order, preserve_range=True)
        if len(img.shape) == 3:
            img = np.moveaxis(img, -1, 0)

    img = np.round(img).astype("uint8")

    return img
