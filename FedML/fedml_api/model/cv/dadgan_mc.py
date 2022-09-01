import torch
from torch import nn

from .perception_loss import vgg16_feat, perceptual_loss
from .base_model import BaseModel
from . import networks
# from parse_config import ConfigParser
# import parse_config
import numpy as np


class DadganMCModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.
        multi-channel version: multiple discriminators for different channel (image modalities), each for a channel
    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).
    The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
    By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    def __init__(self, opt, device):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        opt.gpu_ids = [device]
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G', 'D', 'G_GAN', 'G_L1', 'G_perceptual', 'D_real', 'D_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>

        self.D_size = opt.output_nc

        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, up_mode=opt.up_mode)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = []
            for i in range(self.D_size):
                self.netD.append(networks.define_D(opt.input_nc + 1, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids))

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.gpu_ids[0])
            self.criterionL1 = torch.nn.L1Loss()
            # self.criterionSeg = nn.BCEWithLogitsLoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            if opt.client_optimizer == "sgd":
                self.optimizer_G = torch.optim.SGD(self.netG.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay,
                                                   nesterov=opt.nesterov)
            else:
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
            self.optimizers.append(self.optimizer_G)

            self.optimizer_D = []
            for i in range(self.D_size):
                if opt.client_optimizer == 'sgd':
                    opt_D = torch.optim.SGD(self.netD[i].parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay,
                                                       nesterov=opt.nesterov)
                else:
                    opt_D = torch.optim.Adam(self.netD[i].parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
                self.optimizer_D.append(opt_D)
                self.optimizers.append(opt_D)

            self.vgg_model = vgg16_feat().to(self.gpu_ids[0])  #.cuda()
            self.criterion_perceptual = perceptual_loss()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.
        """
        # self.image_paths = []
        self.real_A = input['A'].to(self.gpu_ids[0])
        self.real_B = input['B'].to(self.gpu_ids[0])
        # self.image_paths.append(input['A_paths_' + str(i)])

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)
        return self.fake_B.detach().cpu().numpy()

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        self.loss_D_fake = []
        self.loss_D_real = []

        for i in range(self.D_size):
            fake_AB = torch.cat((self.real_A, self.fake_B[:, i:i+1, ...]), 1)
            real_AB = torch.cat((self.real_A, self.real_B[:, i:i+1, ...]), 1)
            # Fake; stop backprop to the generator by detaching fake_B
            pred_fake = self.netD[i](fake_AB.detach())
            pred_real = self.netD[i](real_AB)
            self.loss_D_fake.append(self.criterionGAN(pred_fake, False))
            self.loss_D_real.append(self.criterionGAN(pred_real, True))

        self.loss_D_fake_all = None
        self.loss_D_real_all = None
        for i in range(self.D_size):
            if self.loss_D_fake_all is None:
                self.loss_D_fake_all = self.loss_D_fake[i]
                self.loss_D_real_all = self.loss_D_real[i]
            else:
                self.loss_D_fake_all += self.loss_D_fake[i]
                self.loss_D_real_all += self.loss_D_real[i]

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake_all + self.loss_D_real_all) * self.opt.lambda_D  # 0.5
        loss_val = self.loss_D.item()
        self.loss_D.backward()

        return loss_val

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        self.loss_G_GAN = None
        self.loss_G_L1 = None

        for i in range(self.D_size):
            fake_AB = torch.cat((self.real_A, self.fake_B[:, i:i+1, ...]), 1)
            pred_fake = self.netD[i](fake_AB)
            pred_feat = self.vgg_model(networks.gray2color(self.fake_B[:, i:i + 1, ...]))
            target_feat = self.vgg_model(networks.gray2color(self.real_B[:, i:i + 1, ...]))
            # First, G(A) should fake the discriminator
            loss_G_GAN = self.criterionGAN(pred_fake, True)
            # Second, G(A) = B
            loss_G_L1 = self.criterionL1(self.fake_B[:, i:i + 1, ...], self.real_B[:, i:i + 1, ...])
            # Third, perceptual loss
            loss_G_perceptual = self.criterion_perceptual(pred_feat, target_feat)
            if self.loss_G_GAN is None:
                self.loss_G_GAN = loss_G_GAN
                self.loss_G_L1 = loss_G_L1
                self.loss_G_perceptual = loss_G_perceptual
            else:
                self.loss_G_GAN += loss_G_GAN
                self.loss_G_L1 += loss_G_L1
                self.loss_G_perceptual += loss_G_perceptual

        self.loss_G = (self.loss_G_GAN + self.loss_G_L1 * self.opt.lambda_L1 + self.loss_G_perceptual * self.opt.lambda_perceptual) * self.opt.lambda_G  #0.1
        loss_val = self.loss_G.item()
        self.loss_G.backward()

        return loss_val

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        # set D's gradients to zero
        for opt in self.optimizer_D:
            opt.zero_grad()
        loss_D = self.backward_D()                # calculate gradients for D
        # update D's weights
        for opt in self.optimizer_D:
            opt.step()
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        loss_G = self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights
        return loss_D, loss_G

    def evaluate(self, input):
        real_A = input['A'].to(self.gpu_ids[0])
        real_B = input['B'].to(self.gpu_ids[0])

        losses_dict = {}
        with torch.no_grad():
            fake_B = self.netG(real_A)  # G(A)
            loss_D_fake = loss_D_real = loss_G_GAN = loss_G_L1 = loss_G_perceptual = []
            for i in range(0, self.D_size):
                fake_AB = torch.cat((real_A, fake_B[:, i:i+1, ...]), 1)
                real_AB = torch.cat((real_A, real_B[:, i:i+1, ...]), 1)
                # Fake; stop backprop to the generator by detaching fake_B
                pred_fake = self.netD[i](fake_AB.detach())
                pred_real = self.netD[i](real_AB)
                pred_feat = self.vgg_model(networks.gray2color(fake_B[:, i:i + 1, ...]))
                target_feat = self.vgg_model(networks.gray2color(real_B[:, i:i + 1, ...]))

                loss_D_fake.append(self.criterionGAN(pred_fake, False).item())
                loss_D_real.append(self.criterionGAN(pred_real, True).item())
                loss_G_GAN.append(self.criterionGAN(pred_fake, True).item())
                loss_G_L1.append(self.criterionL1(fake_B[:, i:i+1, ...], real_B[:, i:i+1, ...]).item())
                loss_G_perceptual.append(self.criterion_perceptual(pred_feat, target_feat).item())

            loss_D_fake = np.sum(loss_D_fake)
            loss_D_real = np.sum(loss_D_real)
            loss_G_GAN = np.sum(loss_G_GAN)
            loss_G_L1 = np.sum(loss_G_L1)
            loss_G_perceptual = np.sum(loss_G_perceptual)

            # D loss
            losses_dict['loss_D'] = (loss_D_fake + loss_D_real) * self.opt.lambda_D
            losses_dict['loss_D_fake'] = loss_D_fake
            losses_dict['loss_D_real'] = loss_D_real

            losses_dict['loss_G'] = (loss_G_GAN + loss_G_L1 * self.opt.lambda_L1 + loss_G_perceptual * self.opt.lambda_perceptual) * self.opt.lambda_G
            losses_dict['loss_G_GAN'] = loss_G_GAN
            losses_dict['loss_G_L1'] = loss_G_L1
            losses_dict['loss_G_perceptual'] = loss_G_perceptual

        return losses_dict
