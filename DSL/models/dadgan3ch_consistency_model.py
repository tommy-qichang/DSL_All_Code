import torch

from models.perception_loss import vgg16_feat, perceptual_loss
from util.util import grayimg
from . import networks
from .base_model import BaseModel
import numpy as np

class Dadgan3chConsistencyModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--delta_perceptual', type=float, default=1.0, help='weight for perceptual loss')

            parser.add_argument('--lambda_G', type=float, default=0.1, help='weight for dadgan G ')
            parser.add_argument('--lambda_D', type=float, default=0.05, help='weight for dadgan D')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        # self.loss_names = ['G_GAN', 'G_L1', 'G_Seg', 'D_real', 'D_fake']
        self.loss_names = ['G_GAN_all', 'G_L1_all', 'G_perceptual_all', 'D_real_all', 'D_fake_all']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A_2', 'fake_B_2_ch1', 'fake_B_2_ch2', 'fake_B_2_ch3', 'real_B_2_ch1', 'real_B_2_ch2',
                             'real_B_2_ch3',
                             'real_A_7', 'fake_B_7_ch1', 'fake_B_7_ch2', 'fake_B_7_ch3', 'real_B_7_ch1', 'real_B_7_ch2',
                             'real_B_7_ch3',
                             'real_A_11', 'fake_B_11_ch1', 'fake_B_11_ch2', 'fake_B_11_ch3', 'real_B_11_ch1',
                             'real_B_11_ch2', 'real_B_11_ch3']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.D_size = opt.d_size  # 3
        print(f"D size:{opt.d_size}...")
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, opt.memory_gan)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = []
            # for i in range(10):
            for i in range(self.D_size * 3):
                self.netD.append(networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                                   opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain,
                                                   self.gpu_ids, opt.memory_gan, current_memory_gan_id=i % 3))
                # self.netD.append(networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                #                                    opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain,
                #                                    self.gpu_ids, opt.memory_gan))
            # self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
            #                               opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # self.criterionSeg = nn.BCEWithLogitsLoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = []
            for i in self.netD:
                opt_D = torch.optim.Adam(i.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_D.append(opt_D)
                self.optimizers.append(opt_D)
            # self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            # self.optimizers.append(self.optimizer_D)

            self.vgg_model = vgg16_feat().cuda()
            self.criterion_perceptual = perceptual_loss()
            self.criterion_consistency = perceptual_loss()

            # assert(len(opt.dbsizes) == self.D_size)
            # self.weight_center = np.array(opt.dbsizes) / np.sum(self.D_size)
            # self.unet = setup_unet().cuda()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = []
        self.real_B = []
        self.image_paths = []
        # for i in range(10):
        for i in range(self.D_size):
            self.real_A.append(input['A_' + str(i)].to(self.device))
            self.real_B.append(input['B_' + str(i)].to(self.device))
            self.image_paths.append(input['A_paths_' + str(i)])

        self.real_A_2 = self.real_A[0]
        self.real_B_2_ch1 = grayimg(self.real_B[0][:, 0])
        self.real_B_2_ch2 = grayimg(self.real_B[0][:, 1])
        self.real_B_2_ch3 = grayimg(self.real_B[0][:, 2])
        self.real_A_7 = self.real_A[1]
        self.real_B_7_ch1 = grayimg(self.real_B[1][:, 0])
        self.real_B_7_ch2 = grayimg(self.real_B[1][:, 1])
        self.real_B_7_ch3 = grayimg(self.real_B[1][:, 2])
        self.real_A_11 = self.real_A[2]
        self.real_B_11_ch1 = grayimg(self.real_B[2][:, 0])
        self.real_B_11_ch2 = grayimg(self.real_B[2][:, 1])
        self.real_B_11_ch3 = grayimg(self.real_B[2][:, 2])

        # DEBUG: print masks
        # for i in range(len(self.real_A)):
        #     img_A = self.real_A[i].cpu().numpy().astype(np.uint8)
        #     img_B = self.real_B[i].cpu().numpy().astype(np.uint8)
        #     img_A = np.moveaxis(img_A, 1, -1)
        #     img_B = np.moveaxis(img_B, 1, -1)
        #
        #     for j in range(len(img_A)):
        #         img_A_j = (img_A[j] * (255/img_A[j].max())).astype(np.uint8)
        #         img_B_j = (img_B[j] * (255/img_B[j].max())).astype(np.uint8)
        #
        #         img_A_j = Image.fromarray(img_A_j).convert('RGB')
        #         img_B_j = Image.fromarray(img_B_j).convert('RGB')
        #         img_A_j.save(f"/ajax/users/qc58/work/projects/AsynDGANv2/imgs/A_{i}_{j}.jpg")
        #         img_B_j.save(f"/ajax/users/qc58/work/projects/AsynDGANv2/imgs/B_{i}_{j}.jpg")

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = []
        for i in range(len(self.real_A)):
            self.fake_B.append(self.netG(self.real_A[i]))# G(A)
        # self.fake_B = self.netG(self.real_A)  # G(A)
        self.fake_B_2_ch1 = grayimg(self.fake_B[0][:, 0])
        self.fake_B_2_ch2 = grayimg(self.fake_B[0][:, 1])
        self.fake_B_2_ch3 = grayimg(self.fake_B[0][:, 2])
        self.fake_B_7_ch1 = grayimg(self.fake_B[1][:, 0])
        self.fake_B_7_ch2 = grayimg(self.fake_B[1][:, 1])
        self.fake_B_7_ch3 = grayimg(self.fake_B[1][:, 2])
        self.fake_B_11_ch1 = grayimg(self.fake_B[2][:, 0])
        self.fake_B_11_ch2 = grayimg(self.fake_B[2][:, 1])
        self.fake_B_11_ch3 = grayimg(self.fake_B[2][:, 2])

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""

        self.loss_D_fake = []
        self.loss_D_real = []
        for i in range(len(self.real_A)):

            for j in range(3):
                real_A_1 = self.real_A[i][:, j]
                fake_B_1 = self.fake_B[i][:, j]
                real_B_1 = self.real_B[i][:, j]
                fake_AB = torch.cat((torch.stack((real_A_1, real_A_1, real_A_1), 1),
                                     torch.stack((fake_B_1, fake_B_1, fake_B_1), 1)), 1)
                real_AB = torch.cat((torch.stack((real_A_1, real_A_1, real_A_1), 1),
                                     torch.stack((real_B_1, real_B_1, real_B_1), 1)), 1)

                pred_fake = self.netD[i * 3 + j](fake_AB.detach())
                self.loss_D_fake.append(self.criterionGAN(pred_fake, False))
                pred_real = self.netD[i * 3 + j](real_AB)
                self.loss_D_real.append(self.criterionGAN(pred_real, True))

            # Fake; stop backprop to the generator by detaching fake_B
            # fake_AB = torch.cat((self.real_A[i], self.fake_B[i]),1)  # we use conditional GANs; we need to feed both input and output to the discriminator
            # pred_fake = self.netD[i](fake_AB.detach())
            # self.loss_D_fake.append(self.criterionGAN(pred_fake, False))
            # # Real
            # real_AB = torch.cat((self.real_A[i], self.real_B[i]), 1)
            # pred_real = self.netD[i](real_AB)
            # self.loss_D_real.append(self.criterionGAN(pred_real, True))

        self.loss_D_fake_all = None
        self.loss_D_real_all = None
        for i in range(len(self.loss_D_fake)):
            if self.loss_D_fake_all is None:
                self.loss_D_fake_all = self.loss_D_fake[i]
                self.loss_D_real_all = self.loss_D_real[i]
            else:
                self.loss_D_fake_all += self.loss_D_fake[i]
                self.loss_D_real_all += self.loss_D_real[i]

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake_all + self.loss_D_real_all) * self.opt.lambda_D  # 0.05
        self.loss_D.backward()

        # # Fake; stop backprop to the generator by detaching fake_B
        # fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        # pred_fake = self.netD(fake_AB.detach())
        # self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # # Real
        # real_AB = torch.cat((self.real_A, self.real_B), 1)
        # pred_real = self.netD(real_AB)
        # self.loss_D_real = self.criterionGAN(pred_real, True)
        # # combine loss and calculate gradients
        # self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        # self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        self.loss_G_GAN = []
        self.loss_G_L1 = []
        self.loss_G_perceptual = []
        self.loss_G_consistency = []

        # for i in range(10):
        for i in range(self.D_size):
            for j in range(3):
                real_A_1 = self.real_A[i][:, j]
                fake_B_1 = self.fake_B[i][:, j]
                real_B_1 = self.real_B[i][:, j]
                fake_AB = torch.cat((torch.stack((real_A_1, real_A_1, real_A_1), 1),
                                     torch.stack((fake_B_1, fake_B_1, fake_B_1), 1)), 1)
                real_AB = torch.cat((torch.stack((real_A_1, real_A_1, real_A_1), 1),
                                     torch.stack((real_B_1, real_B_1, real_B_1), 1)), 1)

                pred_fake = self.netD[i * 3 + j](fake_AB)
                self.loss_G_GAN.append(self.criterionGAN(pred_fake, True))
                self.loss_G_L1.append(self.criterionL1(fake_B_1, real_B_1) * self.opt.lambda_L1)

                # self.loss_D_fake.append(self.criterionGAN(pred_fake, False))
                # pred_real = self.netD[i * 3 + j](real_AB)
                # self.loss_D_real.append(self.criterionGAN(pred_real, True))

                pred_feat = self.vgg_model(torch.stack((fake_B_1, fake_B_1, fake_B_1), 1))
                target_feat = self.vgg_model(torch.stack((real_B_1, real_B_1, real_B_1), 1))
                self.loss_G_perceptual.append(
                    self.criterion_perceptual(pred_feat, target_feat) * self.opt.delta_perceptual)

            pred_consistency_feat = self.vgg_model(self.fake_B[i])
            target_consistency_feat = self.vgg_model(self.real_B[i])
            self.loss_G_consistency.append(
                self.criterion_consistency(pred_consistency_feat, target_consistency_feat) * self.opt.delta_consistency
            )


            # First, G(A) should fake the discriminator
            # fake_AB = torch.cat((self.real_A[i], self.fake_B[i]), 1)
            # pred_fake = self.netD[i](fake_AB)
            # self.loss_G_GAN.append(self.criterionGAN(pred_fake, True))
            # self.loss_G_L1.append(self.criterionL1(self.fake_B[i], self.real_B[i]) * self.opt.lambda_L1)
            #
            # pred_feat = self.vgg_model(self.fake_B[i])
            # target_feat = self.vgg_model(self.real_B[i])
            # self.loss_G_perceptual.append(self.criterion_perceptual(pred_feat, target_feat) * self.opt.delta_perceptual)

        self.loss_G_GAN_all = None
        self.loss_G_L1_all = None
        self.loss_G_perceptual_all = None
        self.loss_G_consistency_all = None
        for i in range(len(self.loss_G_GAN)):
            if self.loss_G_GAN_all is None:
                self.loss_G_GAN_all = self.loss_G_GAN[i]
                self.loss_G_L1_all = self.loss_G_L1[i]
                self.loss_G_perceptual_all = self.loss_G_perceptual[i]
            else:
                self.loss_G_GAN_all += self.loss_G_GAN[i]
                self.loss_G_L1_all += self.loss_G_L1[i]
                self.loss_G_perceptual_all += self.loss_G_perceptual[i]
        for i in range(self.D_size):
            if self.loss_G_consistency_all is None:
                self.loss_G_consistency_all = self.loss_G_consistency[i]
            else:
                self.loss_G_consistency_all += self.loss_G_consistency[i]

        self.loss_G = (self.loss_G_GAN_all + self.loss_G_L1_all + self.loss_G_perceptual_all + self.loss_G_consistency_all) * self.opt.lambda_G  # 0.1
        self.loss_G.backward()

        # # First, G(A) should fake the discriminator
        # fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        # pred_fake = self.netD(fake_AB)
        # self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # # Second, G(A) = B
        # self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        #
        # pred_feat = self.vgg_model(self.fake_B)
        # target_feat = self.vgg_model(self.real_B)
        # self.loss_G_perceptual = self.criterion_perceptual(pred_feat, target_feat) * self.opt.delta_perceptual

        # pred_seg = self.unet(self.fake_B)
        #
        # pred_seg = torch.sigmoid(pred_seg)
        #
        # self.loss_G_Seg = self.criterionSeg(pred_seg, self.seg)

        # combine loss and calculate gradients
        # self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_Seg
        # self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_perceptual
        # self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()  # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        for opt in self.optimizer_D:
            opt.zero_grad()
        self.backward_D()
        for opt in self.optimizer_D:
            opt.step()
        # self.optimizer_D.zero_grad()     # set D's gradients to zero
        # self.backward_D()                # calculate gradients for D
        # self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate graidents for G
        self.optimizer_G.step()  # udpate G's weights
