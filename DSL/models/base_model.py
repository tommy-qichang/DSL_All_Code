import os
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod
from . import networks


class BaseModel(ABC):
    """This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
        -- <set_input>:                     unpack data from dataset and apply preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
        -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the BaseModel class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions

        When creating your custom class, you need to implement your own initialization.
        In this fucntion, you should first call <BaseModel.__init__(self, opt)>
        Then, you need to define four lists:
            -- self.loss_names (str list):          specify the training losses that you want to plot and save.
            -- self.model_names (str list):         specify the images that you want to display and save.
            -- self.visual_names (str list):        define networks used in our training.
            -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
        """
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)  # save all the checkpoints to save_dir
        if opt.preprocess != 'scale_width':  # with [scale_width], input images might have different sizes, which hurts the performance of cudnn.benchmark.
            torch.backends.cudnn.benchmark = True
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.optimizers = []
        self.image_paths = []
        self.memory_gan = opt.memory_gan
        self.metric = 0  # used for learning rate policy 'plateau'

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new model-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): includes the data itself and its metadata information.
        """
        pass

    @abstractmethod
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass

    @abstractmethod
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        pass

    def setup(self, opt):
        """Load and print networks; create schedulers

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        if self.isTrain:
            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]

        if not self.isTrain or opt.continue_train or opt.memory_gan != "":
            load_suffix = 'iter_%d' % opt.load_iter if opt.load_iter > 0 else opt.epoch
            self.load_networks(load_suffix)
            if opt.memory_gan != "" and self.isTrain:
                print(f"*****Memory-GAN: Load Memory GAN Norm and disable gradient for other layers****")
                self.load_networks_norm(opt.continue_train)
        self.print_networks(opt.verbose)

    def eval(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    def test(self):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        with torch.no_grad():
            self.forward()
            self.compute_visuals()

    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        pass

    def get_image_paths(self):
        """ Return image paths that are used to load current data"""
        return self.image_paths

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        for scheduler in self.schedulers:
            if self.opt.lr_policy == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
        return errors_ret

    def save_networks(self, epoch):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                module = getattr(net, "module", None)
                if module is not None:
                    save_filename = '%s_net_%s.pth' % (epoch, name)
                    save_path = os.path.join(self.save_dir, save_filename)
                    if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                        torch.save(net.module.cpu().state_dict(), save_path)
                        net.cuda(self.gpu_ids[0])
                    else:
                        torch.save(net.cpu().state_dict(), save_path)
                else:
                    for i in range(len(net)):
                        save_filename = '%s_net_%s_%s.pth' % (epoch, name, i)
                        save_path = os.path.join(self.save_dir, save_filename)
                        if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                            sub_module = getattr(net[i], "module", None)
                            if sub_module is not None:
                                torch.save(net[i].module.cpu().state_dict(), save_path)
                                net[i].cuda(self.gpu_ids[0])
                            else:
                                torch.save(net[i].cpu().state_dict(), save_path)
                                net[i].cuda(self.gpu_ids[0])
                        else:
                            torch.save(net[i].cpu().state_dict(), save_path)
                #
                # if isinstance(net, list):
                #     for idx, sub_net in enumerate(net):
                #         module = getattr(sub_net, "module", None)
                #         if module is not None:
                #             save_filename = '%s_net_%s_%s.pth' % (epoch, name, str(idx))
                #             save_path = os.path.join(self.save_dir, save_filename)
                #             if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                #                 torch.save(sub_net.module.cpu().state_dict(), save_path)
                #                 sub_net.cuda(self.gpu_ids[0])
                #             else:
                #                 torch.save(sub_net.cpu().state_dict(), save_path)
                #         else:
                #             for i in range(len(sub_net)):
                #                 save_filename = '%s_net_%s_%s.pth' % (epoch, name, i)
                #                 save_path = os.path.join(self.save_dir, save_filename)
                #                 if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                #                     torch.save(sub_net[i].module.cpu().state_dict(), save_path)
                #                     sub_net[i].cuda(self.gpu_ids[0])
                #                 else:
                #                     torch.save(sub_net[i].cpu().state_dict(), save_path)
                # else:
                #     module = getattr(net, "module", None)
                #     if module is not None:
                #         save_filename = '%s_net_%s.pth' % (epoch, name)
                #         save_path = os.path.join(self.save_dir, save_filename)
                #         if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                #             torch.save(net.module.cpu().state_dict(), save_path)
                #             net.cuda(self.gpu_ids[0])
                #         else:
                #             torch.save(net.cpu().state_dict(), save_path)
                #     else:
                #         for i in range(len(net)):
                #             save_filename = '%s_net_%s_%s.pth' % (epoch, name, i)
                #             save_path = os.path.join(self.save_dir, save_filename)
                #             if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                #                 torch.save(net[i].module.cpu().state_dict(), save_path)
                #                 net[i].cuda(self.gpu_ids[0])
                #             else:
                #                 torch.save(net[i].cpu().state_dict(), save_path)

                #
                # if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                #     torch.save(net.module.cpu().state_dict(), save_path)
                #     net.cuda(self.gpu_ids[0])
                # else:
                #     torch.save(net.cpu().state_dict(), save_path)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    def load_networks(self, epoch):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)

                if isinstance(net,list):
                    for idx, sub_net in enumerate(net):

                        if isinstance(net[idx], torch.nn.DataParallel):
                            net[idx] = net[idx].module

                        load_filename = '%s_net_%s_%s.pth' % (epoch, name, str(idx))
                        load_path = os.path.join(self.save_dir, load_filename)
                        print('loading the model from %s' % load_path)

                        state_dict = torch.load(load_path, map_location=self.device)
                        if hasattr(state_dict, '_metadata'):
                            del state_dict._metadata

                        # patch InstanceNorm checkpoints prior to 0.4
                        for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                            self.__patch_instance_norm_state_dict(state_dict, net[idx], key.split('.'))
                        net[idx].load_state_dict(state_dict,strict=False)
                else:
                    if isinstance(net, torch.nn.DataParallel):
                        net = net.module
                    load_filename = '%s_net_%s.pth' % (epoch, name)
                    load_path = os.path.join(self.save_dir, load_filename)
                    print('loading the model from %s' % load_path)
                    # if you are using PyTorch newer than 0.4 (e.g., built from
                    # GitHub source), you can remove str() on self.device

                    state_dict = torch.load(load_path, map_location=str(self.device))
                    if hasattr(state_dict, '_metadata'):
                        del state_dict._metadata

                    # patch InstanceNorm checkpoints prior to 0.4
                    for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                        self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                    net.load_state_dict(state_dict,strict=False)

    def load_networks_norm(self, is_continue_training=False):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                if isinstance(net, list):
                    for idx, sub_net in enumerate(net):
                        if isinstance(net[idx], torch.nn.DataParallel):
                            net[idx] = net[idx].module
                        self.update_one_network(net[idx], is_continue_training)

                else:
                    if isinstance(net, torch.nn.DataParallel):
                        net = net.module
                    self.update_one_network(net, is_continue_training)

    def update_one_network(self, net, is_continue_training):
        stdd = 1.0
        if not is_continue_training:
            print(f"normalize network:{type(net).__name__}")
            dict_all = net.state_dict()
            model_dict = net.state_dict()

            for k, v in dict_all.items():
                idt = k.find('conv_0.weight')
                if idt >= 0:
                    w_mu = v.mean([2, 3], keepdim=True)
                    w_std = v.std([2, 3], keepdim=True) * stdd
                    dict_all[k].data = (v - w_mu) / (w_std)
                    dict_all[k[:idt] + 'AdaFM_0.style_gama'].data = w_std.data
                    dict_all[k[:idt] + 'AdaFM_0.style_beta'].data = w_mu.data
                    if k[:idt] + 'AdaFM_0.style_gama1' in dict_all:
                        dict_all[k[:idt] + 'AdaFM_0.style_gama1'].data = w_std.data
                        dict_all[k[:idt] + 'AdaFM_0.style_beta1'].data = w_mu.data
                        dict_all[k[:idt] + 'AdaFM_0.style_gama2'].data = w_std.data
                        dict_all[k[:idt] + 'AdaFM_0.style_beta2'].data = w_mu.data
                        dict_all[k[:idt] + 'AdaFM_0.style_gama3'].data = w_std.data
                        dict_all[k[:idt] + 'AdaFM_0.style_beta3'].data = w_mu.data
                idt = k.find('conv_1.weight')
                if idt >= 0:
                    w_mu = v.mean([2, 3], keepdim=True)
                    w_std = v.std([2, 3], keepdim=True) * stdd
                    dict_all[k].data = (v - w_mu) / (w_std)
                    dict_all[k[:idt] + 'AdaFM_1.style_gama'].data = w_std.data
                    dict_all[k[:idt] + 'AdaFM_1.style_beta'].data = w_mu.data

            model_dict.update(dict_all)
            net.load_state_dict(model_dict)

        # Only allow AdaFM and fc layers gradient.
        # for name, param in net.named_parameters():
        #     print("*****memory gan, disable other gradient updates*****")
        #     if name.find('AdaFM_') >= 0:
        #         param.requires_grad = True
        #     elif name.find('fc') >= 0:
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = False
        print(f"freeze parameters for {type(net).__name__}, except AdaConv layers, require grad:True")
        self.fix_parameters(net, True)

    def fix_parameters(self, net, requires_grad):
        # Only allow AdaFM and fc layers gradient.
        # print(f"freeze parameters for {type(net).__name__}, except AdaConv layers, require grad:{requires_grad}")
        for name, param in net.named_parameters():
            # print("*****memory gan, disable other gradient updates*****")
            if name.find('AdaFM_') >= 0:
                # print(f"freeze parameters of:{name}")
                param.requires_grad = requires_grad
            elif name.find('fc') >= 0:
                # print(f"freeze parameters of:{name}")
                param.requires_grad = requires_grad
            else:
                param.requires_grad = False


    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                parameters = getattr(net, "parameters", None)
                if parameters is not None:
                    for param in net.parameters():
                        num_params += param.numel()
                else:
                    for i in net:
                        for param in i.parameters():
                            num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                # for param in net.parameters():
                #     param.requires_grad = requires_grad
                if self.memory_gan != "" and self.isTrain:
                    self.fix_parameters(net, requires_grad)
                else:
                    for param in net.parameters():
                        param.requires_grad = requires_grad
