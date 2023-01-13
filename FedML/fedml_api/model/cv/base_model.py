import os
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod
from . import networks


def get_cpu_state_dict(state_dict):
    return {k: v.cpu() for k, v in state_dict.items()}


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
        self.device = self.gpu_ids[0]  # torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # device: CPU / GPU

        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.optimizers = []
        self.image_paths = []
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

    def setup(self, opt, process_id=-1):
        """Load and print networks; create schedulers

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        if self.isTrain:
            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]

        if not self.isTrain:
            load_suffix = 'iter_%d' % opt.load_iter if opt.load_iter > 0 else opt.epoch
            self.load_networks_from_disk(load_suffix, opt.save_dir, opt.load_filename)
        elif opt.continue_train:
            load_suffix = 'iter_%d' % opt.load_iter if opt.load_iter > 0 else opt.epoch
            if process_id == -1:
                load_filename = 'aggregated_checkpoint_%d.pth.tar' % opt.epoch
            else:
                load_filename = 'client_%d_checkpoint_ep%d.pth.tar' % (process_id, opt.epoch)
            self.load_networks_from_disk(load_suffix, opt.save_dir, load_filename)
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
        # print('learning rate = %.7f' % lr)
        return lr

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

    def save_networks(self, epoch, save_dir):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
            save_dir (str) -- path to the directory
        """
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                module = getattr(net, "module", None)
                if module is not None:
                    save_filename = '%s_net_%s.pth' % (epoch, name)
                    save_path = os.path.join(save_dir, save_filename)
                    if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                        torch.save(get_cpu_state_dict(net.module.state_dict()), save_path)
                        # net.to(self.device)
                    else:
                        torch.save(net.state_dict(), save_path)
                else:
                    for i in range(len(net)):
                        save_filename = '%s_net_%s_%s.pth' % (epoch, name, i)
                        save_path = os.path.join(save_dir, save_filename)
                        if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                            sub_module = getattr(net[i], "module", None)
                            if sub_module is not None:
                                torch.save(get_cpu_state_dict(net[i].module.state_dict()), save_path)
                                # net[i].to(self.device)
                            else:
                                torch.save(get_cpu_state_dict(net[i].state_dict()), save_path)
                                # net[i].to(self.device)
                        else:
                            torch.save(net[i].state_dict(), save_path)

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

    def get_weights(self, model_name=None):
        """return all the networks parameters.
        """
        state_dicts = {}
        model_names = self.model_names if model_name is None else model_name
        for name in model_names:
            if isinstance(name, str):
                net_name = 'net' + name
                net = getattr(self, net_name)
                if isinstance(net, list):
                    state_dict = []
                    for idx, sub_net in enumerate(net):
                        if isinstance(net[idx], torch.nn.DataParallel):
                            state_dict.append(get_cpu_state_dict(net[idx].module.state_dict()))
                        else:
                            state_dict.append(get_cpu_state_dict(net[idx].state_dict()))
                else:
                    if isinstance(net, torch.nn.DataParallel):
                        state_dict = get_cpu_state_dict(net.module.state_dict())
                    else:
                        state_dict = get_cpu_state_dict(net.state_dict())
                state_dicts[net_name] = state_dict
        return state_dicts

    def load_weights(self, state_dicts, model_name=None):
        """Load all the networks from the state_dicts.

        Parameters:
            state_dicts (dict) -- all parameters
            model_name (list of model name) -- if None (default), use the model's default model_names
        """
        model_names = self.model_names if model_name is None else model_name
        for name in model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                if 'net' +name not in state_dicts.keys():
                    print('model net%s not in state_dict' % name)
                    continue
                net_state_dict = state_dicts['net' + name]
                if isinstance(net, list):
                    for idx, sub_net in enumerate(net):

                        if isinstance(sub_net, torch.nn.DataParallel):
                            module = sub_net.module
                        else:
                            module = sub_net

                        state_dict = net_state_dict[idx]
                        if hasattr(state_dict, '_metadata'):
                            del state_dict._metadata

                        module.load_state_dict(state_dict)
                        sub_net.to(self.device)
                else:
                    if isinstance(net, torch.nn.DataParallel):
                        module = net.module
                    else:
                        module = net

                    state_dict = net_state_dict
                    if hasattr(state_dict, '_metadata'):
                        del state_dict._metadata

                    module.load_state_dict(state_dict)
                    net.to(self.device)

    def load_networks_from_disk(self, epoch, save_dir, load_filename=None):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
            save_dir (str) -- path to the directory
            load_filename (str) -- filename of checkpoint
        """
        if load_filename is not None:
            load_path = os.path.join(save_dir, load_filename)
            print('loading the model from %s' % load_path)
            ckpt = torch.load(load_path, map_location=self.device)

        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)

                if load_filename is not None:
                    if 'net' + name not in ckpt["state_dict"].keys():
                        print('net%s model not found in state_dict' % name)
                        continue

                if isinstance(net,list):
                    for idx, sub_net in enumerate(net):

                        if isinstance(sub_net, torch.nn.DataParallel):
                            module = sub_net.module
                        else:
                            module = sub_net

                        if load_filename is None:
                            load_filename = '%s_net_%s_%s.pth' % (epoch, name, str(idx))
                            load_path = os.path.join(save_dir, load_filename)
                            print('loading the model from %s' % load_path)

                            ckpt = torch.load(load_path, map_location=self.device)
                            if 'state_dict' in ckpt.keys():
                                state_dict = ckpt["state_dict"]['net' + name]
                            else:
                                state_dict = ckpt
                        else:
                            state_dict = ckpt["state_dict"]['net' + name][idx]
                        # if hasattr(state_dict, '_metadata'):
                        #     del state_dict._metadata

                        # patch InstanceNorm checkpoints prior to 0.4
                        # for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                        #     self.__patch_instance_norm_state_dict(state_dict, module, key.split('.'))
                        module.load_state_dict(state_dict)
                else:
                    if isinstance(net, torch.nn.DataParallel):
                        module = net.module
                    else:
                        module = net

                    if load_filename is None:
                        load_filename = '%s_net_%s.pth' % (epoch, name)
                        load_path = os.path.join(save_dir, load_filename)
                        print('loading the model from %s' % load_path)

                        ckpt = torch.load(load_path, map_location=self.device)
                        if 'state_dict' in ckpt.keys():
                            state_dict = ckpt["state_dict"]['net' + name]
                        else:
                            state_dict = ckpt
                    else:
                        state_dict = ckpt["state_dict"]['net' + name]
                    # if hasattr(state_dict, '_metadata'):
                    #     del state_dict._metadata

                    # patch InstanceNorm checkpoints prior to 0.4
                    # for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                    #     self.__patch_instance_norm_state_dict(state_dict, module, key.split('.'))
                    module.load_state_dict(state_dict)

    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        # print('---------- Networks initialized -------------')
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
                # print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        # print('-----------------------------------------------')

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
                for param in net.parameters():
                    param.requires_grad = requires_grad
