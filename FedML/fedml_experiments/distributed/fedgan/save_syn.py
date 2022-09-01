import matplotlib
matplotlib.use('agg')
import os
import sys
import argparse
import h5py
import random
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
import torch
import torch.utils.data as data
import yaml
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))

from fedml_api.data_preprocessing.brats.data_utility import init_transform
# from options.test_options import TestOptions
from fedml_api.model.cv.dadgan import DadganModel
# from util.visualizer import save_images
from fedml_api.distributed.fedgan.utils import float_to_uint_img


def add_args():
    """
    return a parser added with args required by fit
    """
    parser = argparse.ArgumentParser()
    # Test settings
    parser.add_argument('--cfg', type=str, default='default.yml')

    parser.add_argument('--model', type=str, default=None, metavar='N',
                        help='neural network used in training')

    parser.add_argument('--dataset', type=str, default=None, metavar='N',
                        choices=['brats', 'brats_t2', 'brats_t1c', 'brats_flair'],
                        help='dataset used for training')

    parser.add_argument('--data_dir', type=str, default=None,
                        help='data directory')

    parser.add_argument('--checkname', type=str, default=None, help='set the checkpoint name')

    parser.add_argument('--input_nc', type=int, default=None, help='# of input image channels: 3 for RGB and 1 for grayscale')
    parser.add_argument('--output_nc', type=int, default=None, help='# of output image channels: 3 for RGB and 1 for grayscale')
    parser.add_argument('--ngf', type=int, default=None, help='# of gen filters in the last conv layer')
    parser.add_argument('--ndf', type=int, default=None, help='# of discrim filters in the first conv layer')
    parser.add_argument('--gan_mode', type=str, default=None,
                        help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
    parser.add_argument('--netG', type=str, default=None,
                        help='specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]')
    parser.add_argument('--norm', type=str, default=None, help='instance normalization or batch normalization [instance | batch | none]')
    parser.add_argument('--init_type', type=str, default=None, help='network initialization [normal | xavier | kaiming | orthogonal]')
    parser.add_argument('--init_gain', type=float, default=None, help='scaling factor for normal, xavier and orthogonal.')
    parser.add_argument('--no_dropout', default=None, action='store_true', help='no dropout for the generator')
    parser.add_argument('--verbose', default=None, action='store_true', help='if specified, print more debugging information')

    # arguments for test
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--load_filename', type=str, default=None)
    parser.add_argument('--epoch', type=int, default=100, metavar='EP',
                        help='which epoch model will be loaded')
    parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                        help='input batch size for training (default: 20)')
    parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
    parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
    parser.add_argument('--num_test', type=int, default=-1, help='how many test images to run')
    # To avoid cropping, the load_size should be the same as crop_size
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--load_iter', type=int, default=0)
    parser.add_argument('--GPUid', type=str, default='0')
    parser.add_argument('--save_data', default=False, action='store_true', help='save for segmentation training')
    parser.add_argument('--up_mode', default='transpose', type=str)

    args = parser.parse_args()

    args.isTrain = False
    if args.load_filename is None:
        args.load_filename = 'aggregated_checkpoint_{}.pth.tar'.format(args.epoch)

    return args


def cfg_parse_infer(args, cfg):
    args_dict = args.__dict__
    args_key_list = [key for key in args_dict]
    cfg_key_list = [key for key in cfg]

    # init None args with cfg values
    undefined_arg_key = filter(lambda x, args_dict=args_dict: args_dict[x] is None, args_key_list)
    undefined_arg_key = filter(lambda x: x in cfg_key_list, undefined_arg_key)
    for key_name in undefined_arg_key:
        args_dict[key_name] = cfg[key_name]

    # add args which are not included in args parser
    # uncontained_arg_key = filter(lambda x: not (x in args_key_list), cfg_key_list)
    # for key_name in uncontained_arg_key:
    #     args_dict[key_name] = cfg[key_name]

    return args


def create_dataset(args, channel_in, test_bs, sample_rate=0.1):
    if 'brats' in args.dataset:
        from fedml_api.data_preprocessing.brats.data_loader_gan import GeneralDataset
        h5_test = os.path.join(args.data_dir, 'General_format_BraTS18_train_2d_4ch.h5')

        transforms_test = ["Resize", "ToTensorScale", "Normalize"]
        transforms_args_test = {
            "Resize": [args.crop_size],
            "ToTensorScale": ['float', 255, 5],
            "Normalize": [0.5, 0.5]
        }
        transform_test = init_transform(transforms_test, transforms_args_test)

        if 't2' in args.dataset:
            channel = 1
        elif 't1' in args.dataset:
            channel = 0
        elif 'flair' in args.dataset:
            channel = 2
        else:
            channel = None

        test_ds = GeneralDataset(h5_test,
                                 channel=channel,
                                 channel_in=channel_in,
                                 path="train",
                                 sample_rate=sample_rate,
                                 transforms=transform_test)
    elif 'path' in args.dataset:
        from fedml_api.data_preprocessing.exp2_path.data_loader_gan import TestDataset
        h5_test = '/data/datasets/exp2_path/for_seg_256/train_all.h5'

        transforms_test = ["CenterCrop", "ToTensorScale", "Normalize"]
        transforms_args_test = {
            "CenterCrop": [args.crop_size],
            "ToTensorScale": ['float', 255, 255],
            "Normalize": [0.5, 0.5]
        }
        transform_test = init_transform(transforms_test, transforms_args_test)

        test_ds = TestDataset(h5_test,
                              channel_in=channel_in,
                              sample_rate=sample_rate,
                              transforms=transform_test)
    elif 'heart' in args.dataset:
        from fedml_api.data_preprocessing.exp1_heart.data_loader_gan import GeneralDataset
        h5_test = os.path.join(args.data_dir, 'all_train_2d_iso_original_size.h5')  # including miccai2008_train_2d_iso, whs_ct_train_2d_iso, asoca_train_2d_iso
        print(h5_test)
        transforms_test = ["CropPadding", "ToTensorScale", "Normalize"]
        transforms_args_test = {
            "CropPadding": [256],
            "ToTensorScale": ['float', 1.0, 7.0],
            "Normalize": [0.5, 0.5]
        }
        transform_test = init_transform(transforms_test, transforms_args_test)

        test_ds = GeneralDataset(h5_test,
                                 channel_in=channel_in,
                                 path="train",
                                 sample_rate=sample_rate,
                                 transforms=transform_test)

    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=False)
    return test_dl


def save_data(A, B, fake_B, labels_ternary, weight_map, key, file, dataset, newsize=None):

    real_img = B
    syn_img = fake_B
    label = A

    syn_img = float_to_uint_img(syn_img, newsize, 1, -1, 1)
    # real_img = float_to_uint_img(real_img, newsize, 1)
    # label = float_to_uint_img(label, newsize, 0)

    if 'brats' in dataset:
        # restore labels, 0~1 -> 0~5 (skull label 5)
        label = np.round(label * 5)

        label = label.astype("uint8")
        if len(label.shape) == 3:
            label = label[0]
        if newsize:
            label = resize(label, newsize, order=0, preserve_range=True)

        # if np.sum(label>0) < 10:
        # print("skip this seg")
        #     return

        gt = np.copy(label)
        gt[gt==5] = 0  # remove skull label

        save_type = "train"
        file.create_dataset(f"{save_type}/{key}/data", data=syn_img.astype("uint8"))
        file.create_dataset(f"{save_type}/{key}/label", data=gt.astype("uint8"))
        # file.create_dataset(f"{save_type}/{key}/labels_with_skull", data=label)
        # file.create_dataset(f"{save_type}/{key}/reference_real_image_please_dont_use", data=real_img)
    elif 'path' in dataset:
        file.create_dataset(f"images/{key}", data=np.moveaxis(syn_img, 0, -1).astype("uint8"))
        file.create_dataset(f"labels_ternary/{key}", data=np.moveaxis(labels_ternary, 0, -1).astype("uint8"))
        file.create_dataset(f"weight_maps/{key}", data=weight_map)
    elif 'heart' in dataset:
        label = np.round(label * 7).astype("uint8")
        if len(label.shape) == 3:
            label = label[0]
        if newsize:
            label = resize(label, newsize, order=0, preserve_range=True)
        save_type = "train"
        file.create_dataset(f"{save_type}/{key}/data", data=syn_img.astype("uint8"))
        file.create_dataset(f"{save_type}/{key}/label", data=label.astype("uint8"))


def plot_syn_brats(A, B, fake_B, key, save_dir, mod_names):
    nc = B.shape[0]
    num_r = 1
    num_c = 1 + 2 * nc
    ctr = 0

    syn_img = float_to_uint_img(fake_B, (240, 240), 1)

    label = A
    if len(label.shape) == 3:
        label = label[0]
    label = float_to_uint_img(label, (240, 240), 0, 0, 1)

    realdata = float_to_uint_img(B, (240, 240), 1, -1, 1)

    n_rot = 0

    if ctr == 0:
        plt.figure(figsize=(20, 10))
        showtitle = True

    ctr += 1
    plt.subplot(num_r, num_c, ctr)
    plt.imshow(np.rot90(label, n_rot), cmap="gray")
    if showtitle:
        plt.title("Label")
    plt.axis('off')

    for k in range(nc):
        ctr += 1
        plt.subplot(num_r, num_c, ctr)
        plt.imshow(np.rot90(syn_img[k], n_rot), cmap="gray")
        if showtitle:
            plt.title(mod_names[k])
        plt.axis('off')

    for k in range(nc):
        ctr += 1
        plt.subplot(num_r, num_c, ctr)
        plt.imshow(np.rot90(realdata[k], n_rot), cmap="gray")
        if showtitle:
            plt.title(mod_names[k])
        plt.axis('off')

    if ctr == num_r * num_c:
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "images-"+key))
        plt.close()
    else:
        showtitle = False


def plot_syn_path(A, B, fake_B, labels_ternary, weight_map, key, save_dir):
    # nc = B.shape[0]
    num_r = 1
    num_c = 5
    ctr = 0
    # print(fake_B.max(), fake_B.min())

    syn_img = float_to_uint_img(fake_B, None, 1)

    label = A
    if len(label.shape) == 3:
        label = label[0]
    # label = resize(label, (240, 240), order=0, preserve_range=True)
    label = float_to_uint_img(label, None, 0, 0, 1)

    realdata = float_to_uint_img(B, None, 1, -1, 1)

    if ctr == 0:
        plt.figure(figsize=(20, 10))
        showtitle = True

    ctr += 1
    plt.subplot(num_r, num_c, ctr)
    plt.imshow(label)
    if showtitle:
        plt.title("Label")
    plt.axis('off')

    ctr += 1
    plt.subplot(num_r, num_c, ctr)
    plt.imshow(np.moveaxis(syn_img, 0, -1))
    if showtitle:
        plt.title('syn')
    plt.axis('off')

    ctr += 1
    plt.subplot(num_r, num_c, ctr)
    plt.imshow(np.moveaxis(realdata, 0, -1))
    if showtitle:
        plt.title('real')
    plt.axis('off')

    ctr += 1
    plt.subplot(num_r, num_c, ctr)
    plt.imshow(np.moveaxis(labels_ternary, 0, -1))
    if showtitle:
        plt.title('labels_ternary')
    plt.axis('off')

    ctr += 1
    plt.subplot(num_r, num_c, ctr)
    plt.imshow(weight_map)
    if showtitle:
        plt.title('weight_map')
    plt.axis('off')

    if ctr == num_r * num_c:
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "images-"+key))
        plt.close()
    else:
        showtitle = False


def plot_syn_heart(A, B, fake_B, key, save_dir):
    # nc = B.shape[0]
    num_r = 1
    num_c = 3
    ctr = 0
    # print(fake_B.max(), fake_B.min())

    syn_img = float_to_uint_img(fake_B, None, 1, -1, 1)

    label = A
    if len(label.shape) == 3:
        label = label[0]
    # label = resize(label, (240, 240), order=0, preserve_range=True)
    label = float_to_uint_img(label, None, 0, 0, 1)

    realdata = float_to_uint_img(B, None, 1, -1, 1)

    if ctr == 0:
        plt.figure(figsize=(20, 10))
        showtitle = True

    ctr += 1
    plt.subplot(num_r, num_c, ctr)
    plt.imshow(label)
    if showtitle:
        plt.title("Label")
    plt.axis('off')

    ctr += 1
    plt.subplot(num_r, num_c, ctr)
    plt.imshow(syn_img[0], cmap='gray')
    if showtitle:
        plt.title('syn')
    plt.axis('off')

    ctr += 1
    plt.subplot(num_r, num_c, ctr)
    plt.imshow(realdata[0], cmap='gray')
    if showtitle:
        plt.title('real')
    plt.axis('off')

    if ctr == num_r * num_c:
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "images-"+key))
        plt.close()
    else:
        showtitle = False


if __name__ == '__main__':

    args = add_args()
    with open(args.cfg, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    args = cfg_parse_infer(args, cfg)

    print(args)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = True  # fixed input size [256, 256]

    device = torch.device("cuda:{}".format(args.GPUid)) if torch.cuda.is_available() else torch.device("cpu")

    model = DadganModel(args, device)

    # hard-code some parameters for test
    if args.save_data:
        sample_rate = 1.0
    else:
        sample_rate = 0.1
    dataloader = create_dataset(args, args.input_nc, args.batch_size, sample_rate)  # create a dataset given opt.dataset_mode and other options

    model.setup(args)  # regular setup: load and print networks; create schedulers
    # create a website
    web_dir = os.path.join(args.results_dir, args.checkname, '%s_%s' % (args.phase, args.epoch))  # define the website directory

    # model.eval()
    if args.save_dir[-1] == '/':
        args.save_dir = args.save_dir[:-1]
    exp_name = os.path.basename(args.save_dir)
    folder_name = f"{args.dataset}_{args.netG}_epoch{args.epoch}_{exp_name}"
    save_dir = os.path.join(web_dir, folder_name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    syn_round = 1
    if args.save_data:
        file = h5py.File(os.path.join(web_dir, f"{folder_name}.h5"), 'w')

    print(len(dataloader))

    for si in range(syn_round):
        for i, data in enumerate(dataloader):
            if 0 < args.num_test <= i:  # only apply our model to opt.num_test images if args.num_test > 0.
                break
            # import pdb
            # pdb.set_trace()

            model.set_input(data)  # unpack data from data loader
            with torch.no_grad():
                syn_img = model.forward()

            print(syn_img.shape)
            img = data['B'].detach().cpu().numpy()
            A = data['A'].detach().cpu().numpy()
            if 'path' in args.dataset:
                labels_ternary = data['labels_ternary'].detach().cpu().numpy()
                weight_maps = data['weight_maps'].detach().cpu().numpy()

            for j in range(img.shape[0]):
                if args.save_data:
                    if 'brats' in args.dataset:
                        save_data(A[j], img[j], syn_img[j], None, None, data['key'][j], file, args.dataset, (240, 240))
                    elif 'path' in args.dataset:
                        save_data(A[j], img[j], syn_img[j], labels_ternary[j], weight_maps[j], data['key'][j]+'_%i'%si, file, args.dataset)
                    elif 'heart' in args.dataset:
                        save_data(A[j], img[j], syn_img[j], None, None, data['key'][j], file, args.dataset)
                else:
                    if 'brats' in args.dataset:
                        if 't2' in args.dataset:
                            mod_names = ['T2']
                        elif 't1' in args.dataset:
                            mod_names = ['T1c']
                        elif 'flair' in args.dataset:
                            mod_names = ['Flair']
                        else:
                            mod_names = ['T1', 'T2', 'Flair', 'T1c']
                        plot_syn_brats(A[j], img[j], syn_img[j], data['key'][j], save_dir, mod_names)
                    elif 'path' in args.dataset:
                        plot_syn_path(A[j], img[j], syn_img[j], labels_ternary[j], weight_maps[j], data['key'][j]+'_%i'%si, save_dir)
                    elif 'heart' in args.dataset:
                        plot_syn_heart(A[j], img[j], syn_img[j], data['key'][j], save_dir)

            print(f"{i} processed {data['key']}")
            sys.stdout.flush()

    if args.save_data:
        file.close()
