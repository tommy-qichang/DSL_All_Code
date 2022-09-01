import matplotlib
matplotlib.use('agg')
import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image


def normalize_data(realdata, syndata):
    new_syndata = []
    for i in range(3):
        realdata_ch = realdata[i]
        syndata_ch = syndata[i]
        mean_ref = realdata_ch.mean()
        std_ref = realdata_ch.std()

        mean_scale = syndata_ch.mean()
        std_scale = syndata_ch.std()

        adjustedimage = mean_ref + (syndata_ch - mean_scale) * std_ref/std_scale
        new_syndata.append(adjustedimage)

    return np.stack(new_syndata, axis=0)



if __name__=='__main__':
    epoch = 200
    root_dir = '/research/cbim/vast/qc58/work/projects/AsynDGANv2/results/brats_AsynDGANv2_3db_exp11_v4/Brats3db_resnet_3ch_9d_v4/test_%d' % epoch
    syn_h5 = os.path.join(root_dir, 'brats_multilabel_perceptionloss100_hgglgg_resnet_9blocks_epoch%d.h5' % epoch)
    brats_syn = h5py.File(syn_h5, 'r')
    syn_db = brats_syn['train']

    # real_h5 = '/data/qichang/AsynDGAN/General_format_BraTS18_train_2d_3ch_new.h5'

    for center in [0, 1, 2]:
        # id_file = 'keys%d.txt' % center
        # key_ids = np.loadtxt(id_file, dtype='str')

        real_h5 = '/research/cbim/medical/medical-share/public/BraTS2018/AsynDGANv2/General_format_BraTS18_train_three_center_%d_2d_3ch.h5' % center

        brats_real = h5py.File(real_h5, 'r')
        real_db = brats_real['train']

        # assert (len(key_ids) == len(list(real_db.keys())))
        key_ids = list(real_db.keys())

        # real_ids = [0, 1, 2]
        # real_ids.remove(center)
        outdir = os.path.join(root_dir, 'plots_%d' % center)
        if not os.path.isdir(outdir):
            os.makedirs(outdir)

        n_rot = 0
        num_r = 3
        num_c = 7
        ctr = 0
        mod_names = ['T1', 'T2', 'Flair']

        for key in key_ids[::2]:
            syndata = np.array(syn_db[f"{key}/data"][()])
            gt = np.array(syn_db[f"{key}/labels_with_skull"][()])[0]
            gt = gt - 1
            values = np.unique(gt)
            maxv = len(values) - 1
            for v in values[::-1]:
                gt[gt == v] = maxv
                maxv -= 1
            # gt[gt==2] = -1
            # gt[gt > 0] = 1
            # gt[gt == -1] = 2
            gt = gt.astype("uint8")
            gt = Image.fromarray(gt)
            label = np.array(gt.resize((240, 240), resample=Image.NEAREST))

            realdata = np.array(real_db[f"{key}/data"][()])
            label1 = np.array(real_db[f"{key}/label"][()])

            syndata = normalize_data(realdata, syndata)


            if np.sum(label1 > 0) < 100:
                continue
            # if center == 0:
            #     data = np.stack((syndata[0], realdata[1], realdata[2]), axis=0)
            # elif center == 1:
            #     data = np.stack((realdata[0], syndata[1], realdata[2]), axis=0)
            # else:
            #     data = np.stack((realdata[0], realdata[1], syndata[2]), axis=0)


            if ctr == 0:
                file_name = []
                plt.figure(figsize=(20, 10))
                showtitle = True

            ctr += 1
            plt.subplot(num_r, num_c, ctr)
            plt.imshow(np.rot90(label, n_rot), cmap="gray")
            if showtitle:
                plt.title("Label")
            plt.axis('off')

            for k in range(3):
                ctr += 1
                plt.subplot(num_r, num_c, ctr)
                plt.imshow(np.rot90(syndata[k], n_rot), cmap="gray")
                if showtitle:
                    plt.title('syn ' + mod_names[k])
                plt.axis('off')

            for k in range(3):
                ctr += 1
                plt.subplot(num_r, num_c, ctr)
                plt.imshow(np.rot90(realdata[k], n_rot), cmap="gray")
                if showtitle:
                    plt.title('real ' + mod_names[k])
                plt.axis('off')

            file_name.append(key)

            if ctr == num_r * num_c:
                for k in range(3):
                    realdata_ch = np.rot90(realdata[k], n_rot)
                    syndata_ch = np.rot90(syndata[k], n_rot)
                    label_ch = np.rot90(label, n_rot)
                    path = os.path.join(outdir, '-'.join(file_name))
                    plt.imsave(path+f"_{k}_real.png", realdata_ch, cmap="gray")
                    plt.imsave(path+f"_{k}_syn.png", syndata_ch, cmap="gray")
                    plt.imsave(path+f"_{k}_label.png", label_ch, cmap="gray")

                # plt.tight_layout()
                # plt.savefig(os.path.join(outdir, '-'.join(file_name)))
                # plt.close()

                ctr = 0
                # break
            else:
                showtitle = False

        brats_real.close()
    brats_syn.close()



def plot_syn_all():

    epoch = 160
    root_dir = '/data/zhennan/pytorch-CycleGAN-and-pix2pix/results/brats_AsynDGANv2_3db_exp43_2_doubleemod_t1c_new/Brats3db_resnet_3ch_doubleemod_t1c_new/test_%d' % epoch

    syn_h5 = os.path.join(root_dir, 'brats_multilabel_perceptionloss100_hgglgg_resnet_9blocks_epoch%d.h5' % epoch)
    brats_syn = h5py.File(syn_h5, 'r')
    syn_db = brats_syn['train']

    # real_h5 = '/data/qichang/AsynDGAN/General_format_BraTS18_train_2d_3ch_new.h5'

    num_r = 3
    num_c = 7
    ctr = 0
    mod_names = ['T1c', 'T2', 'Flair']

    for key in syn_db.keys():
        syndata = np.array(syn_db[f"{key}/data"][()])
        # label = np.array(syn_db[f"{key}/label"][()])

        gt = np.array(syn_db[f"{key}/labels_with_skull"][()])[0]
        gt = gt - 1
        values = np.unique(gt)
        maxv = len(values)-1
        for v in values[::-1]:
            gt[gt==v] = maxv
            maxv -= 1
        # gt[gt==2] = -1
        # gt[gt > 0] = 1
        # gt[gt == -1] = 2
        gt = gt.astype("uint8")
        gt = Image.fromarray(gt)
        label = np.array(gt.resize((240, 240), resample=Image.NEAREST))

        img = np.moveaxis(syn_db[f"{key}/reference_real_image_please_dont_use"][()], 0, -1)
        img = (img + 1) * (255 / 2)
        img = img.astype("uint8")
        realdata = np.array(Image.fromarray(img).resize((240, 240)))
        realdata = np.moveaxis(realdata, -1, 0)

        # import pdb
        # pdb.set_trace()

        n_rot = 0

        if ctr == 0:
            file_name = []
            plt.figure(figsize=(20, 10))
            showtitle = True

        ctr += 1
        plt.subplot(num_r, num_c, ctr)
        plt.imshow(np.rot90(label, n_rot), cmap="gray")
        if showtitle:
            plt.title("Label")
        plt.axis('off')

        for k in range(3):
            ctr += 1
            plt.subplot(num_r, num_c, ctr)
            plt.imshow(np.rot90(syndata[k], n_rot), cmap="gray")
            if showtitle:
                plt.title(mod_names[k])
            plt.axis('off')

        for k in range(3):
            ctr += 1
            plt.subplot(num_r, num_c, ctr)
            plt.imshow(np.rot90(realdata[k], n_rot), cmap="gray")
            if showtitle:
                plt.title(mod_names[k])
            plt.axis('off')

        file_name.append(key)

        if ctr == num_r * num_c:
            plt.tight_layout()
            plt.savefig(os.path.join(root_dir, "images", '-'.join(file_name)))
            plt.close()

            ctr = 0
            # break
        else:
            showtitle = False

    brats_syn.close()

