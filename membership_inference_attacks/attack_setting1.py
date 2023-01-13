
import h5py
import numpy as np
import os
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize

from  calculate_similarity_resnet import ResnetSimilarity

def mr_normalization(img, bd_low=0.1, bd_up=99.9, mean_i=None, std_i=None):
    # exclude some outlier intensity if necessary
    # print('norm: ', np.min(img), np.max(img), bd_low, np.percentile(img, bd_low), bd_up, np.percentile(img, bd_up), np.mean(img), np.std(img))
    img[img > np.percentile(img, bd_up)] = np.percentile(img, bd_up)
    img[img < np.percentile(img, bd_low)] = np.percentile(img, bd_low)

    if mean_i is not None and std_i is not None:
        factor_shift = np.mean(img) - mean_i
        img = img - factor_shift
        factor_scale = np.std(img) / std_i
        img = img / factor_scale

    return img


def compare_images(imageA, imageB):
    # compute the mean squared error and structural similarity
    # index for the images
    if len(imageA.shape) > 2:
        s = []
        for k in range(imageA.shape[0]):
            s.append(ssim(imageA[k], imageB[k]))
        s = np.mean(s)
    else:
        s = ssim(imageA, imageB)
    return s


res = ResnetSimilarity()

# Dimension of our vector space
dimension = 2048


def measure_image_similarity_real_2_synthetic(h5db, h5db_syn, save_metric_file):
    keys = list(h5db.keys())
    keys_syn = list(h5db_syn.keys())

    count = 0

    scores_similarity = []
    scores_ssim = []

    for key in keys_syn:
    # for key in keys:
        # print(key in keys_syn)
        data = np.array(h5db[key]['data'], dtype='int16')  # real data

        for i in range(data.shape[0]):
            data[i] = mr_normalization(data[i])
            data[i] = data[i] * (255 / (data[i].max() + 1e-8))
        data = data.astype("uint8")

        data_fake = np.array(h5db_syn[key]['data'], dtype='uint8')  # syn data

        if data[0].shape != data_fake[0].shape:
            data_fake_new = []
            for k in range(data.shape[0]):
                data_fake_new.append(resize(data_fake[k], data[0].shape, order=1, preserve_range=True).astype('uint8'))
            data_fake = np.array(data_fake_new)

        value_ch = []
        for k in range(data.shape[0]):
            data_k = data[k]
            data_ch3 = np.tile(data_k, (3,1,1))
            data_ch3 = np.moveaxis(data_ch3, 0, -1)
            img = Image.fromarray(data_ch3)

            data_fake_k = data_fake[k]
            data_ch3 = np.tile(data_fake_k, (3,1,1))
            data_ch3 = np.moveaxis(data_ch3, 0, -1)
            img2 = Image.fromarray(data_ch3)

            value = res.resnetSimmilarity_C(img, img2)
            value_ch.append(value)

        scores_similarity.append(np.mean(value_ch))

        s = compare_images(data, data_fake)
        scores_ssim.append(s)

    scores_similarity = np.array(scores_similarity)
    # scores_mse = np.array(scores_mse)
    scores_ssim = np.array(scores_ssim)

    print('scores_similarity ', np.mean(scores_similarity), np.std(scores_similarity))
    print('scores_ssim ', np.mean(scores_ssim), np.std(scores_ssim))

    np.savez(save_metric_file, scores_similarity=scores_similarity, scores_ssim=scores_ssim)


train_h5 = '../datasets/asdgan_syn/brats_h5_all/brats_resnet_9blocks_epoch160_Brats4ch3db_modality_nature.h5'
hfile_syn = h5py.File(train_h5, 'r')
h5db_syn = hfile_syn['train']

real_h5 = '../datasets/brats/General_format_BraTS18_train_2d_4ch.h5'
hfile = h5py.File(real_h5, 'r')
h5db = hfile['train']

measure_image_similarity_real_2_synthetic(h5db, h5db_syn, 'attack_setting1_trainset_metrics.npz')

test_h5 = '../datasets/asdgan_syn/brats_h5_all/brats_resnet_9blocks_epoch160_Brats4ch3db_modality_nature_testset.h5'
hfile_syn = h5py.File(test_h5, 'r')
h5db_syn = hfile_syn['test']

real_h5 = '../datasets/brats/General_format_BraTS18_test_2d_4ch.h5'
hfile = h5py.File(real_h5, 'r')
h5db = hfile['test']

measure_image_similarity_real_2_synthetic(h5db, h5db_syn, 'attack_setting1_testset_metrics.npz')

