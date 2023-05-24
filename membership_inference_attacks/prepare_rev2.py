
import h5py
import numpy as np
from multiprocessing import pool
import os

import torch
from skimage.transform import resize
from skimage.metrics import mean_squared_error, normalized_root_mse
from PIL import Image

np.random.seed(123)


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
        # ss = []
        # mse = []
        nrmse = []
        for k in range(imageA.shape[0]):
            # ss.append(ssim(imageA[k], imageB[k], data_range=imageB[k].max() - imageB[k].min()))
            # mse.append(mean_squared_error(imageA[k], imageB[k]))
            nrmse.append(normalized_root_mse(imageA[k], imageB[k]))
        # ss = np.mean(ss)
        # mse = np.mean(mse)
        nrmse = np.mean(nrmse)
    else:
        # ss = ssim(imageA, imageB)
        # mse = mean_squared_error(imageA, imageB)
        nrmse = normalized_root_mse(imageA, imageB)

    # return [ss, nrmse]
    return [nrmse]


def extract_syn_image_feat():
    train_h5 = '/data/datasets/asdgan_data/asdgan_syn/brats_h5_all/brats_resnet_9blocks_epoch160_Brats4ch3db_modality_nature_augment.h5'
    hfile = h5py.File(train_h5, 'r')
    h5db = hfile['train']
    keys = list(h5db.keys())

    feat_all = []
    for key in keys:
        data = np.array(h5db[key]['data'], dtype='uint8')  # syn data

        data_emb = []
        for k in range(data.shape[0]):
            data_k = data[k]

            data_ch3 = np.tile(data_k, (3, 1, 1))
            data_ch3 = np.moveaxis(data_ch3, 0, -1)
            img = Image.fromarray(data_ch3)

            img_emb = res.getMapping(img)
            img_emb = img_emb.view(-1, 2048)
            img_emb = img_emb.numpy()
            data_emb.append(img_emb[0])

        data_emb = np.array(data_emb)
        print(data_emb.shape)
        feat_all.append(data_emb)

    np.save('syn_image_feat.npy', np.array(feat_all))


def preprocess_resize_syn_images(h5_old, h5_new):

    hfile_syn = h5py.File(h5_old, 'r')
    h5db_syn = hfile_syn['train']
    keys_syn = list(h5db_syn.keys())

    hfile_write = h5py.File(h5_new, 'w')
    for key_id in keys_syn:
        data_fake = np.array(h5db_syn[key_id]['data'], dtype='uint8')  # syn data

        data_fake_new = []
        for k in range(data_fake.shape[0]):
            data_fake_new.append(resize(data_fake[k], (240, 240), order=1, preserve_range=True).astype('uint8'))
        data_fake_new = np.array(data_fake_new)
        hfile_write.create_dataset(f"train/{key_id}/data", data=data_fake_new.astype("uint8"), compression="gzip")
    hfile_write.close()
    hfile_syn.close()


def retrieve_metrics(args):
    keys, save_prefix, h5_real, h5_syn, real_set_name = args

    hfile_syn = h5py.File(h5_syn, 'r')
    h5db_syn = hfile_syn['train']
    keys_syn = list(h5db_syn.keys())

    hfile_real = h5py.File(h5_real, 'r')
    h5db_real = hfile_real[real_set_name]  # 'test' or 'train'

    scores_all = []

    for key in keys:
        data = np.array(h5db_real[key]['data'], dtype='int16')  # real data

        for i in range(data.shape[0]):
            data[i] = mr_normalization(data[i])
            data[i] = data[i] * (255 / (data[i].max() + 1e-8))
        data = data.astype("uint8")

        print('retrieve for ', key)

        score_case = []
        for key_id in keys_syn:
            data_fake = np.array(h5db_syn[key_id]['data'], dtype='uint8')  # syn data

            metrics = compare_images(data, data_fake)
            score_case.append(metrics)

        scores_all.append(score_case)

    np.save(save_prefix+'_score_all.npy', np.array(scores_all))
    np.save(save_prefix + '_score_keys.npy', keys)
    hfile_real.close()
    hfile_syn.close()


def compute_perception_similarity(sub_keys, sub_save_prefix, h5_real, h5_syn, real_set_name):

    hfile_real = h5py.File(h5_real, 'r')
    h5db_real = hfile_real[real_set_name]  # 'test' or 'train'

    syn_feat_all = np.load('syn_image_feat.npy')
    f_cos = nn.CosineSimilarity(dim=2, eps=1e-3)

    for keys, save_prefix in zip(sub_keys, sub_save_prefix):
        scores_all = []
        for key in keys:
            # print(key in keys_syn)
            data = np.array(h5db_real[key]['data'], dtype='int16')  # real data

            data_emb = []
            for i in range(data.shape[0]):
                data[i] = mr_normalization(data[i])
                data[i] = data[i] * (255 / (data[i].max() + 1e-8))

                data_ch3 = np.tile(data[i].astype('uint8'), (3, 1, 1))
                data_ch3 = np.moveaxis(data_ch3, 0, -1)
                img = Image.fromarray(data_ch3)

                img_emb = res.getMapping(img)
                img_emb = img_emb.view(-1, 2048)
                img_emb = img_emb.numpy()
                data_emb.append(img_emb[0])
            data_emb = np.array(data_emb)[None, :, :] # syn data feat 1 x 4 x 2048

            print('retrieve for ', key)

            cos_scores = f_cos(torch.from_numpy(syn_feat_all), torch.from_numpy(data_emb))
            # print(cos_scores.shape)
            score_case = np.mean(cos_scores.numpy(), axis=1)
            # print(score_case.shape)
            scores_all.append(score_case)

        np.save(save_prefix+'_psim_all.npy', np.array(scores_all))
        np.save(save_prefix + '_psim_keys.npy', keys)
    hfile_real.close()


def run_preprocessing2(h5_real, h5_syn, keys, save_prefix, real_set_name):

    assert(len(keys) % 10 == 0)

    save_prefix_subs = [save_prefix+'_%02d'%k for k in range(10)]
    n_key_sub = len(keys) // 10
    keys_subs = []
    for start in range(10):
        keys_subs.append([keys[k] for k in range(start * n_key_sub, start * n_key_sub + n_key_sub)])

    compute_perception_similarity(keys_subs, save_prefix_subs, h5_real, h5_syn, real_set_name)


def run_preprocessing(h5_real, h5_syn, keys, save_prefix, real_set_name):

    assert (len(keys) % 10 == 0)

    save_prefix_subs = [save_prefix + '_%02d' % k for k in range(10)]
    n_key_sub = len(keys) // 10
    keys_subs = []
    for start in range(10):
        keys_subs.append([keys[k] for k in range(start * n_key_sub, start * n_key_sub + n_key_sub)])

    p = pool.Pool(10)
    p.map(retrieve_metrics, zip(keys_subs, save_prefix_subs, [h5_real]*10, [h5_syn]*10, [real_set_name]*10))
    p.close()
    p.join()

train_h5_orig = '../datasets/asdgan_syn/brats_h5_all/brats_resnet_9blocks_epoch160_Brats4ch3db_modality_nature_augment.h5'
train_h5 = '../datasets/asdgan_syn/brats_h5_all/brats_resnet_9blocks_epoch160_Brats4ch3db_modality_nature_augment_resize240.h5'
if not os.path.isfile(train_h5):
    preprocess_resize_syn_images(train_h5_orig, train_h5)

train_set_n = 300
test_set_n = 1000

def extract_trainset_pos_keys(fname='trainset_pos_keys.npy'):
    if not os.path.isfile(fname):
    # use 'Other' center for training, 975 images in 'Other' train set, 393 images in 'Other' test set
        real_h5 = '../datasets/brats//General_format_BraTS18_train_2d_4ch.h5'
        hfile = h5py.File(real_h5, 'r')
        h5db = hfile['train']
        keys = list(h5db.keys())  # 11349 cases
        hfile.close()

        pos_count = 0
        keys_trainset = []

        for key in keys:
            if 'CBICA' in key or 'TCIA' in key:
                continue
            if pos_count == train_set_n:
                break
            # print(key in keys_syn)
            keys_trainset.append(key)
            pos_count += 1

        np.save(fname, keys_trainset)


def extract_trainset_neg_keys(fname='trainset_neg_keys.npy'):
    if not os.path.isfile(fname):
    ## unseen real images
        real_h5_unseen = '../datasets/brats//General_format_BraTS18_test_2d_4ch.h5'
        hfile_unseen = h5py.File(real_h5_unseen, 'r')
        h5db_unseen = hfile_unseen['test']
        keys_unseen = list(h5db_unseen.keys())  # 2730 cases
        hfile_unseen.close()

        neg_count = 0

        keys_unseen_trainset = []
        for key in keys_unseen:
            if 'CBICA' in key or 'TCIA' in key:
                continue
            if neg_count == train_set_n:
                break
            keys_unseen_trainset.append(key)
            neg_count += 1
        np.save(fname, keys_unseen_trainset)


def extract_trainset_pos():
    key_file = 'trainset_pos_keys.npy'
    extract_trainset_pos_keys(key_file)
    # use 'Other' center for training, 975 images in 'Other' train set, 393 images in 'Other' test set
    real_h5 = '../datasets/brats/General_format_BraTS18_train_2d_4ch.h5'

    keys_trainset = np.load(key_file)

    run_preprocessing(real_h5, train_h5, keys_trainset, 'closet_train_4ch_%d_pos'%train_set_n, 'train')
    run_preprocessing2(real_h5, train_h5, keys_trainset, 'closet_train_4ch_%d_pos'%train_set_n, 'train')


def extract_trainset_neg():
    key_file = 'trainset_neg_keys.npy'
    extract_trainset_neg_keys(key_file)
    ## unseen real images
    real_h5_unseen = '../datasets/brats/General_format_BraTS18_test_2d_4ch.h5'

    keys_unseen_trainset = np.load(key_file)

    run_preprocessing(real_h5_unseen, train_h5, keys_unseen_trainset, 'closet_train_4ch_%d_neg'%train_set_n, 'test')
    run_preprocessing2(real_h5_unseen, train_h5, keys_unseen_trainset, 'closet_train_4ch_%d_neg' % train_set_n, 'test')


# test set
def extract_testset_pos_keys(fname='testset_pos_keys.npy'):
    if not os.path.isfile(fname):
        real_h5 = '../datasets/brats/General_format_BraTS18_train_2d_4ch.h5'
        hfile = h5py.File(real_h5, 'r')
        h5db = hfile['train']
        keys = list(h5db.keys())  # 11349 cases
        hfile.close()

        keys_test_pos = []
        for key in keys:
            if 'CBICA' in key or 'TCIA' in key:
                keys_test_pos.append(key)

        keys_test_pos = np.random.choice(keys_test_pos, size=test_set_n, replace=False)
        np.save(fname, keys_test_pos)


def extract_testset_neg_keys(fname='testset_neg_keys.npy'):
    if not os.path.isfile(fname):
        real_h5_unseen = '../datasets/brats/General_format_BraTS18_test_2d_4ch.h5'
        hfile_unseen = h5py.File(real_h5_unseen, 'r')
        h5db_unseen = hfile_unseen['test']
        keys_unseen = list(h5db_unseen.keys())  # 2730 cases total
        hfile_unseen.close()

        keys_test_neg = []
        for key in keys_unseen:
            if 'CBICA' in key or 'TCIA' in key:  # 1165 + 1172
                keys_test_neg.append(key)
        keys_test_neg = np.random.choice(keys_test_neg, size=test_set_n, replace=False)
        np.save(fname, keys_test_neg)


def extract_testset_pos():
    keys_file = 'testset_pos_keys.npy'
    extract_testset_pos_keys(keys_file)
    real_h5 = '../datasets/brats/General_format_BraTS18_train_2d_4ch.h5'

    keys_test_pos = np.load(keys_file)

    run_preprocessing(real_h5, train_h5, keys_test_pos, 'closet_test_4ch_%d_pos'%test_set_n, 'train')
    run_preprocessing2(real_h5, train_h5, keys_test_pos, 'closet_test_4ch_%d_pos' % test_set_n, 'train')


## unseen real images
def extract_testset_neg():
    keys_file = 'testset_neg_keys.npy'
    extract_testset_neg_keys(keys_file)
    real_h5_unseen = '../datasets/brats/General_format_BraTS18_test_2d_4ch.h5'

    keys_test_neg = np.load(keys_file)

    run_preprocessing(real_h5_unseen, train_h5, keys_test_neg, 'closet_test_4ch_%d_neg'%test_set_n, 'test')
    run_preprocessing2(real_h5_unseen, train_h5, keys_test_neg, 'closet_test_4ch_%d_neg' % test_set_n, 'test')


def combine_sub_files():

    # temp: merge train data
    fea_all = []
    lab_all = []
    train_keys = []
    for i in range(10):
        file_pos = 'closet_train_4ch_300_pos_%02d_score_keys.npy' % i
        file_neg = 'closet_train_4ch_300_neg_%02d_score_keys.npy' % i
        keys_pos = np.load(file_pos)
        keys_neg = np.load(file_neg)
        train_keys = train_keys + list(keys_pos) + list(keys_neg)

        file_pos = 'closet_train_4ch_300_pos_%02d_score_all.npy' % i
        file_neg = 'closet_train_4ch_300_neg_%02d_score_all.npy' % i
        feat_pos = np.load(file_pos)
        feat_neg = np.load(file_neg)
        feat_pos[:, :, 0] = 1 - feat_pos[:, :, 0]
        feat_neg[:, :, 0] = 1 - feat_neg[:, :, 0]

        file_pos = 'closet_train_4ch_300_pos_%02d_psim_all.npy' % i
        file_neg = 'closet_train_4ch_300_neg_%02d_psim_all.npy' % i
        feat_pos2 = np.load(file_pos)
        feat_neg2 = np.load(file_neg)
        feat_pos2 = 1 - feat_pos2[:, :, None]
        feat_neg2 = 1 - feat_neg2[:, :, None]

        feat_pos = np.concatenate([feat_pos, feat_pos2], axis=2)
        feat_neg = np.concatenate([feat_neg, feat_neg2], axis=2)

        features = np.concatenate([feat_pos, feat_neg], axis=0)
        label = np.concatenate([np.ones(len(keys_pos)), np.zeros(len(keys_neg))])
        fea_all.append(features)
        lab_all.append(label)
    X = np.concatenate(fea_all, axis=0)
    y = np.concatenate(lab_all)
    print(X.shape, y.shape)
    np.savez('closet_train_4ch_300_feat.npz', X=X, y=y)
    np.save('closet_train_4ch_300_keys.npy', train_keys)

    # temp: combine all test data
    fea_all = []
    lab_all = []
    test_keys = []
    for i in range(10):
        file_pos = 'closet_test_4ch_1000_pos_%02d_score_keys.npy' % i
        file_neg = 'closet_test_4ch_1000_neg_%02d_score_keys.npy' % i
        keys_pos = np.load(file_pos)
        keys_neg = np.load(file_neg)
        test_keys = test_keys + list(keys_pos) + list(keys_neg)

        file_pos = 'closet_test_4ch_1000_pos_%02d_score_all.npy' % i
        file_neg = 'closet_test_4ch_1000_neg_%02d_score_all.npy' % i
        test_feat_pos = np.load(file_pos)
        test_feat_neg = np.load(file_neg)
        test_feat_pos[:, :, 0] = 1 - test_feat_pos[:, :, 0]
        test_feat_neg[:, :, 0] = 1 - test_feat_neg[:, :, 0]

        file_pos = 'closet_test_4ch_1000_pos_%02d_psim_all.npy' % i
        file_neg = 'closet_test_4ch_1000_neg_%02d_psim_all.npy' % i
        test_feat_pos2 = np.load(file_pos)
        test_feat_neg2 = np.load(file_neg)
        test_feat_pos2 = 1 - test_feat_pos2[:, :, None]
        test_feat_neg2 = 1 - test_feat_neg2[:, :, None]

        test_feat_pos = np.concatenate([test_feat_pos, test_feat_pos2], axis=2)
        test_feat_neg = np.concatenate([test_feat_neg, test_feat_neg2], axis=2)

        features = np.concatenate([test_feat_pos, test_feat_neg], axis=0)
        label = np.concatenate([np.ones(len(keys_pos)), np.zeros(len(keys_neg))])
        fea_all.append(features)
        lab_all.append(label)
    X_test = np.concatenate(fea_all, axis=0)
    y_test = np.concatenate(lab_all)
    print(X_test.shape, y_test.shape)
    np.savez('closet_test_4ch_1000_feat.npz', X=X_test, y=y_test)
    np.save('closet_test_4ch_1000_keys.npy', test_keys)



extract_syn_image_feat()

# print('build train set...')
extract_trainset_pos()
extract_trainset_neg()

# print('build test set...')
extract_testset_pos()
extract_testset_neg()

print('combine features')
combine_sub_files()

