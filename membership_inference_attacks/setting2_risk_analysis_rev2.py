
import h5py
import numpy as np
import os
from PIL import Image
# from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize
from skimage.metrics import mean_squared_error, normalized_root_mse
from  calculate_similarity_resnet import ResnetSimilarity


res = ResnetSimilarity()

# Dimension of our vector space
dimension = 2048


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
        nrmse = []
        for k in range(imageA.shape[0]):
            nrmse.append(normalized_root_mse(imageA[k], imageB[k]))

        nrmse = np.mean(nrmse)
    else:
        nrmse = normalized_root_mse(imageA, imageB)

    return nrmse


def measure_image_similarity_real_2_synthetic(h5db, h5db_syn, save_metric_file):
    keys = list(h5db.keys())
    keys_syn = list(h5db_syn.keys())

    count = 0

    scores_similarity = []
    # scores_ssim = []
    scores_nrmse = []
    for key in keys_syn:
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

        # cos distance
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

        mse = compare_images(data, data_fake)
        scores_nrmse.append(mse)

    scores_nrmse = np.array(scores_nrmse)
    print('scores_nrmse ', np.mean(scores_nrmse), np.std(scores_nrmse))

    scores_similarity = np.array(scores_similarity)
    # scores_ssim = np.array(scores_ssim)
    print('scores_similarity ', np.mean(scores_similarity), np.std(scores_similarity))
    # print('scores_ssim ', np.mean(scores_ssim), np.std(scores_ssim))
    np.savez(save_metric_file, scores_similarity=scores_similarity, scores_nrmse=scores_nrmse)


def preprocess_resize_syn_images(h5_old, h5_new, dataset='train'):
    hfile_syn = h5py.File(h5_old, 'r')
    h5db_syn = hfile_syn[dataset]
    keys_syn = list(h5db_syn.keys())

    hfile_write = h5py.File(h5_new, 'w')
    for key_id in keys_syn:
        data_fake = np.array(h5db_syn[key_id]['data'], dtype='uint8')  # syn data

        data_fake_new = []
        for k in range(data_fake.shape[0]):
            data_fake_new.append(resize(data_fake[k], (240, 240), order=1, preserve_range=True).astype('uint8'))
        data_fake_new = np.array(data_fake_new)
        hfile_write.create_dataset(f"{dataset}/{key_id}/data", data=data_fake_new.astype("uint8"), compression="gzip")
    hfile_write.close()
    hfile_syn.close()


train_h5_orig = '../datasets/asdgan_syn/brats_h5_all/brats_resnet_9blocks_epoch160_Brats4ch3db_modality_nature.h5'
train_h5 = '../datasets/asdgan_syn/brats_h5_all/brats_resnet_9blocks_epoch160_Brats4ch3db_modality_nature_resize240.h5'
test_h5_orig = '../datasets/asdgan_syn/brats_h5_all/brats_resnet_9blocks_epoch160_Brats4ch3db_modality_nature_testset.h5'
test_h5 = '../datasets/asdgan_syn/brats_h5_all/brats_resnet_9blocks_epoch160_Brats4ch3db_modality_nature_testset_resize240.h5'

## just do once to resize synthetic data
# preprocess_resize_syn_images(train_h5_orig, train_h5, 'train')
# preprocess_resize_syn_images(test_h5_orig, test_h5, 'test')

hfile_syn = h5py.File(train_h5, 'r')
h5db_syn = hfile_syn['train']

real_h5 = '../datasets/brats/General_format_BraTS18_train_2d_4ch.h5'
hfile = h5py.File(real_h5, 'r')
h5db = hfile['train']
keys_seen = list(h5db.keys())
measure_image_similarity_real_2_synthetic(h5db, h5db_syn, 'attack_setting1_trainset_metrics.npz')


hfile_syn = h5py.File(test_h5, 'r')
h5db_syn = hfile_syn['test']

real_h5 = '../datasets/brats/General_format_BraTS18_test_2d_4ch.h5'
hfile = h5py.File(real_h5, 'r')
h5db = hfile['test']
keys_unseen = list(h5db.keys())
measure_image_similarity_real_2_synthetic(h5db, h5db_syn, 'attack_setting1_testset_metrics.npz')

###############################

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, roc_auc_score

from scipy import stats

## metrics in the original trainset and testset for DSL

data_pos = np.load('attack_setting1_trainset_metrics.npz')
data_neg = np.load('attack_setting1_testset_metrics.npz')
psim_pos = data_pos['scores_similarity']
psim_neg = data_neg['scores_similarity']
psim_pos = 1 - psim_pos  # 1-cos
psim_neg = 1 - psim_neg
nrmse_pos = data_pos['scores_nrmse']
nrmse_neg = data_neg['scores_nrmse']

feat_pos = np.stack([nrmse_pos, psim_pos], axis=1)
feat_neg = np.stack([nrmse_neg, psim_neg], axis=1)

## selected keys for training and testing attack classifier
train_keys = np.load('closet_train_4ch_300_keys.npy')
test_keys = np.load('closet_test_4ch_1000_keys.npy')

idx_train_pos = []
idx_train_neg = []

for key in train_keys:
    if key in keys_seen:
        idx = keys_seen.index(key)
        idx_train_pos.append(idx)
    else:
        idx = keys_unseen.index(key)
        idx_train_neg.append(idx)

idx_test_pos = []
idx_test_neg = []
for key in test_keys:
    if key in keys_seen:
        idx = keys_seen.index(key)
        idx_test_pos.append(idx)
    else:
        idx = keys_unseen.index(key)
        idx_test_neg.append(idx)

X1_train = feat_pos[idx_train_pos]
X0_train = feat_neg[idx_train_neg]
X = np.concatenate([X1_train, X0_train], axis=0)
y = np.concatenate([np.ones(len(X1_train)), np.zeros(len(X0_train))])

X1_test = feat_pos[idx_test_pos]
X0_test = feat_neg[idx_test_neg]
X_test = np.concatenate([X1_test, X0_test], axis=0)
y_test = np.concatenate([np.ones(len(X1_test)), np.zeros(len(X0_test))])

log_regression = LogisticRegression()

log_regression.fit(X,y)

y_pred_proba = log_regression.predict_proba(X_test)[::,1]

auc = roc_auc_score(y_test, y_pred_proba)

pred = log_regression.predict(X_test)

cm = confusion_matrix(y_test, pred)
f1 = f1_score(y_test, pred)
acc = accuracy_score(y_test, pred)
recall = cm[1, 1] / (cm[1, 1] + cm[1, 0])
precision = cm[1, 1] / (cm[1, 1] + cm[0, 1])

print('LR auc:',  auc)
print('f1: ', f1)
print('accuracy: ', acc)
print('CM: ', cm)
print(recall, precision)

print()


clf = make_pipeline(StandardScaler(), SVC(random_state=0))
clf.fit(X, y)

pred_prob = clf.decision_function(X_test)
auc = roc_auc_score(y_test, pred_prob)

pred = clf.predict(X_test)

cm = confusion_matrix(y_test, pred)
f1 = f1_score(y_test, pred)
acc = accuracy_score(y_test, pred)
recall = cm[1, 1] / (cm[1, 1] + cm[1, 0])
precision = cm[1, 1] / (cm[1, 1] + cm[0, 1])

print('SVC auc:',  auc)
print('f1: ', f1)
print('accuracy: ', acc)
print('CM: ', cm)
print(recall, precision)

print()
print('liver SVC:')
clf2 = make_pipeline(StandardScaler(), LinearSVC(random_state=0, tol=1e-5))
clf2.fit(X, y)

y_pred_proba = clf2.decision_function(X_test)
auc = roc_auc_score(y_test, y_pred_proba)

pred = clf2.predict(X_test)

cm = confusion_matrix(y_test, pred)
f1 = f1_score(y_test, pred)
acc = accuracy_score(y_test, pred)
recall = cm[1, 1] / (cm[1, 1] + cm[1, 0])
precision = cm[1, 1] / (cm[1, 1] + cm[0, 1])

print('auc: ', auc)
print('f1: ', f1)
print('accuracy: ', acc)
print('CM: ', cm)
print(recall, precision)


# LR auc: 0.5327379999999999
# f1:  0.5497630331753554
# accuracy:  0.525
# CM:  [[470 530]
#  [420 580]]
# 0.58 0.5225225225225225
#
# SVC auc: 0.545216
# f1:  0.44575471698113206
# accuracy:  0.53
# CM:  [[682 318]
#  [622 378]]
# 0.378 0.5431034482758621

# liver SVC:
# auc:  0.554108
# f1:  0.541891229789319
# accuracy:  0.5325
# CM:  [[512 488]
#  [447 553]]
# 0.553 0.5312199807877042

