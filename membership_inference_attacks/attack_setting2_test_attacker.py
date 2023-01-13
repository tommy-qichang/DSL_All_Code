from  calculate_similarity_resnet import ResnetSimilarity

# import sys
import h5py
import numpy as np
import pickle
import os
from PIL import Image
from skimage.transform import resize
from attack_setting1 import mr_normalization, compare_images


res = ResnetSimilarity()

# Dimension of our vector space
dimension = 2048


train_h5 = '../datasets/asdgan_syn/brats_h5_all/brats_resnet_9blocks_epoch160_Brats4ch3db_modality_nature_augment.h5'
hfile_syn = h5py.File(train_h5, 'r')
h5db_syn = hfile_syn['train']
# keys_syn = list(h5db_syn.keys())

real_h5 = '../datasets/brats/General_format_BraTS18_train_2d_4ch.h5'
hfile = h5py.File(real_h5, 'r')
h5db = hfile['train']
keys = list(h5db.keys())
count = 0

# k = 0

KNN = 1

filehandler = open("hashed_object_Cos_4ch_aug.pkl", 'rb')

en_loaded = pickle.load(filehandler)

keys_test = []
for key in keys:
    if 'CBICA' in key or 'TCIA' in key:
        keys_test.append(key)

keys_test = np.random.choice(keys_test, size=1755, replace=False)

found = []
scores = []
no_match = []
score_ssim = []

for key in keys_test:
    # print(key in keys_syn)
    data = np.array(h5db[key]['data'], dtype='int16')  # real data

    for i in range(data.shape[0]):
        data[i] = mr_normalization(data[i])
        data[i] = data[i] * (255 / (data[i].max() + 1e-8))
    data = data.astype("uint8")

    # print(data.shape)
    query = []
    for k in range(data.shape[0]):
        data_ch3 = np.repeat(data[k:k+1], 3, 0)
        data_ch3 = np.moveaxis(data_ch3, 0, -1)
        img = Image.fromarray(data_ch3)

        img_emb = res.getMapping(img)
        img_emb = img_emb.view(-1, 2048)
        img_emb = img_emb.numpy()

        query.append(img_emb[0])
    query = np.array(query).flatten()

    N = en_loaded.neighbours(query)
    candidates = []
    print('retrieve for ', key)
    if len(N) == 0:
        no_match.append(key)
        # similarity = 100
        # found.append(0)
    else:
        similarity = round(float(N[0][2]), 5)  # distance, smaller more similar
        key_id = N[0][1]
        # print(key_id, similarity)
        if key.split('-')[0] == key_id.split("-")[0]:
            found.append(1)
        else:
            found.append(0)
        scores.append(similarity)

        data_fake = np.array(h5db_syn[key_id]['data'], dtype='uint8')  # syn data

        if data[0].shape != data_fake[0].shape:
            data_fake_new = []
            for k in range(data.shape[0]):
                data_fake_new.append(resize(data_fake[k], data[0].shape, order=1, preserve_range=True).astype('uint8'))
            data_fake = np.array(data_fake_new)

        s = compare_images(data, data_fake)
        score_ssim.append(s)

    # print(np.sum(found) / len(found))

found = np.array(found)
scores = np.array(scores)
score_ssim = np.array(score_ssim)

print('all (cos dist)', np.mean(scores), np.std(scores))
print('(ssim) ', np.mean(score_ssim), np.std(score_ssim))

feature_1 = np.array([scores, score_ssim]).T
label_1 = np.ones(len(scores))
# print('found ', np.mean(scores[found==1]), np.std(scores[found==1]))
# print('miss ', np.mean(scores[found==0]), np.std(scores[found==0]))


real_h5 = '../datasets/brats/General_format_BraTS18_test_2d_4ch.h5'
hfile = h5py.File(real_h5, 'r')
h5db = hfile['test']
keys = list(h5db.keys())

found = []
scores = []
no_match = []
score_ssim = []

for key in keys[975:]:
    # print(key in keys_syn)
    data = np.array(h5db[key]['data'], dtype='int16')  # real data

    for i in range(data.shape[0]):
        data[i] = mr_normalization(data[i])
        data[i] = data[i] * (255 / (data[i].max() + 1e-8))
    data = data.astype("uint8")

    # print(data.shape)
    query = []
    for k in range(data.shape[0]):
        data_ch3 = np.repeat(data[k:k+1], 3, 0)
        data_ch3 = np.moveaxis(data_ch3, 0, -1)
        img = Image.fromarray(data_ch3)

        img_emb = res.getMapping(img)
        img_emb = img_emb.view(-1, 2048)
        img_emb = img_emb.numpy()

        query.append(img_emb[0])
    query = np.array(query).flatten()

    N = en_loaded.neighbours(query)
    candidates = []
    print('retrieve for ', key)
    if len(N) == 0:
        no_match.append(key)
        # similarity = 100
        # found.append(0)
    else:
        similarity = round(float(N[0][2]), 5)  # distance, smaller more similar
        key_id = N[0][1]
        # print(key_id, similarity)
        if key.split('-')[0] == key_id.split("-")[0]:
            found.append(1)
        else:
            found.append(0)
        scores.append(similarity)

        data_fake = np.array(h5db_syn[key_id]['data'], dtype='uint8')  # syn data

        if data[0].shape != data_fake[0].shape:
            data_fake_new = []
            for k in range(data.shape[0]):
                data_fake_new.append(resize(data_fake[k], data[0].shape, order=1, preserve_range=True).astype('uint8'))
            data_fake = np.array(data_fake_new)

        s = compare_images(data, data_fake)
        score_ssim.append(s)

    # print(np.sum(found) / len(found))

scores = np.array(scores)
score_ssim = np.array(score_ssim)

feature_0 = np.array([scores, score_ssim]).T
label_0 = np.zeros(len(scores))

features = np.concatenate([feature_1, feature_0], axis=0)
label = np.concatenate([label_1, label_0])

np.savez('findLSH_testing_4ch_data.npz', X=features, y=label)

data = np.load('findLSH_training_4ch_data.npz')
X = data['X']
y = data['y']

from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
clf = make_pipeline(StandardScaler(), SVC(random_state=0))
clf.fit(X, y)

pred = clf.predict(features)

cm = confusion_matrix(label, pred)
f1 = f1_score(label, pred)
acc = accuracy_score(label, pred)

print('f1: ', f1)
print('accuracy: ', acc)
print('CM: ', cm)


clf2 = make_pipeline(StandardScaler(), LinearSVC(random_state=0, tol=1e-5))
clf2.fit(X, y)

pred = clf2.predict(features)

cm = confusion_matrix(label, pred)
f1 = f1_score(label, pred)
acc = accuracy_score(label, pred)

print('f1: ', f1)
print('accuracy: ', acc)
print('CM: ', cm)