
import h5py
import numpy as np

from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, roc_auc_score


train_set_n = 300

data = np.load('closet_train_4ch_300_feat.npz')
X = data['X']
y = data['y']
train_keys = np.load('closet_train_4ch_300_keys.npy')

print('train on: ', X.shape, y.shape)
print('class cases: 0 ', np.sum(y==0), ' 1 ', np.sum(y==1))


data_test = np.load('closet_test_4ch_1000_feat.npz')
X_test = data_test['X']
y_test = data_test['y']
test_keys = np.load('closet_test_4ch_1000_keys.npy')



def select_closest(X):
    # norm values to [0,1] for each feature in each sample
    max_vals = X.max(1)[:, None, :]
    min_vals = X.min(1)[:, None, :]
    X_norm = np.divide((X - min_vals), (max_vals - min_vals))
    # closest in terms of distance
    dist = np.sum(X_norm, axis=2)
    match = np.argmin(dist, axis=1)
    # extract features
    X = np.array([X[i, match[i]] for i in range(len(X))])

    return X, match


X, match_train = select_closest(X[:, :, 2:])
X_test, match_test = select_closest(X_test[:, :, 2:])


print('SVC:')
clf = make_pipeline(StandardScaler(), SVC(random_state=0))
clf.fit(X, y)

y_pred_proba = clf.decision_function(X_test)
auc = roc_auc_score(y_test, y_pred_proba)

pred = clf.predict(X_test)

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

print('test on trainset:')
pred = clf2.predict(X)

cm = confusion_matrix(y, pred)
f1 = f1_score(y, pred)
acc = accuracy_score(y, pred)
recall = cm[1, 1] / (cm[1, 1] + cm[1, 0])
precision = cm[1, 1] / (cm[1, 1] + cm[0, 1])

y_pred_proba = clf2.decision_function(X)
auc = roc_auc_score(y, y_pred_proba)
print('auc: ', auc)
print('f1: ', f1)
print('accuracy: ', acc)
print('CM: ', cm)
print(recall, precision)


# train on:  (600, 11349, 4) (600,)
# class cases: 0  300  1  300
#
# SVC:
# auc:  0.6242300000000001
# f1:  0.5667198298777246
# accuracy:  0.5925
# CM:  [[652 348]
#  [467 533]]
# 0.533 0.604994324631101
#
# liver SVC:
# auc:  0.64952
# f1:  0.5822222222222222
# accuracy:  0.624
# CM:  [[724 276]
#  [476 524]]
# 0.524 0.655
#
# test on trainset:
# auc:  0.6646333333333334
# f1:  0.6427406199021206
# accuracy:  0.635
# CM:  [[184 116]
#  [103 197]]
# 0.6566666666666666 0.6293929712460063
