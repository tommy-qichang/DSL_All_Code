from  calculate_similarity_resnet import ResnetSimilarity

from nearpy import Engine
from nearpy.hashes import RandomBinaryProjections
from nearpy.distances import CosineDistance
# from nearpy.distances import EuclideanDistance

# import sys
import h5py
import numpy as np
import pickle
import os
from scipy import ndimage
from PIL import Image
from nearpy.storage import MemoryStorage
from skimage.transform import resize

res = ResnetSimilarity()

# Dimension of our vector space
dimension = 2048

# Create a random binary hash with 10 bits
rbp = RandomBinaryProjections('rbp', 10)

msote = MemoryStorage()

# engine = Engine(dimension, lshashes=[rbp],storage=msote,distance=EuclideanDistance())
engine = Engine(dimension * 4, lshashes=[rbp],storage=msote,distance=CosineDistance())

## Adding the Images to the Hash Table

train_h5 = '../datasets/asdgan_syn/brats_h5_all/brats_resnet_9blocks_epoch160_Brats4ch3db_modality_nature_augment.h5'
hfile = h5py.File(train_h5, 'r')
h5db = hfile['train']
keys = list(h5db.keys())
count = 0

for key in keys:
    data = np.array(h5db[key]['data'], dtype='uint8')  # syn data

    # print(data.shape)

    data_emb = []
    for k in range(data.shape[0]):
        data_k = data[k]
        if data_k.shape != (240, 240):
            data_k = resize(data_k, (240, 240), order=1, preserve_range=True).astype('uint8')


        data_ch3 = np.tile(data_k,(3,1,1))
        data_ch3 = np.moveaxis(data_ch3, 0, -1)
        img = Image.fromarray(data_ch3)

        img_emb = res.getMapping(img)
        img_emb = img_emb.view(-1, 2048)
        img_emb = img_emb.numpy()
        data_emb.append(img_emb[0])

    engine.store_vector(np.array(data_emb).flatten(), key)
    # engine.store_vector(img_emb[0], key+'_ch'+str(k))

filehandler = open("hashed_object_Cos_4ch_aug.pkl", 'wb')
pickle.dump(engine, filehandler)
