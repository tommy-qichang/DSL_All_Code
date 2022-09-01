
import h5py
import numpy as np


train_all = "/data/datasets/asdgan_data/fedgan_syn/brats_h5_all/brats_resnet_9blocks_epoch160_experiment_0.h5"
train_h5 = "/data/datasets/asdgan_data/fedgan_seg_data_brats/brats_resnet_9blocks_epoch160_experiment_0_trainset.h5"
val_h5 = "/data/datasets/asdgan_data/fedgan_seg_data_brats/brats_resnet_9blocks_epoch160_experiment_0_valset.h5"

val_ids_other = {'Brats18_2013_11_1', 'Brats18_2013_18_1', 'Brats18_2013_5_1'}

val_ids_cbica = {'Brats18_CBICA_AWH_1', 'Brats18_CBICA_ATD_1', 'Brats18_CBICA_AQT_1', 'Brats18_CBICA_ATX_1', 'Brats18_CBICA_BHB_1', 'Brats18_CBICA_AYW_1', 'Brats18_CBICA_APR_1', 'Brats18_CBICA_AAP_1', 'Brats18_CBICA_AQR_1', 'Brats18_CBICA_AQV_1', 'Brats18_CBICA_ASW_1', 'Brats18_CBICA_AXN_1', 'Brats18_CBICA_ANP_1', 'Brats18_CBICA_APZ_1'}

val_ids_tcia = {'Brats18_TCIA04_149_1', 'Brats18_TCIA02_226_1', 'Brats18_TCIA03_265_1', 'Brats18_TCIA08_205_1', 'Brats18_TCIA08_218_1', 'Brats18_TCIA04_437_1', 'Brats18_TCIA06_332_1', 'Brats18_TCIA01_231_1', 'Brats18_TCIA04_328_1', 'Brats18_TCIA04_343_1', 'Brats18_TCIA06_603_1', 'Brats18_TCIA01_186_1', 'Brats18_TCIA05_396_1', 'Brats18_TCIA02_117_1', 'Brats18_TCIA02_331_1', 'Brats18_TCIA03_338_1', 'Brats18_TCIA08_167_1'}

val_ids = list(val_ids_other) + list(val_ids_cbica) + list(val_ids_tcia)


hfile = h5py.File(train_all, 'r')
db = hfile['train']
keys = list(db.keys())

trainfile = h5py.File(train_h5, 'w')
testfile = h5py.File(val_h5, 'w')

for key in keys:
    data = np.array(db[key]['data'], dtype='uint8')  # syn data
    # data = np.array(db[key]['data'], dtype='int16')  # real data
    label = np.array(db[key]['label'], dtype='uint8')

    id = key.split("-")[0]
    if id in val_ids:
        testfile.create_dataset(f"val/{key}/data", data=data)
        testfile.create_dataset(f"val/{key}/label", data=label)
    else:
        trainfile.create_dataset(f"train/{key}/data", data=data)
        trainfile.create_dataset(f"train/{key}/label", data=label)


hfile.close()

trainfile.close()
testfile.close()
