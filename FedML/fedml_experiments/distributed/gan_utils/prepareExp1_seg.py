
import h5py
import numpy as np

train_all = "../../../../datasets/fedgan_syn/heart_h5_all/heart_resnet_9blocks_epoch110_experiment_0.h5"
train_h5 = "../../../../datasets/fedgan_seg_data_heart/heart_resnet_9blocks_epoch110_experiment_0_trainset.h5"
val_h5 = "../../../../datasets/fedgan_seg_data_heart/heart_resnet_9blocks_epoch110_experiment_0_valset.h5"


val_ids_whs = {'whs_0', 'whs_1', 'whs_3', 'whs_14'}

val_ids_miccai = {'miccai2008_04', 'miccai2008_05', 'miccai2008_08', 'miccai2008_16', 'miccai2008_18', 'miccai2008_20'}

val_ids_asoca = {'asoca_2', 'asoca_10', 'asoca_22', 'asoca_24', 'asoca_28', 'asoca_29', 'asoca_33', 'asoca_35'}

val_ids = list(val_ids_whs) + list(val_ids_asoca) + list(val_ids_miccai)

# prefix = ['miccai2008', 'whs', 'asoca']

hfile = h5py.File(train_all, 'r')
db = hfile['train']
keys = list(db.keys())

trainfile = h5py.File(train_h5, 'w')
testfile = h5py.File(val_h5, 'w')

for key in keys:
    data = np.array(db[key]['data'], dtype='uint8')  # syn data
    # data = np.array(db[key]['data'], dtype='int16')  # real data
    label = np.array(db[key]['label'], dtype='uint8')

    id = '_'.join(key.split("_")[:2])
    if id in val_ids:
        testfile.create_dataset(f"val/{key}/data", data=data)
        testfile.create_dataset(f"val/{key}/label", data=label)
    else:
        trainfile.create_dataset(f"train/{key}/data", data=data)
        trainfile.create_dataset(f"train/{key}/label", data=label)


hfile.close()

trainfile.close()
testfile.close()
