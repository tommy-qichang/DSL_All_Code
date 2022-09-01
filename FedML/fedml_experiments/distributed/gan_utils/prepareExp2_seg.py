
import h5py
import numpy as np

val_ids = ['Breast_TCGA-A7-A13E-01Z-00-DX1_3',
 'Prostate_TCGA-G9-6363-01Z-00-DX1_11',
 'Breast_TCGA-AR-A1AS-01Z-00-DX1_11',
 'Prostate_TCGA-G9-6356-01Z-00-DX1_6',
 'Kidney_TCGA-B0-5711-01Z-00-DX1_11',
 'Prostate_TCGA-G9-6356-01Z-00-DX1_2',
 'Liver_TCGA-18-5592-01Z-00-DX1_2',
 'Liver_TCGA-49-4488-01Z-00-DX1_13',
 'Kidney_TCGA-HE-7129-01Z-00-DX1_1',
 'Prostate_TCGA-G9-6336-01Z-00-DX1_14',
 'Breast_TCGA-A7-A13F-01Z-00-DX1_2',
 'Breast_TCGA-A7-A13E-01Z-00-DX1_11',
 'Prostate_TCGA-G9-6356-01Z-00-DX1_3',
 'Breast_TCGA-AR-A1AS-01Z-00-DX1_0',
 'Prostate_TCGA-G9-6348-01Z-00-DX1_8',
 'Kidney_TCGA-HE-7128-01Z-00-DX1_2',
 'Breast_TCGA-AR-A1AS-01Z-00-DX1_2',
 'Liver_TCGA-50-5931-01Z-00-DX1_9',
 'Liver_TCGA-49-4488-01Z-00-DX1_5',
 'Liver_TCGA-18-5592-01Z-00-DX1_0',
 'Prostate_TCGA-G9-6356-01Z-00-DX1_11',
 'Prostate_TCGA-G9-6356-01Z-00-DX1_0',
 'Kidney_TCGA-B0-5711-01Z-00-DX1_0',
 'Prostate_TCGA-G9-6363-01Z-00-DX1_7',
 'Breast_TCGA-AR-A1AK-01Z-00-DX1_4',
 'Breast_TCGA-A7-A13F-01Z-00-DX1_12',
 'Kidney_TCGA-HE-7130-01Z-00-DX1_7',
 'Breast_TCGA-A7-A13F-01Z-00-DX1_3',
 'Kidney_TCGA-B0-5711-01Z-00-DX1_1',
 'Liver_TCGA-50-5931-01Z-00-DX1_3',
 'Kidney_TCGA-B0-5711-01Z-00-DX1_6',
 'Liver_TCGA-18-5592-01Z-00-DX1_13',
 'Breast_TCGA-AR-A1AK-01Z-00-DX1_5',
 'Liver_TCGA-38-6178-01Z-00-DX1_14',
 'Kidney_TCGA-B0-5711-01Z-00-DX1_8',
 'Kidney_TCGA-HE-7129-01Z-00-DX1_7',
 'Prostate_TCGA-G9-6348-01Z-00-DX1_0',
 'Breast_TCGA-A7-A13F-01Z-00-DX1_9',
 'Breast_TCGA-AR-A1AK-01Z-00-DX1_13',
 'Liver_TCGA-49-4488-01Z-00-DX1_0',
 'Liver_TCGA-18-5592-01Z-00-DX1_15',
 'Kidney_TCGA-B0-5711-01Z-00-DX1_9',
 'Prostate_TCGA-G9-6348-01Z-00-DX1_12',
 'Liver_TCGA-18-5592-01Z-00-DX1_3',
 'Prostate_TCGA-G9-6336-01Z-00-DX1_4',
 'Kidney_TCGA-B0-5711-01Z-00-DX1_14',
 'Liver_TCGA-38-6178-01Z-00-DX1_13',
 'Kidney_TCGA-HE-7128-01Z-00-DX1_10']



train_all = "/data/datasets/asdgan_data/asdgan_syn/path_h5_all_gen_from_286_random_crop/path_resnet_9blocks_epoch150_experiment_0_2x.h5"
train_h5 = "/data/datasets/asdgan_data/asdgan_syn/path_h5_all_gen_from_286_random_crop/path_resnet_9blocks_epoch150_experiment_0_2x_trainset.h5"
val_h5 = "/data/datasets/asdgan_data/asdgan_syn/path_h5_all_gen_from_286_random_crop/path_resnet_9blocks_epoch150_experiment_0_2x_valset.h5"


hfile = h5py.File(train_all, 'r')

keys = list(hfile['images'].keys())

trainfile = h5py.File(train_h5, 'w')
testfile = h5py.File(val_h5, 'w')

for key in keys:
    id = key[:key.rfind('_')]
    if id in val_ids:
        testfile.create_dataset(f"images/{key}", data=hfile['images'][key])
        # testfile.create_dataset(f"labels/{key}", data=hfile['labels'][key])
        # testfile.create_dataset(f"labels_instance/{key}", data=hfile['labels_instance'][key])
        testfile.create_dataset(f"labels_ternary/{key}", data=hfile['labels_ternary'][key])
        testfile.create_dataset(f"weight_maps/{key}", data=hfile['weight_maps'][key])
    else:
        trainfile.create_dataset(f"images/{key}", data=hfile['images'][key])
        # trainfile.create_dataset(f"labels/{key}", data=hfile['labels'][key])
        # trainfile.create_dataset(f"labels_instance/{key}", data=hfile['labels_instance'][key])
        trainfile.create_dataset(f"labels_ternary/{key}", data=hfile['labels_ternary'][key])
        trainfile.create_dataset(f"weight_maps/{key}", data=hfile['weight_maps'][key])


hfile.close()

trainfile.close()
testfile.close()