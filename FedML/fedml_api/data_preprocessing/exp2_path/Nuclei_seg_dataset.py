import torch
import torch.utils.data as data
from PIL import Image
import h5py
import numpy as np


def get_train_imgs_list(h5_filepath):
    img_list = []

    with h5py.File(h5_filepath, 'r') as h5_file:
        img_filenames = list(h5_file['images'].keys())
        label_filenames = list(h5_file['labels_ternary'].keys())
        weight_filenames = list(h5_file['weight_maps'].keys())

        for img_name in img_filenames:
            if img_name in label_filenames and img_name in weight_filenames:
                item = ('images/{:s}'.format(img_name),
                        'labels_ternary/{:s}'.format(img_name),
                        'weight_maps/{:s}'.format(img_name))
                img_list.append(tuple(item))

    return img_list


def get_test_imgs_list(h5_filepath):
    img_list = []

    with h5py.File(h5_filepath, 'r') as h5_file:
        img_filenames = list(h5_file['images'].keys())
        label_filenames = list(h5_file['labels_ternary'].keys())
        weight_filenames = list(h5_file['weight_maps'].keys())
        instance_label_filenames = list(h5_file['labels_instance'].keys())

        for img_name in img_filenames:
            if img_name in label_filenames and img_name in weight_filenames and img_name in instance_label_filenames:
                item = ('images/{:s}'.format(img_name),
                        'labels_ternary/{:s}'.format(img_name),
                        'weight_maps/{:s}'.format(img_name),
                        'labels_instance/{:s}'.format(img_name))
                img_list.append(tuple(item))

    return img_list


class NucleiSegTrainDataset(data.Dataset):
    def __init__(self, h5_filepath, data_transform=None):
        super(NucleiSegTrainDataset, self).__init__()
        self.h5_filepath = h5_filepath
        self.data_transform = data_transform

        self.img_list = get_train_imgs_list(h5_filepath)
        if len(self.img_list) == 0:
            raise(RuntimeError('Found 0 image pairs in given directories.'))
        self.h5_file = h5py.File(self.h5_filepath, 'r')

    def __getitem__(self, index):

        img_path, label_path, weight_map_path = self.img_list[index]
        img, label, weight_map = self.h5_file[img_path][()], self.h5_file[label_path][()], self.h5_file[weight_map_path][()]
        if np.max(label) == 1:
            label = (label * 255).astype(np.uint8)
        img = Image.fromarray(img, 'RGB')
        label = Image.fromarray(label)
        weight_map = Image.fromarray(weight_map)
        if self.data_transform is not None:
            img, weight_map, label = self.data_transform((img, weight_map, label))

        weight_map = weight_map.float().div(20)
        if label.max() == 255:
            label /= 255
        if weight_map.dim() == 3:
            weight_map = weight_map.squeeze(0)
        if label.dim() == 3:
            label = label.squeeze(0)

        # return img, weight_map, label, img_path.split('/')[-1]
        return {'image': img, 'label': label, 'weight_map': weight_map}

    def __len__(self):
        return len(self.img_list)


class NucleiSegTestDataset(data.Dataset):
    def __init__(self, h5_filepath, data_transform=None):
        super(NucleiSegTestDataset, self).__init__()
        self.h5_filepath = h5_filepath
        self.data_transform = data_transform

        self.img_list = get_test_imgs_list(h5_filepath)
        if len(self.img_list) == 0:
            raise(RuntimeError('Found 0 image pairs in given directories.'))
        self.h5_file = h5py.File(self.h5_filepath, 'r')

    def __getitem__(self, index):
        img_path, label_path, weight_map_path, instance_label_path = self.img_list[index]
        img, label, weight_map, instance_label = self.h5_file[img_path][()], self.h5_file[label_path][()], \
                                                 self.h5_file[weight_map_path][()], self.h5_file[instance_label_path][()]
        img = Image.fromarray(img, 'RGB')
        label = Image.fromarray(label)
        weight_map = Image.fromarray(weight_map)
        instance_label = torch.from_numpy(instance_label.astype(np.int16))
        if self.data_transform is not None:
            img, weight_map, label = self.data_transform((img, weight_map, label))

        weight_map = weight_map.float().div(20)
        if label.max() == 255:
            label /= 255
        if weight_map.dim() == 3:
            weight_map = weight_map.squeeze(0)
        if label.dim() == 3:
            label = label.squeeze(0)

        # return img, weight_map, label, instance_label, img_path.split('/')[-1]
        return {'image': img, 'label': label, 'weight_map': weight_map, 'instance_label': instance_label}

    def __len__(self):
        return len(self.img_list)