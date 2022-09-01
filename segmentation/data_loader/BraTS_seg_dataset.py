
import torch.utils.data as data
import os
from PIL import Image
import numpy as np


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def img_loader(path, num_channels):
    if num_channels == 1:
        img = Image.open(path).convert("L")
    else:
        img = Image.open(path).convert('RGB')

    return img


# get the image list pairs
def get_imgs_list(data_dir):
    img_list = []

    img_filenames = os.listdir('{:s}/images'.format(data_dir))
    label_filenames = os.listdir('{:s}/labels'.format(data_dir))

    for img_name in img_filenames:
        if not is_image_file(img_name):
            continue

        if img_name in label_filenames:
            item = ('{:s}/images/{:s}'.format(data_dir, img_name),
                    '{:s}/labels/{:s}'.format(data_dir, img_name))
            img_list.append(tuple(item))

    return img_list


# dataset that supports multiple images
class BraTSSegDataset(data.Dataset):
    def __init__(self, data_root, data_transform=None, loader=img_loader):
        super(BraTSSegDataset, self).__init__()

        self.img_list = get_imgs_list(data_root)
        if len(self.img_list) == 0:
            raise(RuntimeError('Found 0 image pairs in given directories.'))

        self.data_transform = data_transform
        self.loader = loader

    def __getitem__(self, index):
        img_path, label_path = self.img_list[index]
        img, label = self.loader(img_path, 3), self.loader(label_path, 1)
        if self.data_transform is not None:
            img, label = self.data_transform((img, label))

        return img, label

    def __len__(self):
        return len(self.img_list)

