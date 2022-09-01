from torchvision import datasets, transforms

from base import BaseDataLoader
from data_loader.mura_dataset import MuraDataset
from data_loader.BraTS_seg_dataset import BraTSSegDataset
import data_loader.my_transforms as my_transforms


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class Cifar10DataLoader(BaseDataLoader):

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True,
                 img_size=32):
        trsfm = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        self.data_dir = data_dir
        self.dataset = datasets.CIFAR10(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class MuraDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True,
                 img_size=128):
        trsfm = transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation((5,5)),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))]
        )
        self.data_dir = data_dir
        self.dataset = MuraDataset(self.data_dir, train=training, transforms=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class BraTSSegDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True,
                 img_size=224):
        if training:
            trsfm = transforms.Compose([
                my_transforms.RandomHorizontalFlip(),
                my_transforms.RandomRotation(10),
                my_transforms.RandomCrop(img_size),
                my_transforms.LabelBinarization(),
                my_transforms.ToTensor(),
                my_transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
            )
        else:
            trsfm = transforms.Compose([
                my_transforms.ToTensor(),
                my_transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
            )
        self.data_dir = data_dir
        self.dataset = BraTSSegDataset(self.data_dir, data_transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)