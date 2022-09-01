from base import BaseDataLoader
from data_loader import my_transforms
from dataset.brats_dataset import BratsDataset


class BratsDataLoader(BaseDataLoader):
    def __init__(self, h5_filepath, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True,
                 img_size=224):
        if training:
            trsfm = my_transforms.Compose([
                my_transforms.RandomHorizontalFlip(),
                my_transforms.RandomRotation(10),
                my_transforms.RandomCrop(img_size),
                my_transforms.LabelBinarization(),
                my_transforms.ToTensor(),
                my_transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
            )
        else:
            trsfm = my_transforms.Compose([
                my_transforms.ToTensor(),
                my_transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
            )
        self.h5_filepath = h5_filepath
        self.dataset = BratsDataset(self.h5_filepath, data_transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
