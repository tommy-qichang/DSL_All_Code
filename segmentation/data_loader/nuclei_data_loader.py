from base import BaseDataLoader
from data_loader import my_transforms
from dataset.nuclei_dataset import NucleiTrainDataset, NucleiTestDataset


class NucleiDataLoader(BaseDataLoader):
    def __init__(self, h5_filepath, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True,
                 img_size=224):
        self.h5_filepath = h5_filepath
        if training:
            trsfm = my_transforms.Compose([
                my_transforms.RandomResize(0.8, 1.25),
                my_transforms.RandomHorizontalFlip(),
                my_transforms.RandomAffine(0.3),
                my_transforms.RandomRotation(90),
                my_transforms.RandomCrop(img_size),
                my_transforms.LabelEncoding(),
                my_transforms.ToTensor(),
                my_transforms.Normalize((0.7442, 0.5381, 0.6650), (0.1580, 0.1969, 0.1504))]
            )
            self.dataset = NucleiTrainDataset(self.h5_filepath, data_transform=trsfm)
        else:
            trsfm = my_transforms.Compose([
                my_transforms.LabelEncoding(),
                my_transforms.ToTensor(),
                my_transforms.Normalize((0.7442, 0.5381, 0.6650), (0.1580, 0.1969, 0.1504))]
            )
            self.dataset = NucleiTestDataset(self.h5_filepath, data_transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
