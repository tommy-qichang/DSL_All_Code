from base import BaseDataLoader
from dataset.acdc_dataset import ACDC_dataset
from dataset.acdc_test_dataset import AcdcTestDataset


class AcdcDataLoader(BaseDataLoader):
    def __init__(self, data_root, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True,
                 transforms=None):
        # if training:
        #     trsfm = my_transforms.Compose([
        #         # my_transforms.RandomHorizontalFlip(),
        #         # my_transforms.RandomRotation(10),
        #         my_transforms.RandomCrop(img_size),
        #         # my_transforms.LabelBinarization(),
        #         my_transforms.ToTensor(),
        #         my_transforms.Normalize((67.78, 67.78, 67.78), (62.8, 62.8, 62.8))
        #         ]
        #     )
        # else:
        #     trsfm = my_transforms.Compose([
        #         my_transforms.ToTensor(),
        #         my_transforms.Normalize((67.78, 67.78, 67.78), (62.8, 62.8, 62.8))
        #         ]
        #     )
        if training:
            self.dataset = ACDC_dataset(data_root, training, transform=transforms)
        else:
            # self.dataset = ACDC_dataset(data_root, training, transform=transforms)
            self.dataset = AcdcTestDataset(data_root, training, transform=transforms)

        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
