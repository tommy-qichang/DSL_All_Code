from base import BaseDataLoader
from dataset.acdc_miccai_dataset import AcdcMiccaiTrainDataset


class AcdcMiccaiDataLoader(BaseDataLoader):
    def __init__(self, data_root, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True,
                 transforms=None, type="sax_15p"):
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
            self.dataset = AcdcMiccaiTrainDataset(data_root, type=type, transform=transforms)

        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
