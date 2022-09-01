from torchvision import transforms

from base import BaseDataLoader
from dataset.mura_dataset import MuraDataset


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
