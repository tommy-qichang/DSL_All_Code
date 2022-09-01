from torchvision import transforms

from base import BaseDataLoader
from dataset.mnist_split_dataset import MnistSplitDataset
from dataset.mnist_synthetic_dataset import MnistSyntheticDataset


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True,
                 synthetic_db=False, cls_filter=None, transforms = None):
        trsfm = transforms
        # trsfm = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.1307,), (0.3081,))
        # ])
        self.data_dir = data_dir
        if not synthetic_db:
            self.dataset = MnistSplitDataset(self.data_dir, train=training, download=True, transform=trsfm,
                                             cls_filter=cls_filter)
        else:
            self.dataset = MnistSyntheticDataset(self.data_dir, data_transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
