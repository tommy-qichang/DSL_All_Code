import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
from data_loader import nuclei_transforms as my_transforms
from dataset.nuclei_dataset import NucleiTrainDataset, NucleiTestDataset


class NucleiDataLoader(DataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True,
                 img_size=224, collate_fn=default_collate):
        self.h5_filepath = data_dir
        self.batch_size = batch_size
        self.validation_split = validation_split

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

            if isinstance(validation_split, list):
                assert len(validation_split) >= 2, "The validation_split array should contains at least 2 sub arrays for " \
                                                   "train and validation"
                # Should load two h5 files which is training and validation dataset.
                if isinstance(data_dir, str):
                    data_dir = [data_dir, data_dir]
                self.train_dataset = NucleiTrainDataset(data_dir[0], data_transform=trsfm)

                trsfm_val = my_transforms.Compose([
                    my_transforms.LabelEncoding(),
                    my_transforms.ToTensor(),
                    my_transforms.Normalize((0.7442, 0.5381, 0.6650), (0.1580, 0.1969, 0.1504))]
                )
                self.valid_dataset = NucleiTrainDataset(data_dir[1], data_transform=trsfm_val)

                print(f"GeneralDataLoader: training dataset length:{len(self.train_dataset)}, "
                      f"validation dataset length:{len(self.valid_dataset)}")

                self.init_kwargs = {
                    'dataset': self.train_dataset,
                    'batch_size': batch_size,
                    'shuffle': shuffle,
                    'collate_fn': collate_fn,
                    'num_workers': num_workers
                }
                super().__init__(**self.init_kwargs)

            elif isinstance(validation_split, float) or isinstance(validation_split, int):

                self.dataset = NucleiTrainDataset(self.h5_filepath, data_transform=trsfm)

                self.n_samples = len(self.dataset)

                self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)

                self.init_kwargs = {
                    'dataset': self.dataset,
                    'batch_size': batch_size,
                    # 'shuffle': shuffle,
                    'collate_fn': collate_fn,
                    'num_workers': num_workers
                }
                super().__init__(sampler=self.sampler, **self.init_kwargs)
        else:
            trsfm = my_transforms.Compose([
                my_transforms.LabelEncoding(),
                my_transforms.ToTensor(),
                my_transforms.Normalize((0.7442, 0.5381, 0.6650), (0.1580, 0.1969, 0.1504))]
            )
            dataset = NucleiTestDataset(self.h5_filepath, data_transform=trsfm)

            self.init_kwargs = {
                'dataset': dataset,
                'batch_size': batch_size,
                'shuffle': shuffle,
                'collate_fn': collate_fn,
                'num_workers': num_workers
            }
            super().__init__(**self.init_kwargs)

    def _split_sampler(self, split):
        # TODO: need to add split as tuple which indicates
        # TODO: need to add x fold cross validation sampler.
        if split == 0.0:
            return None, None

        idx_full = np.arange(self.n_samples)

        np.random.seed(0)
        np.random.shuffle(idx_full)

        if isinstance(split, int):
            assert split > 0
            assert split < self.n_samples, "validation set size is configured to be larger than entire dataset."
            len_valid = split
        else:
            len_valid = int(self.n_samples * split)

        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def split_validation(self):

        if hasattr(self, 'valid_sampler'):
            if self.valid_sampler is None:
                return None
            else:
                return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)
        elif hasattr(self, 'valid_dataset'):
            val_init_kwargs = self.init_kwargs.copy()
            val_init_kwargs['dataset'] = self.valid_dataset
            val_init_kwargs["shuffle"] = False
            return DataLoader(**val_init_kwargs)
