import torch
import torchvision
from torchvision import transforms, datasets
from torchvision import transforms as torch_transforms
from base import BaseDataLoader


class Cifar10DataLoader(BaseDataLoader):

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True,
                 img_size=32, transforms=None):

        trsfm = torch_transforms.Compose([
            torch_transforms.RandomCrop(32, padding=4),
            torch_transforms.RandomHorizontalFlip(),
            torch_transforms.ToTensor(),
            torch_transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        # transform_test = torch_transforms.Compose([
        #     torch_transforms.ToTensor(),
        #     torch_transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        # ])
        #
        # trainset = torchvision.datasets.CIFAR10(
        #     root='./data', train=True, download=True, transform=transform_train)
        # trainloader = torch.utils.data.DataLoader(
        #     trainset, batch_size=128, shuffle=True, num_workers=2)
        #
        # testset = torchvision.datasets.CIFAR10(
        #     root='./data', train=False, download=True, transform=transform_test)
        # testloader = torch.utils.data.DataLoader(
        #     testset, batch_size=100, shuffle=False, num_workers=2)


        # trsfm = torch_transforms.Compose([
        #     torch_transforms.Resize(img_size),
        #     torch_transforms.ToTensor(),
        #     torch_transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        # )
        self.data_dir = data_dir
        self.dataset = datasets.CIFAR10(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
