import torch


class ToTensor:
    def __init__(self, mask_type="long", training=True):
        """
        Convert numpy array to Torch.Tensor
        """
        self.training = training
        self.mask_type = mask_type

    def __call__(self, sample):
        image, mask, misc = sample['image'], sample['mask'], sample['misc']

        image = torch.tensor(image).float()
        if self.mask_type == "long":
            mask = torch.tensor(mask).long()
        elif self.mask_type == "float":
            mask = torch.tensor(mask).float()

        if len(image.shape) == 2:
            image.unsqueeze_(0)

        return {'image': image, 'mask': mask, 'misc': misc}
