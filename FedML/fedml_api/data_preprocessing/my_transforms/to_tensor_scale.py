import torch


class ToTensorScale:
    def __init__(self, mask_type="long", maxv_im=None, maxv_lb=None, training=True):
        """
        Convert numpy array to Torch.Tensor
        """
        self.training = training
        self.mask_type = mask_type
        self.maxv_im = maxv_im
        self.maxv_lb = maxv_lb

    def __call__(self, sample):
        image, mask, misc = sample['image'], sample['mask'], sample['misc']

        if self.maxv_im is None:
            image = image / image.max()
        else:
            image = image / self.maxv_im
        image = torch.tensor(image, dtype=torch.float32)

        if self.mask_type == "long":
            mask = torch.tensor(mask, dtype=torch.long)
        elif self.mask_type == "float":
            if self.maxv_lb is None:
                mask = mask / mask.max()
            else:
                mask = mask / self.maxv_lb
            mask = torch.tensor(mask, dtype=torch.float32)

        if len(image.shape) == 2:
            image.unsqueeze_(0)

        return {'image': image, 'mask': mask, 'misc': misc}
