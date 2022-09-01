class Homochrome:
    def __init__(self, img_single_ch=0, label_single_ch=0):
        """
        Just select one channel for labels. For instance, from (X, Y, Z, 3) to (X, Y, Z).

        """
        self.img_single_ch = img_single_ch
        self.label_single_ch = label_single_ch

    def __call__(self, sample):
        image, mask, misc = sample['image'], sample['mask'], sample['misc']
        if self.img_single_ch >= 0 and len(image.shape) == 3:
            if image.shape[2] == 3:
                image = image[:, :, self.img_single_ch]
            elif self.img_single_ch >= 0 and image.shape[0] == 3:
                image = image[self.img_single_ch, :, :]

        if self.label_single_ch >= 0 and len(mask.shape) == 3:
            mask = mask[:, :, self.label_single_ch]

        return {'image': image, 'mask': mask, 'misc': misc}
