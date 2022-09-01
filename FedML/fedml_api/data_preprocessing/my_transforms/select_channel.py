class SelectChannel:
    def __init__(self, img_ch=[0], label_ch=[0]):
        """
        Just select image channel(s) and label channel(s) by given img_ch and label_ch array.

        """
        assert isinstance(img_ch, list), "Image channels should be a list."
        assert isinstance(label_ch, list), "Label channels should be a list."
        self.img_ch = img_ch
        self.label_ch = label_ch

    def __call__(self, sample):
        image, mask, misc = sample['image'], sample['mask'], sample['misc']
        if len(self.img_ch) >= 0 and len(image.shape) == 3:
            if image.shape[2] == 3:
                image = image[:, :, self.img_ch]
            elif len(self.img_ch) >= 0 and image.shape[0] == 3:
                image = image[self.img_ch, :, :]

        if len(self.label_ch) >= 0 and len(mask.shape) == 3:
            mask = mask[:, :, self.label_ch]

        return {'image': image, 'mask': mask, 'misc': misc}
