import numpy as np


class HistNorm:
    def __init__(self, bd_low=0.1, bd_up=99.9, mean_i=0, std_i=1):
        self.bd_low = bd_low
        self.bd_up = bd_up
        self.mean_i = mean_i
        self.std_i = std_i

    def __call__(self, sample):
        image, mask, misc = sample['image'], sample['mask'], sample['misc']

        new_image = []
        for i in range(image.shape[0]):
            sub_img = image[i]

            sub_img[sub_img > np.percentile(sub_img, self.bd_up)] = np.percentile(sub_img, self.bd_up)
            sub_img[sub_img < np.percentile(sub_img, self.bd_low)] = np.percentile(sub_img, self.bd_low)
            # factor_scale = np.std(sub_img) / self.std_i
            # sub_img = sub_img / (factor_scale + 1e-6)
            # factor_shift = np.mean(sub_img) - self.mean_i
            # sub_img = sub_img - factor_shift
            new_image.append(sub_img)

        new_image = np.stack(new_image, axis=0)

        return {'image': new_image, 'mask': mask, 'misc': misc}
