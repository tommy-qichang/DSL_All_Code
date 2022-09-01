from preprocess.brain_ct_preprocessing.save_dcm_data import window_image


class Windowfilter:
    def __init__(self, window_center, window_width):
        """
        Just select one channel for labels. For instance, from (X, Y, Z, 3) to (X, Y, Z).

        """
        self.window_center = window_center
        self.window_width = window_width

    def __call__(self, sample):
        image, mask, misc = sample['image'], sample['mask'], sample['misc']

        image = window_image(image, 200, 1000)

        return {'image': image, 'mask': mask, 'misc': misc}
