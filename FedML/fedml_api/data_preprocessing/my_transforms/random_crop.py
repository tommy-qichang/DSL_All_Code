import numpy as np
from .common_func import crop

class RandomCrop:
    """
    Crop the 2D numpy [c, h, w] and mask [h, w]/[c, h, w] in a sample
    """

    def __init__(self, output_size, padding=0, training=True):
        assert isinstance(output_size, (int, list))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = tuple(output_size)

        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            assert len(padding) == 2
            self.padding = tuple(padding)

        self.training = training

    def __call__(self, sample):
        if 'image' in sample:
            image, mask, misc = sample['image'], sample['mask'], sample['misc']

            c, h, w = image.shape
            new_h, new_w = self.output_size

            assert (h - new_h) >= 0 and (w - new_w) >= 0, f"The orig size:({h},{w}) should be equal or larger than " \
                                                          f"crop size:({new_h},{new_w})"

            if not self.training:
                # Test stage, will set the crop position at 0,0
                top = 0
                left = 0
            else:
                if "RandomCrop" in misc:  # preset parameters
                    top, left = misc["RandomCrop"]
                else:
                    top = np.random.randint(h - new_h + 1)
                    left = np.random.randint(w - new_w + 1)

            misc['RandomCrop'] = [top, left]

            image = crop(image, top, left, new_h, new_w, self.padding)
            mask = crop(mask, top, left, new_h, new_w, self.padding)
            if 'labels_ternary' in misc:
                misc['labels_ternary'] = crop(misc['labels_ternary'], top, left, new_h, new_w, self.padding)
            if 'weight_maps' in misc:
                misc['weight_maps'] = crop(misc['weight_maps'], top, left, new_h, new_w, self.padding)

            return {'image': image, 'mask': mask, 'misc': misc}
        else:
            mask, misc = sample['mask'], sample['misc']

            h, w = mask.shape[-2:]
            new_h, new_w = self.output_size

            assert (h - new_h) >= 0 and (w - new_w) >= 0, f"The orig size:({h},{w}) should be equal or larger than " \
                                                          f"crop size:({new_h},{new_w})"

            if not self.training:
                # Test stage, will set the crop position at 0,0
                top = 0
                left = 0
            else:
                top = np.random.randint(h - new_h + 1)
                left = np.random.randint(w - new_w + 1)

            misc['RandomCrop'] = [top, left]

            mask = crop(mask, top, left, new_h, new_w, self.padding)
            if 'labels_ternary' in misc:
                misc['labels_ternary'] = crop(misc['labels_ternary'], top, left, new_h, new_w, self.padding)
            if 'weight_maps' in misc:
                misc['weight_maps'] = crop(misc['weight_maps'], top, left, new_h, new_w, self.padding)

            return {'mask': mask, 'misc': misc}
