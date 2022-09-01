import numpy as np
import random

class HistMatch2:
    def __init__(self):
        self.templates = None

    def __call__(self, sample):
        image, mask, misc = sample['image'], sample['mask'], sample['misc']
        if "train" in misc['img_path']:
            if self.templates is None:
                self.templates = [None for i in range(int(misc['len']))]
            idx = misc['index']
            if self.templates[idx] is None:
                #do not has the source.
                t_values, t_counts = np.unique(image.ravel(), return_counts=True)
                self.templates.append([t_values,t_counts])
            else:
                #has already stored the template, so use it.
                if random.random()>0.5:
                    oldshape = image.shape
                    s_values, bin_idx, s_counts = np.unique(image.ravel(), return_inverse=True,
                                                            return_counts=True)
                    template_id = random.randint(0, len(self.templates)-1)
                    t_values, t_counts = self.templates[template_id]

                    s_quantiles = np.cumsum(s_counts).astype(np.float64)
                    s_quantiles /= s_quantiles[-1]
                    t_quantiles = np.cumsum(t_counts).astype(np.float64)
                    t_quantiles /= t_quantiles[-1]

                    # interpolate linearly to find the pixel values in the template image
                    # that correspond most closely to the quantiles in the source image
                    #interp_t_values = np.zeros_like(source,dtype=float)
                    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)
                    image = interp_t_values[bin_idx].reshape(oldshape)


        return {'image': image, 'mask': mask, 'misc': misc}
