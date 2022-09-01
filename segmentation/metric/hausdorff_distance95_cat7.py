
import surface_distance
import torch
import numpy as np

from metric.hausdorff_distance95_avg import hausdorff_distance95_calc


def hausdorff_distance95_cat7(output, target, misc):

    return hausdorff_distance95_calc(output, target, misc, 7)



