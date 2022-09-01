import numpy as np
import random
from scipy import ndimage, misc

class RotationTransformBase(SpatialTransformBase):
    """
    Rotation transformation base class.
    """
    @staticmethod
    def get_rotation_transform(dim, angles):
        """
        Returns the sitk transform based on the given parameters.
        :param dim: The dimension.
        :param angles: List of angles for each dimension (in radians).
        :return: The sitk.AffineTransform().
        """
        if not isinstance(angles, list):
            angles = [angles]
        assert isinstance(angles, list), 'Angles parameter must be a list of floats, one for each dimension.'
        assert len(angles) in [1, 3], 'Angles must be a list of length 1 for 2D, or 3 for 3D.'

        t = sitk.AffineTransform(dim)

        if len(angles) == 1:
            # 2D
            t.Rotate(0, 1, angle=angles[0])
        elif len(angles) > 1:
            # 3D
            # rotate about x axis
            t.Rotate(1, 2, angle=angles[0])
            # rotate about y axis
            t.Rotate(0, 2, angle=angles[1])
            # rotate about z axis
            t.Rotate(0, 1, angle=angles[2])

        return t

    def get(self, **kwargs):
        """
        Returns the actual sitk transfrom object with the current parameters.
        :param kwargs: Various arguments that may be used by the transformation, e.g., 'image', 'input_size, 'landmarks', etc.
        :return: sitk transform.
        """
        raise NotImplementedError

class RandomRotate3d:
    def __init__(self, training=True):
        self.training = training

    def __call__(self, sample):
        """
        Randomly flip the numpy image horizontal or/and vertical
        Args:
            sample: {'image':..., 'mask':...}
            image size: [c, h, w]
            mask size: [h, w]
        Returns:
            Randomly flipped image.
        """
        image, mask, misc = sample['image'], sample['mask'], sample['misc']
        if not self.training or "train" not in misc['img_path']:
            # If testing, will not flip.
            return {'image': image, 'mask': mask}

        if len(image.shape) == 2:
            image = np.expand_dims(image, 0)

        if random.random() < 0.5:
            #rotate image and mask +=10deg
            deg = [random.randint(-35,35),random.randint(-35,35),random.randint(-35,35)]

            image = ndimage.rotate(image, deg[0], axes=(0,1), reshape=False)
            mask = ndimage.rotate(mask, deg[0], axes=(0,1), reshape=False,order=0)

            image = ndimage.rotate(image, deg[1], axes=(1,2), reshape=False)
            mask = ndimage.rotate(mask, deg[1], axes=(1,2), reshape=False,order=0)

            image = ndimage.rotate(image, deg[2], axes=(0,2), reshape=False)
            mask = ndimage.rotate(mask, deg[2], axes=(0,2), reshape=False,order=0)

        return {'image': image, 'mask': mask, 'misc': misc}
