import logging

from torchvision import transforms
from scipy.ndimage import zoom
import numpy as np

logger = logging.getLogger("dinov2")


class DataAugmentationVolumeDINO(object):
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=96,
    ):

        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size

        logger.info("###################################")
        logger.info("Using Volume data augmentation parameters:")
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info("###################################")



class VolumeRandomCrop(object):
    def __init__(
        self,
        size,
        scale,
        aspect_ratio=(1,1),
        prob_apply_aspect_ratio=1.0,
        interpolation_order=1, # 1=linear
    ):
        # XXX apply random rotation before this
        if isinstance(size, tuple):
            assert len(size) == 3
            self.size = size
        elif isinstance(size, int):
            self.size = (size, size, size)
        else:
            raise ValueError("Size must be an int or a tuple")

        self.size = size

        if isinstance(scale, tuple):
            assert len(scale) == 2
            assert scale[0] < scale[1]
            assert scale[1] <= 1.0
            self.scale = scale
        elif isinstance(scale, float):
            assert 0 < scale <= 1.0
            self.scale = (scale, 1.0)
        else:
            raise ValueError("Scale must be a float or a tuple")

        self.aspect_ratio = aspect_ratio
        self.interpolation_order = interpolation_order
        self.prob_apply_aspect_ratio = prob_apply_aspect_ratio

    def __call__(self, volume: np.ndarray) -> np.ndarray:
        # volume shape: (C, D, H, W)
        if volume.ndim != 4:
            raise ValueError("Input volume must have 4 dimensions,"+
                             " not {}".format(volume.ndim))
        _, depth, height, width = volume.shape

        target_area = depth * height * width * np.random.uniform(self.scale[0],
                                                                 self.scale[1])


        if self.prob_apply_aspect_ratio < np.random.rand():
            x_aspect_ratio = 1
            y_aspect_ratio = 1
            z_aspect_ratio = 1
        else:
            x_aspect_ratio = np.random.uniform(self.aspect_ratio[0], self.aspect_ratio[1])
            y_aspect_ratio = np.random.uniform(self.aspect_ratio[0], self.aspect_ratio[1])
            z_aspect_ratio = np.random.uniform(self.aspect_ratio[0], self.aspect_ratio[1])

        # now we need to find some value p such that xp * yp * zp = target_area
        # thus, p = (target_area / xyz)^(1/3)

        p = (target_area / (x_aspect_ratio * y_aspect_ratio * z_aspect_ratio))**(1/3)
        # rounded to int so this is approx = to target_area
        d = int(x_aspect_ratio * p)
        h = int(y_aspect_ratio * p)
        w = int(z_aspect_ratio * p)

        # Ensure the crop dimensions are within the bounds of the original dimensions
        if d > depth or h > height or w > width:
            d = min(d, depth)
            h = min(h, height)
            w = min(w, width)

        # now we need to find the start and end points for the crop
        d_start = np.random.randint(0, depth - d + 1) if depth > d else 0
        h_start = np.random.randint(0, height - h + 1) if height > h else 0
        w_start = np.random.randint(0, width - w + 1) if width > w else 0
        d_end, h_end, w_end = d_start + d, h_start + h, w_start + w

        cropped_volume = volume[:, d_start:d_end, h_start:h_end, w_start:w_end]

        # now we need to resize the volume to the target size
        # we will use linear interpolation_order for this
        resized_volume = zoom(cropped_volume, (1, self.size[0]/d, self.size[1]/h, self.size[2]/w), order=self.interpolation_order)

        return resized_volume

class VolumeRandomRotation(object):
    def __call__(self, volume: np.ndarray) -> np.ndarray:
        # volume shape: (C, D, H, W)
        if volume.ndim != 4:
            raise ValueError("Input volume must have 4 dimensions,"+
                             " not {}".format(volume.ndim))
        _, depth, height, width = volume.shape

        axes = [(1, 2), (1, 3), (2, 3)]
        axis = np.random.choice(len(axes))
        k = np.random.choice([1, 2, 3])
        rotated_volume = np.rot90(volume, k, axes=axes[axis])

        return np.ascontiguousarray(rotated_volume)


class VolumeContrastJitter(object):
    def __init__(self, contrast_range=(0.5, 1.5)):
        self.contrast_range = contrast_range

    def __call__(self, volume: np.ndarray, mask=None) -> np.ndarray:
        # volume shape: (C, D, H, W)
        if volume.ndim != 4:
            raise ValueError("Input volume must have 4 dimensions,"+
                             " not {}".format(volume.ndim))
        _, depth, height, width = volume.shape

        if mask is not None:
            mean = np.mean(volume[mask])
        else:
            mean = np.mean(volume)

        factor = np.random.uniform(*self.contrast_range)
        volume = (volume - mean) * factor + mean

        return volume


class VolumeAugmentationCompose(object):
    def __init__(self, transforms, probs=None):
        if probs is None:
            probs = [1 for _ in range(len(transforms))]
        assert len(transforms) == len(probs)
        self.transforms = transforms
        self.probs = probs

    def __call__(self, volume: np.ndarray) -> np.ndarray:
        for transform, prob in zip(self.transforms, self.probs):
            if np.random.rand() < prob:
                volume = transform(volume)
        return volume