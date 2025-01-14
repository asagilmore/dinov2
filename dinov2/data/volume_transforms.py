from typing import Sequence

import torch
from torchvision import transforms
from diplib import Gauss
from scipy.ndimage import rotate

# WIP
class GaussianBlur(transforms.RandomApply):
    """
    Apply Gaussian Blur to the np.ndarray image.
    """

    def __init__(self, *, p: float = 0.5, radius_min: float = 0.1, radius_max: float = 2.0):
        # NOTE: torchvision is applying 1 - probability to return the original image
        keep_p = 1 - p
        transform = Gauss(kernel_size=9, sigma=(radius_min, radius_max))
        super().__init__(transforms=[transform], p=keep_p)

# WIP
class RandomRotation(transforms.RandomApply):
    """
    Apply Random Rotation to the np.ndarray image.
    """

    def __init__(self, *, p: float = 0.5, degrees_range: float = 10):
        # NOTE: torchvision is applying 1 - probability to return the original image
        keep_p = 1 - p
        transform = rotate(degrees_range)

def randomRotation(volume, degrees_range):
    """
    Apply Random Rotation to the np.ndarray image.
    """

    axis_combinations = [
    [(0, 1)],
    [(0, 2)],
    [(1, 2)],
    [(0, 1), (0, 2)],
    [(0, 1), (1, 2)],
    [(0, 2), (1, 2)],
    [(0, 1), (0, 2), (1, 2)]
    ]

    axis = axis_combinations[np.random.randint(0, len(axis_combinations))]
    random.shuffle(axis)

    for axis in axis:
        degrees = np.random.uniform(-degrees_range, degrees_range)
        volume = rotate(volume, degrees_range, axes=axis, reshape)

    return volume


def gaussianBlur(volume, sigma, kernel_size):
    """
    Apply Gaussian Blur to the np.ndarray image.
    """
    if type(kernel_size) == int:
        if type(sigma)
    elif type(kernel_size) == tuple:

    return