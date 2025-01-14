import logging

from torchvision import transforms

from .transforms import (
    GaussianBlur,
    make_normalize_transform,
)

class DataAugmentationVolumeDINO(object):
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=96,
    )

    