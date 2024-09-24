# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import csv
from enum import Enum
import logging
import os
from typing import Callable, List, Optional, Tuple, Union

import numpy as np

from .extended import ExtendedVisionDataset

class HCP1200Dataset(ExtendedVisionDataset):
    def __init__(
        self,
        *,
        root: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self._root = root

        self._setup_img_paths()

    def _setup_img_paths(self) -> None:
        files = os.listdir(self._root)
        files = [f for f in files if f.endswith(".jpg") or f.endswith(".JPEG")]
        self._imgs = files

    def get_target(self, index: int):
        return np.nan

    def __len__(self) -> int:
        return len(self._imgs)

    def get_image_data(self, index: int) -> bytes:
        img_path = os.path.join(self._root, self._imgs[index])
        with open(img_path, "rb") as f:
            img_data = f.read()
        return img_data