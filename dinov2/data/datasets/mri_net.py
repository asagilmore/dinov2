import csv
from enum import Enum
import logging
import os
from typing import Callable, List, Optional, Tuple, Union
from torch.utils.data import Dataset

import nibabel as nib
import numpy as np

class MRINet(Dataset):
    def __init__(self,
                 root: str,
                 split: str,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 output_dtype=np.float32,
                 load_pth=None) -> None:
        super().__init__()
        self._root = root
        self._split = split
        self._transform = transform
        self._target_transform = target_transform
        self._output_dtype = output_dtype

    def _mmap_dataset(self, path: str) -> None:
        files = os.listdir(path)
        for file in files:
            if file.endswith(".nii"):
                self._data.append(nib.load(os.path.join(path, file), mmap=True))

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        try:
            volume_data = self._data[index].get_fdata(dtype=self._output_dtype)
        except Exception as e:
            raise RuntimeError(f"can not read volume for sample {index}") from e
        image = self._transform(volume_data)
        target = self._target_transform(volume_data)

        return image, target

    # need to dump loaded entries to disk such that we can
    # have parity between machines NOTE: MAYBE NOT NEEDED
    # def _dump_loaded_entries(self, path: str) -> None:
        