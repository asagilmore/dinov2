import csv
from enum import Enum
import logging
import os
from typing import Callable, List, Optional, Tuple, Union
from torch.utils.data import Dataset

import nibabel as nib
import numpy as np
from tqdm import tqdm

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
        self._data = []

        self._mmap_dataset(self._root)

    def _mmap_dataset(self, path: str) -> None:
        files = os.listdir(path)
        for file in tqdm(files):
            if file.endswith(".nii") or file.endswith(".nii.gz"):
                self._data.append(nib.load(os.path.join(path, file), mmap=True))

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        try:
            volume_data = self._data[index].get_fdata(dtype=self._output_dtype)
            if volume_data.ndim == 3:
                volume_data = np.expand_dims(volume_data, axis=0)

            if volume_data.ndim != 4:
                raise ValueError("Input volume must have 4 dimensions,"+
                                 " not {}".format(volume_data.ndim))
        except Exception as e:
            raise RuntimeError(f"can not read volume for sample {index}") from e
        if self._transform:
            image = self._transform(volume_data)
        else:
            image = volume_data
        if self._target_transform:
            target = self._target_transform(volume_data)
        else:
            target = volume_data

        return image, target


    def __len__(self) -> int:
        return len(self._data)

    # need to dump loaded entries to disk such that we can
    # have parity between machines NOTE: MAYBE NOT NEEDED
    # def _dump_loaded_entries(self, path: str) -> None:
