# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pickle, os
from typing import Iterable, Optional
import cv2
import h5py
import numpy
from torch.utils.data import Dataset
from PIL import Image
import random

class GaoSuDataset(Dataset):
    def __init__(self,root: str, listFile: str) -> None:
        """Load cached dataset from file.

        Parameters
        ----------
        filepath : str
            Path of the file to load.
        """
        super().__init__()
        img_list_ = []
        with open(listFile, "r") as f:
            img_lines = f.readlines()
            for img_ in img_lines:
                img_list_.append(img_.strip())
        random.shuffle(img_list_)
        self._img_list = img_list_
        self._root_path = root

    def __getitem__(self, index):
        img = cv2.imread(os.path.join(self._root_path, self._img_list[index]))
        # img[925:995, 47:815, :] = 0
        # img[995:1060, 47:328, :] = 0
        # img[990:1060, 940:1680, :] = 0
        # img = Image.open(os.path.join(self._root_path, self._img_list[index]))
        
        image = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        if(self._img_list[index].split("/")[1]=="colmap_gaosu"):
            image = image.resize((720, 408), Image.Resampling.BILINEAR)
        # image = image.resize((960, 540), Image.Resampling.BILINEAR)
        # image = image.resize((640, 360), Image.Resampling.BILINEAR)

        return (image, "default_str_")
    
    def __len__(self):
        return len(self._img_list)
    



class CachedDataset(Dataset):
    """Cached dataset of serialized python objects."""

    @staticmethod
    def from_iterable(
        filepath: str, iterable: Iterable, take_n: Optional[int] = None
    ) -> "CachedDataset":
        """Creates a pytorch dataset from an iterable and serialize it to disk for easy loading.

        Parameters
        ----------
        filepath : str
            Path of file to save the cached dataset to.
        iterable : Iterable
            Iterable to traverse to store in cached dataset.
        take_n : Optional[int], optional
            Number of elements to store from iterable, by default None. When None is specified, the parameter will be inferred from iterable.

        Returns
        -------
        CachedDataset
            Newly created dataset.
        """
        db = h5py.File(filepath, mode="w")

        for i, item in enumerate(iterable):
            # early stopping if specified
            if i == take_n:
                break

            # converts python object to bytes
            obj_bytes = pickle.dumps(item)
            obj_bytes = numpy.void(obj_bytes)

            # save bytes as one h5py dataset
            key = str(i)
            dtype = h5py.opaque_dtype(obj_bytes.dtype)
            dset = db.create_dataset(key, (1,), dtype=dtype, maxshape=(1,))

            dset[0] = obj_bytes

        return CachedDataset(filepath)

    def __init__(self, filepath: str) -> None:
        """Load cached dataset from file.

        Parameters
        ----------
        filepath : str
            Path of the file to load.
        """
        super().__init__()

        self._filepath = filepath
        self._db = h5py.File(self._filepath, mode="r")

    # iteration doesn't work without this
    # TODO(Pierre) : investigate why
    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]

    def __len__(self):
        return len(self._db)

    def __getitem__(self, index):
        obj_bytes = self._db[str(index)][0]
        obj_bytes = obj_bytes.tobytes()
        # TODO(Pierre): Fix use of pickle to solve `python_pickle_is_bad`
        # reference : https://fburl.com/pickle_is_bad
        return pickle.loads(obj_bytes)

    def __del__(self):
        self._db.close()

    def __getstate__(self):
        return {"filepath": self._filepath}

    def __setstate__(self, newstate):
        self._filepath = newstate["filepath"]
        self._db = h5py.File(self._filepath, mode="r")
