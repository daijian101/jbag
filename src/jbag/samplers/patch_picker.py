from collections import OrderedDict

import numpy as np
import torch

from .coordinate_generator import CoordinateGenerator
from .utils import central_crop


class WeightedPatchPicker:
    def __init__(self, data: tuple | list | dict | np.ndarray | torch.Tensor,
                 patch_size,
                 coordinates: CoordinateGenerator,
                 keys=None):
        """
        An iterable for picking a patch from data. The original data can be a single numpy.ndarray or a set of data.

        Patches are independent copies of the source data, so in-place transforms applied to a patch do not modify
        the cached sample.

        Args:
            data (sequence or mapping or numpy.ndarray or torch.Tensor):
            patch_size (sequence):
            coordinates (CoordinateGenerator):
            keys (str or sequence or None, optional, default=None): keys (mapping) or indices (sequence) that should
                be cropped. If `None`, every array-like value whose trailing dimensions can hold the patch is cropped.
                Non-spatial metadata arrays (e.g. spacing or affine matrices) are therefore skipped automatically.
        """
        self.data = data
        self.patch_size = tuple(int(s) for s in np.atleast_1d(patch_size))
        if not self.patch_size:
            raise ValueError('The patch size must not be empty.')
        self.index = 0
        self.coordinates = coordinates
        if keys is None:
            self.keys = None
        elif isinstance(keys, str):
            self.keys = (keys,)
        else:
            try:
                self.keys = tuple(keys)
            except TypeError:
                # A single key/index is also accepted.
                self.keys = (keys,)

    def reset(self):
        self.coordinates.regenerate()
        self.index = 0

    def _should_crop(self, value):
        """Return whether `value` looks like spatial data that can hold a patch."""
        if not isinstance(value, (np.ndarray, torch.Tensor)):
            return False
        n = len(self.patch_size)
        if value.ndim < n:
            return False
        return all(int(s) >= p for s, p in zip(value.shape[-n:], self.patch_size))

    def _crop(self, value, index):
        if not isinstance(value, (np.ndarray, torch.Tensor)):
            raise TypeError(
                f'Value selected for cropping must be a numpy.ndarray or torch.Tensor, but got {type(value)!r}.')
        return central_crop(value, self.coordinates[index], self.patch_size)

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self):
            raise StopIteration

        index = self.index
        self.index += 1

        if isinstance(self.data, (np.ndarray, torch.Tensor)):
            return self._crop(self.data, index)

        if isinstance(self.data, dict):
            results = OrderedDict()
            for k, v in self.data.items():
                crop = self._should_crop(v) if self.keys is None else k in self.keys
                results[k] = self._crop(v, index) if crop else v
            return results

        if isinstance(self.data, (tuple, list)):
            # if the data is a bundle of samples, all array data will be cropped according the candidate coordinate.
            results = []
            for i, v in enumerate(self.data):
                crop = self._should_crop(v) if self.keys is None else i in self.keys
                results.append(self._crop(v, index) if crop else v)
            return tuple(results)

        raise TypeError(
            f'Unsupported data type {type(self.data)!r}; expected a numpy.ndarray, torch.Tensor, dict, tuple or list.')

    def __len__(self):
        return len(self.coordinates)
