from collections import OrderedDict

import numpy as np
import torch

from .coordinate_generator import CoordinateGenerator
from .utils import central_crop


class WeightedPatchPicker:
    def __init__(self, data: tuple | list | dict | np.ndarray | torch.Tensor,
                 patch_size,
                 coordinates: CoordinateGenerator):
        """
        An iterable for picking a patch from data. The original data can be a single numpy.ndarray or a set of data.

        Args:
            data (sequence or mapping or numpy.ndarray or torch.Tensor):
            patch_size (sequence):
            coordinates (CoordinateGenerator):
        """
        self.data = data
        self.patch_size = patch_size
        self.index = 0
        self.coordinates = coordinates

    def reset(self):
        self.coordinates.regenerate()
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        index = self.index
        self.index += 1

        if index >= len(self):
            raise StopIteration

        if isinstance(self.data, (np.ndarray, torch.Tensor)):
            return central_crop(self.data, self.coordinates[index], self.patch_size)

        elif isinstance(self.data, dict):
            results = OrderedDict()
            for k, v in self.data.items():
                if isinstance(v, (np.ndarray, torch.Tensor)):
                    results[k] = central_crop(v, self.coordinates[index], self.patch_size)
                else:
                    results[k] = v
            return results

        else:
            # if the data is a bundle of samples, all ndarray data will be cropped according the candidate coordinate.
            results = []
            for data in self.data:
                if isinstance(data, (np.ndarray, torch.Tensor)):
                    results.append(central_crop(data, self.coordinates[index], self.patch_size))
                else:
                    results.append(data)
            return tuple(results)

    def __len__(self):
        return len(self.coordinates)
