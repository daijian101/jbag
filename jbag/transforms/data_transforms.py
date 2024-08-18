import numpy as np
import torch
from einops import rearrange, repeat

from jbag.transforms.transforms import Transform


class ToType(Transform):
    def __init__(self, keys, dtype):
        super().__init__(keys)
        self.dtype = dtype

    def _call_fun(self, data):
        for key in self.keys:
            value = data[key].astype(self.dtype)
            data[key] = value
        return data


class ToTensor(Transform):
    def __init__(self, keys):
        super().__init__(keys)

    def _call_fun(self, data):
        for key in self.keys:
            data[key] = torch.from_numpy(data[key])
        return data


class Rearrange(Transform):
    def __init__(self, keys, pattern):
        """
        Change the arrangement of given elements.

        Args:
            keys (str or sequence):
            pattern (str): Arranging pattern. For example "i j k -> j k i".
        """
        super().__init__(keys)
        self.pattern = pattern

    def _call_fun(self, data):
        for key in self.keys:
            value = data[key]
            value = rearrange(value, self.pattern)
            data[key] = value
        return data


class Repeat(Transform):
    def __init__(self, keys, pattern, **kwargs):
        super().__init__(keys)
        self.pattern = pattern
        self.kwargs = kwargs

    def _call_fun(self, data):
        for key in self.keys:
            value = data[key]
            value = repeat(value, self.pattern, **self.kwargs)
            data[key] = value
        return data


class AddChannel(Transform):
    def __init__(self, keys, dim):
        """
        Add additional dimension in specific position.

        Args:
            keys (str or sequence):
            dim (int):
        """
        super().__init__(keys)
        self.dim = dim

    def _call_fun(self, data):
        for key in self.keys:
            value = data[key]
            value = np.expand_dims(value, axis=self.dim)
            data[key] = value
        return data
