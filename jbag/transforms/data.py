import numpy as np
import torch

from jbag.transforms import Transform


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
            value = data[key]
            if isinstance(value, np.ndarray):
                data[key] = torch.from_numpy(value)
            else:
                data[key] = torch.tensor(value)
        return data


class AddChannel(Transform):
    def __init__(self, keys, axis):
        """
        Add additional dimension in specific position.

        Args:
            keys (str or sequence):
            axis (int):
        """
        super().__init__(keys)
        self.axis = axis

    def _call_fun(self, data):
        for key in self.keys:
            value = data[key]
            value = np.expand_dims(value, axis=self.axis)
            data[key] = value
        return data


class Repeat(Transform):
    def __init__(self, keys, repeats, axis):
        """
        Repeat specific dimension.

        Args:
            keys (str or sequence):
            axis (int):
            repeats (int):
        """
        super().__init__(keys)
        self.axis = axis
        self.repeats = repeats

    def _call_fun(self, data):
        for key in self.keys:
            value = data[key]
            value = np.repeat(value, self.repeats, axis=self.axis)
            data[key] = value
        return data
