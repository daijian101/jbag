import numpy as np
import torch

from jbag.transforms.transform import Transform


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
            if isinstance(value, np.ndarray):
                value = np.expand_dims(value, axis=self.axis)
            elif isinstance(value, torch.Tensor):
                value = torch.unsqueeze(value, dim=self.axis)
            else:
                raise TypeError(f"Unsupported type {type(value)}.")
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


class Convert3DTo2D(Transform):
    def __init__(self, keys):
        super().__init__(keys)

    def _call_fun(self, data):
        for key in self.keys:
            shape_key = f'orig_shape_{key}'
            if shape_key in data:
                raise KeyError(f"{shape_key} already exists.")

            value = data[key]
            data[shape_key] = value.shape
            data[key] = value.reshape(-1, *value.shape[-2:])
        return data


class Convert2DTo3D(Transform):
    def __init__(self, keys):
        super().__init__(keys)

    def _call_fun(self, data):
        for key in self.keys:
            shape_key = f'orig_shape_{key}'
            if not shape_key in data:
                raise KeyError(f"{shape_key} does not exist.")
            orig_shape = data.pop(shape_key)

            data[key] = data[key].reshape(orig_shape)
        return data
