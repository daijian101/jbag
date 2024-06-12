import numpy as np
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


class ZscoreNormalization(Transform):
    def __init__(self, keys, mean_value=None, std_value=None, mean_key=None, std_key=None):
        """
        Perform z-score normalization. `mean_key` and `std_key` indicate keys of mean and std value in data. You can
        also set common mean and std values for all data. Mean and std values provided by each sample will be used
        firstly if they exist.

        Args:
            keys (str or sequence):
            mean_value (float or None, optional, default=None):
            std_value (float or None, optional, default=None):
            mean_key (str or None, optional, default=None):
            std_key (str or None, optional, default=None):
        """

        super().__init__(keys)
        self.mean_value = mean_value
        self.std_value = std_value
        self.mean_key = mean_key
        self.std_key = std_key

    def _call_fun(self, data):
        mean = data[self.mean_key] if self.mean_key in data else self.mean_value
        std = data[self.std_key] if self.std_key in data else self.std_value
        assert mean and std

        for key in self.keys:
            value = data[key]
            value = (value - mean) / std
            data[key] = value
        return data


class MinMaxNormalization(Transform):
    def __init__(self, keys, lower_bound_percentile=1, upper_bound_percentile=99):
        """
        Perform min-max normalization.

        Args:
            keys (str or sequence):
            lower_bound_percentile (int, optional, default=1):
            upper_bound_percentile (int, optional, default=99):
        """

        super().__init__(keys)
        self.lower_bound_percentile = lower_bound_percentile
        self.upper_bound_percentile = upper_bound_percentile

    def _call_fun(self, data):
        for key in self.keys:
            image = data[key]
            min_value, max_value = np.percentile(image, (self.lower_bound_percentile, self.upper_bound_percentile))
            image = (image - min_value) / (max_value - min_value)
            data[key] = image
        return data
