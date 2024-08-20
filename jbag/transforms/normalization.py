import numpy as np

from jbag.transforms import Transform


class ZscoreNormalization(Transform):
    def __init__(self, keys, mean, std, lower_bound=None, upper_bound=None):
        """
        Z-score normalization.

        Args:
            keys (str or sequence):
            mean (float):
            std (float):
            lower_bound (float, optional, default=None):
            upper_bound (float, optional, default=None):
        """

        super().__init__(keys)
        self.mean = mean
        self.std = std
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def _call_fun(self, data):
        for key in self.keys:
            value = data[key]
            if self.lower_bound is not None or self.upper_bound is not None:
                np.clip(value, self.lower_bound, self.upper_bound, out=value)
            value -= self.mean
            value /= max(self.std, 1e-8)
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
