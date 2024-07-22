from abc import ABC, abstractmethod

import numpy as np


class Transform(ABC):
    def __init__(self, keys):
        assert keys
        if isinstance(keys, str):
            keys = [keys]
        self.keys = keys

    def __call__(self, data):
        return self._call_fun(data)

    @abstractmethod
    def _call_fun(self, data):
        ...


class RandomTransform(Transform, ABC):
    def __init__(self, keys, apply_probability):
        super().__init__(keys)
        self.apply_probability = apply_probability

    def __call__(self, data):
        if np.random.random_sample() < self.apply_probability:
            return self._call_fun(data)
        else:
            return data
