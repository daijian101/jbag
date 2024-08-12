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
    def __init__(self, keys, apply_probability, log, log_name):
        super().__init__(keys)
        self.apply_probability = apply_probability
        self.log = log
        self.log_name = log_name
        self.total_count = 0
        self.changed_count = 0

    def __call__(self, data):
        self.total_count += 1
        if np.random.random_sample() < self.apply_probability:
            # return self._call_fun(data)
            self.changed_count += 1
            data = self._call_fun(data)
            for key in self.keys:
                changed_data = data[key]
                self.log.add_image(f'log_name/{key}/{self.changed_count}', changed_data.squeeze(0))
            return data
        else:
            return data
