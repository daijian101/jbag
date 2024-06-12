from abc import ABC, abstractmethod


class Transform(ABC):
    def __init__(self, keys):
        assert keys
        if isinstance(keys, str):
            keys = (keys, )
        self.keys = keys

    def __call__(self, data):
        return self._call_fun(data)

    @abstractmethod
    def _call_fun(self, data):
        ...
