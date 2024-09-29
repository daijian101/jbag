from typing import Union

from torch.nn.functional import interpolate

from jbag.transforms import Transform


class DownsampleTransform(Transform):
    def __init__(self, keys, scales: Union[list, tuple]):
        """
        Down sample transform.
        Args:
            keys (str or sequence):
            scales (list, tuple): The element type should be float (used for all spatial dimensions) or list or tuple (each spatial dimension has a specific scale)
        """
        super().__init__(keys)
        self.scales = scales

    def _call_fun(self, data):
        for key in self.keys:
            value = data[key]
            results = []
            for scale in self.scales:
                if not isinstance(scale, (list, tuple)):
                    scale = [scale] * (value.ndim - 1)
                assert len(scale) == value.ndim - 1

                if all([i == 1 for i in scale]):
                    results.append(value)
                else:
                    new_shape = [round(i * j) for i, j in zip(value.shape[1:], scale)]
                    results.append(interpolate(value[None].float(), new_shape, mode='nearest-exact')[0].to(value.dtype))
            data[key] = results
        return data
