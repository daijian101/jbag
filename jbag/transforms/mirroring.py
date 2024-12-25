import torch

from jbag.transforms import RandomTransform


class MirrorTransform(RandomTransform):
    def __init__(self, keys,
                 apply_probability,
                 allowed_axes,
                 p_per_axes: float = 0.5):
        """
        Mirror transform.
        Args:
            keys (str or sequence):
            apply_probability (float):
            allowed_axes (int or sequence): Axis(es) for mirroring.
            p_per_axes (bool, optional, default=False): Probability for performing transform on each axis.

        """
        super().__init__(keys, apply_probability)
        if isinstance(allowed_axes, int):
            allowed_axes = [allowed_axes]
        self.allowed_axes = allowed_axes
        self.p_per_axes = p_per_axes

    def _call_fun(self, data):
        allowed_axes = [axis for axis in self.allowed_axes if torch.rand(1) < self.p_per_axes]
        if len(allowed_axes) == 0:
            return data
        for key in self.keys:
            value: torch.Tensor = data[key]
            value = torch.flip(value, allowed_axes)
            data[key] = value
        return data
