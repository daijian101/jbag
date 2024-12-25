from typing import Union

import torch

from jbag.transforms._utils import get_non_one_scalar
from jbag.transforms import RandomTransform


class MultiplicativeBrightnessTransform(RandomTransform):
    def __init__(self, keys,
                 apply_probability,
                 multiplier_range: Union[tuple[float], list[float]],
                 synchronize_channels: bool = False,
                 p_per_channel: float = 1):
        """
        Brightness transform.
        Args:
            keys (str or sequence):
            apply_probability (float):
            multiplier_range (sequence): Multiplier for brightness adjustment is sampled from this range without value of `1` if `1` is in range.
            synchronize_channels (bool, optional, default=False): If True, use the same parameters for all channels.
            p_per_channel (float, optional, default=1): Probability of applying transform to each channel.
        """
        assert len(multiplier_range) == 2 and multiplier_range[1] >= multiplier_range[0]
        super().__init__(keys, apply_probability)
        self.multiplier_range = multiplier_range
        self.synchronize_channels = synchronize_channels
        self.p_per_channel = p_per_channel

    def _call_fun(self, data):
        apply_to_channel = torch.where(torch.rand(len(self.keys)) < self.p_per_channel)[0]
        if len(apply_to_channel) == 0:
            return data
        if self.synchronize_channels:
            multipliers = [get_non_one_scalar(self.multiplier_range), ] * len(apply_to_channel)
        else:
            multipliers = [get_non_one_scalar(self.multiplier_range) for _ in range(len(apply_to_channel))]

        for c, m in zip(apply_to_channel, multipliers):
            value = data[self.keys[c]]
            value *= m
            data[self.keys[c]] = value
        return data
