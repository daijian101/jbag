from typing import Union

import torch

from jbag.transforms._utils import get_scalar
from jbag.transforms import RandomTransform


class GaussianNoiseTransform(RandomTransform):
    def __init__(self, keys,
                 apply_probability,
                 noise_variance=Union[tuple[float], list[float]],
                 synchronize_channels: bool = False,
                 p_per_channel: float = 1):
        """
        Apply Gaussian noise to image.
        Args:
            keys (str or sequence):
            apply_probability (float):
            noise_variance (sequence): Range of noise variance.
            synchronize_channels (bool, optional, default=False): If True, use the same parameters for generating Gaussian noise.
            p_per_channel (float, optional, default=1): Probability of applying Gaussian noise for each channel.
        """
        super().__init__(keys, apply_probability)
        self.noise_variance = noise_variance
        self.synchronize_channels = synchronize_channels
        self.p_per_channel = p_per_channel

    def _call_fun(self, data):
        apply_to_channel = torch.where(torch.rand(len(self.keys)) < self.p_per_channel)[0]
        if len(apply_to_channel) == 0:
            return data
        if self.synchronize_channels:
            sigmas = [get_scalar(self.noise_variance), ] * len(apply_to_channel)
        else:
            sigmas = [get_scalar(self.noise_variance) for _ in range(len(apply_to_channel))]

        for c, sigma in zip(apply_to_channel, sigmas):
            value = data[self.keys[c]]
            image_shape = value.shape
            gaussian = torch.normal(0, sigma, image_shape)
            value += gaussian
            data[self.keys[c]] = value
        return data
