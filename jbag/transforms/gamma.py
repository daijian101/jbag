from typing import Union

import torch

from jbag.transforms._utils import get_non_one_scalar
from jbag.transforms import RandomTransform


class GammaTransform(RandomTransform):
    def __init__(self, keys,
                 apply_probability,
                 gamma: Union[tuple[float], list[float]],
                 p_invert_image: float,
                 synchronize_channels: bool = False,
                 p_per_channel: float = 1,
                 p_retain_stats: float = 1, ):
        """
        Gamma transform.
        Args:
            keys (str or sequence):
            apply_probability (float):
            gamma (sequence): Range for gamma value.
            p_invert_image(float): Probability of inverting output image.
            synchronize_channels (bool, optional, default=False): If True, use the same parameters for all channels.
            p_per_channel (float, optional, default=1): Probability of applying transform to each channel.
            p_retain_stats (float, optional, default=1): Probability of retaining data statistics.
        """
        assert len(gamma) == 2 and gamma[1] >= gamma[0]
        super().__init__(keys, apply_probability)
        self.gamma = gamma
        self.p_invert_image = p_invert_image
        self.synchronize_channels = synchronize_channels
        self.p_per_channel = p_per_channel
        self.p_retain_stats = p_retain_stats

    def _call_fun(self, data):
        apply_to_channel = torch.where(torch.rand(len(self.keys)) < self.p_per_channel)[0]
        if len(apply_to_channel) == 0:
            return data
        retain_stats = torch.rand(len(apply_to_channel)) < self.p_retain_stats
        invert_images = torch.rand(len(apply_to_channel)) < self.p_invert_image
        if self.synchronize_channels:
            gamma = [get_non_one_scalar(self.gamma), ] * len(apply_to_channel)
        else:
            gamma = [get_non_one_scalar(self.gamma) for _ in range(len(apply_to_channel))]

        for c, r, i, g in zip(apply_to_channel, retain_stats, invert_images, gamma):
            value = data[self.keys[c]]
            if i:
                value *= -1
            if r:
                mean_intensity = value.mean()
                std_intensity = value.std()
            min_intensity = value.min()
            intensity_range = value.max() - min_intensity
            value = ((value - min_intensity) / intensity_range.clamp(min=1e-7)).pow(g) * intensity_range + min_intensity
            if r:
                mean_here = value.mean()
                std_here = value.std()
                value -= mean_here
                value *= (std_intensity / std_here.clamp(min=1e-7))
                value += mean_intensity
            if i:
                value *= -1
            data[self.keys[c]] = value
        return data
