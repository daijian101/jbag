from typing import Union

from jbag.transforms._utils import get_non_one_scalar
from jbag.transforms.transforms import RandomTransform
import torch


class ContrastTransform(RandomTransform):
    def __init__(self, keys,
                 apply_probability,
                 contrast_range:Union[tuple[float], list[float]],
                 preserve_range: bool,
                 synchronize_channels: bool = False,
                 p_per_channel: float = 1):
        """
        Contrast transform.
        Args:
            keys (str or sequence):
            apply_probability (float):
            contrast_range (sequence): Multiplier for contrast adjustment is sampled from this range without value of `1` if `1` is in range.
            preserve_range (bool): If True, preserve the intensity range of the original image.
            synchronize_channels (bool, optional, default=False): if True, use the same parameters for all channels.
            p_per_channel (float, optional, default=1): Probability of applying transform to each channel.
        """
        assert len(contrast_range) == 2 and contrast_range[1] >= contrast_range[0]
        super().__init__(keys, apply_probability)
        self.contrast_range = contrast_range
        self.preserve_range = preserve_range
        self.synchronize_channels = synchronize_channels
        self.p_per_channel = p_per_channel

    def _call_fun(self, data):
        apply_to_channel = torch.where(torch.rand(len(self.keys)) < self.p_per_channel)[0]
        if len(apply_to_channel) == 0:
            return data
        if self.synchronize_channels:
            multipliers = [get_non_one_scalar(self.contrast_range), ] * len(apply_to_channel)
        else:
            multipliers = [get_non_one_scalar(self.contrast_range) for _ in range(len(apply_to_channel))]

        for c, m in zip(apply_to_channel, multipliers):
            value = data[self.keys[c]]
            mean_intensity = value.mean()
            if self.preserve_range:
                min_intensity = value.min()
                max_intensity = value.max()

            value -= mean_intensity
            value *= m
            value += mean_intensity
            if self.preserve_range:
                value.clamp_(min_intensity, max_intensity)
            data[self.keys[c]] = value
        return data


if __name__ == '__main__':
    from cavass.ops import read_cavass_file, save_cavass_file
    import numpy as np

    image = read_cavass_file('/data1/dj/data/bca/cavass_data/images/N007PETCT.IM0')
    image = image[None].astype(np.float64)
    image = torch.from_numpy(image)
    data = {'image': image}

    t = ContrastTransform(keys=['image'], apply_probability=1,contrast_range=(1, 2), preserve_range=True, synchronize_channels=False,  p_per_channel=1)

    data = t(data)

    image = data['image'][0]
    image = image.numpy()
    save_cavass_file('/data1/dj/tmp/image.IM0', image.astype(np.uint16),
                     reference_file='/data1/dj/data/bca/cavass_data/images/N007PETCT.IM0')


