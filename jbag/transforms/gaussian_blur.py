import torch

from jbag.image.gaussian_filter import gaussian_filter
from jbag.transforms._utils import get_scalar
from jbag.transforms.transforms import RandomTransform


class GaussianBlurTransform(RandomTransform):
    def __init__(self, keys,
                 apply_probability,
                 blur_sigma=(1, 5),
                 synchronize_channels: bool = False,
                 synchronize_axes: bool = False,
                 p_per_channel: float = 1):
        """
        Filter image using Gaussian filter.
        Args:
            keys (str or sequence):
            apply_probability (float):
            blur_sigma (float or sequence[2], optional, default=(1, 5)): Sigma for Gaussian blur. If sequence with two elements, Gaussian blur sigma is uniformly sampled from [blur_sigma[0], blur_sigma[1]).
            synchronize_channels (bool, optional, default=False): If True, use the same parameters for all channels.
            synchronize_axes (bool, optional, default=False): If True, use the same parameters for all axes of an image.
            p_per_channel (float, optional, default=1): Probability of applying transform to each channel.
        """
        super().__init__(keys, apply_probability)

        self.blur_sigma = blur_sigma
        self.synchronize_channels = synchronize_channels
        self.synchronize_axes = synchronize_axes
        self.p_per_channel = p_per_channel

    def _call_fun(self, data):
        apply_to_channel = torch.where(torch.rand(len(self.keys)) < self.p_per_channel)[0]
        if len(apply_to_channel) == 0:
            return data

        max_spatial_dims = 0
        for c in apply_to_channel:
            value = data[self.keys[c]]
            spatial_dims = len(value.shape) - 1
            max_spatial_dims = max(max_spatial_dims, spatial_dims)

        if self.synchronize_axes:
            sigmas = [[get_scalar(self.blur_sigma), ] * max_spatial_dims, ] * len(apply_to_channel) \
                if self.synchronize_channels else \
                [[get_scalar(self.blur_sigma), ] * max_spatial_dims for _ in range(len(apply_to_channel))]
        else:
            sigmas = [[get_scalar(self.blur_sigma) for _ in range(max_spatial_dims)], ] * len(apply_to_channel) \
                if self.synchronize_channels else \
                [[get_scalar(self.blur_sigma) for _ in range(max_spatial_dims)] for _ in  range(len(apply_to_channel))]

        for c, sigma in zip(apply_to_channel, sigmas):
            value = data[self.keys[c]]
            spatial_dim = value.shape[1:]
            sigma = sigma[:len(spatial_dim)]
            axes = list(range(1, len(spatial_dim) + 1))
            value = gaussian_filter(value, sigma=sigma, axes=axes)
            data[self.keys[c]] = value
        return data


if __name__ == '__main__':
    from cavass.ops import read_cavass_file, save_cavass_file
    import numpy as np

    image = read_cavass_file('/data1/dj/data/bca/cavass_data/images/N007PETCT.IM0')
    image = image[None].astype(np.float64)
    image = torch.from_numpy(image)
    data = {'image': image, 'image1': image.clone()}

    t = GaussianBlurTransform(keys=['image', 'image1'], apply_probability=1, blur_sigma=(0.5, 1),
                              synchronize_channels=False, synchronize_axes=False, p_per_channel=1)

    data = t(data)

    image = data['image'][0].numpy()
    image1 = data['image1'][0].numpy()
    save_cavass_file('/data1/dj/tmp/image.IM0', image.astype(np.uint16),
                     reference_file='/data1/dj/data/bca/cavass_data/images/N007PETCT.IM0')
    save_cavass_file('/data1/dj/tmp/image1.IM0', image1.astype(np.uint16),
                     reference_file='/data1/dj/data/bca/cavass_data/images/N007PETCT.IM0')
