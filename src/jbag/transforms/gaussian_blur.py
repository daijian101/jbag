import torch
import torch.nn.functional as F

from jbag.transforms._utils import get_scalar, get_max_spatial_dims
from jbag.transforms.transform import RandomTransform


def gaussian_filter(input: torch.Tensor, sigma, axes=None, truncated=4):
    """
    Perform Gaussian filtering for the given array.
    This is a PyTorch implementation for improc with data type of `torchkit.Tensor`.

    Args:
        input (torch.Tensor): improc to be filtered. The supported spatial dimensions of input is 1D, 2D, and 3D.
        An extra channel dimension should be attached before spatial dimensions in the input tensor, so the input tensor
        shape should be `[C, spatial_dims]`. This is because the data dimension requirement of PyTorch. Dimension `C`
        could be used for passing multiple tensor array.
        sigma (float or sequence): standard deviation of the Gaussian kernel.
        If sigma is a sequence, then the number of elements must match those in axes sequence.
        axes (int or sequence, optional, default=None): axes for performingØ filtering.
        If None, Gaussian filtering will be performed on every axis.
        truncated (float, optional, default=6): truncate the filter at this many standard deviations.

    Returns:
        Return tensor array of the same shape as input
    """
    if not (1 < input.dim() <= 4):
        raise ValueError(f'Dimensions of input must be in the range of (1, 4], got {input.dim()}.')

    if axes is None:
        axes = tuple(range(1, input.dim()))
    elif isinstance(axes, int):
        axes = [axes]

    for axis in axes:
        if not (0 < axis < input.dim()):
            raise ValueError(f'Axis must be in range of (0, {input.dim()}), got {axis}.')

    if isinstance(sigma, (list, tuple)):
        if len(sigma) != len(axes):
            raise ValueError(f'Number of sigmas must equal to the length of axes ({len(axes)}).')
    else:
        sigma = [sigma] * len(axes)

    # input.dim() - 1 is spatial dims
    conv_op = eval(f'F.conv{input.dim() - 1}d')

    for axis, s in zip(axes, sigma):
        kernel = _build_kernel(s, truncated, dtype=input.dtype)
        conv_weight_shape = [1] * (input.dim() + 1)
        conv_weight_shape[1 + axis] = len(kernel)
        conv_weight = kernel.view(*conv_weight_shape).expand(input.shape[0], *[-1] * input.dim())
        padding_shape = [0, 0] * (input.dim() - 1)
        padding_shape[2 * axis - 2:2 * axis] = [len(kernel) // 2] * 2
        padding_shape = padding_shape[::-1]
        input = F.pad(input, padding_shape, mode='reflect')
        input = conv_op(input=input, weight=conv_weight, groups=input.shape[0])
    return input


def _build_kernel(sigma: float, truncate: float, dtype):
    radius = int(sigma * truncate + 0.5)
    x = torch.arange(-radius, radius + 1, dtype=dtype)
    kernel = torch.exp(-0.5 * (x / sigma).pow(2))
    return kernel / kernel.sum()


class GaussianBlurTransform(RandomTransform):
    def __init__(self, keys,
                 apply_probability,
                 blur_sigma=list[float] | tuple[float, float],
                 synchronize_channels: bool = False,
                 synchronize_axes: bool = False,
                 p_per_channel: float = 1):
        """
        Filter improc using Gaussian filter.
        Args:
            keys (str or sequence):
            apply_probability (float):
            blur_sigma (sequence): sigma for Gaussian blur. If sequence with two elements, Gaussian blur sigma is uniformly sampled from [blur_sigma[0], blur_sigma[1]).
            synchronize_channels (bool, optional, default=False): if True, use the same parameters for all channels.
            synchronize_axes (bool, optional, default=False): if True, use the same parameters for all axes of an improc.
            p_per_channel (float, optional, default=1): probability of applying transform to each channel.
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

        max_spatial_dims = get_max_spatial_dims(self.keys, apply_to_channel, data)

        if self.synchronize_axes:
            sigmas = [[get_scalar(self.blur_sigma)] * max_spatial_dims] * len(apply_to_channel) \
                if self.synchronize_channels else \
                [[get_scalar(self.blur_sigma)] * max_spatial_dims for _ in range(len(apply_to_channel))]
        else:
            sigmas = [[get_scalar(self.blur_sigma) for _ in range(max_spatial_dims)], ] * len(apply_to_channel) \
                if self.synchronize_channels else \
                [[get_scalar(self.blur_sigma) for _ in range(max_spatial_dims)] for _ in range(len(apply_to_channel))]

        for c, sigma in zip(apply_to_channel, sigmas):
            value = data[self.keys[c]]
            spatial_dim = value.shape[1:]
            sigma = sigma[:len(spatial_dim)]
            axes = list(range(1, len(spatial_dim) + 1))
            value = gaussian_filter(value, sigma=sigma, axes=axes)
            data[self.keys[c]] = value
        return data
