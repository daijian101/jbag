import torch
import torch.nn.functional as F


def gaussian_filter(input: torch.Tensor, sigma, axes=None, truncated=4):
    """
    Perform Gaussian filtering for the given array.
    This is a PyTorch implementation for image with data type of `torch.Tensor`.
    Args:
        input (torch.Tensor): Image to be filtered. The supported spatial dimensions of input is 1D, 2D, and 3D.
        An extra channel dimension should be attached before spatial dimensions in the input tensor, so the input tensor
        shape should be `[C, spatial_dims]`. This is because the data dimension requirement of PyTorch. Dimension `C`
        could be used for passing multiple tensor array.
        sigma (float or sequence): Standard deviation of the Gaussian kernel.
        If sigma is a sequence, then the number of elements must match those in axes sequence.
        axes (int or sequence, optional, default=None): axes for performing filtering.
        If None, Gaussian filtering will be performed on every axis.
        truncated (float, optional, default=6): Truncate the filter at this many standard deviations.

    Returns:
        Return tensor array of the same shape as input
    """
    assert 1 < input.dim() <= 4, 'Input tensor must have 2 to 4 dimensions.'

    if axes is None:
        axes = tuple(range(1, input.dim()))
    elif isinstance(axes, int):
        axes = [axes]

    for axis in axes:
        assert 0 < axis < input.dim(), 'Axis must be within the valid range.'

    if isinstance(sigma, (list, tuple)):
        assert len(sigma) == len(axes), 'Length of sigma must match length of axes.'
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
