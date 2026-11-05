import numpy as np
import torch


def get_margin(patch_size):
    """
    Determine the margin length that cannot be the central point of the patch of which cannot fit the size of patch_size.
    Return a len(patch_size) X 2 matrix with left and right (before and after, top and bottom) margin size.

    For an even patch size ``p``, the left margin is ``p // 2`` and the right margin is ``p // 2 - 1`` (they sum to
    ``p - 1``). This means a full-size crop of length ``p`` has center ``p // 2``. For an odd patch size both margins
    are ``p // 2``.

    Args:
        patch_size (sequence):

    Returns:

    """
    patch_size = np.atleast_1d(np.asarray(patch_size, dtype=np.int64))
    if patch_size.ndim != 1:
        raise ValueError(f'The patch size must be a 1-dimensional sequence, but got shape {patch_size.shape}.')
    if np.any(patch_size <= 0):
        raise ValueError(f'The patch size must be positive, but got {patch_size.tolist()}.')

    left = patch_size // 2
    right = patch_size - 1 - left
    return np.stack([left, right], axis=-1)


def get_non_margin_region(image_shape: tuple | list, patch_size: tuple | list):
    """
    Build a mask that cover the area which can be viewed as the central point of the patch with the size of
    patch_size. Mask the central area that can be extracted. Left the margin area.

    Args:
        image_shape (sequence):
        patch_size (sequence):

    Returns:

    """
    image_shape = np.atleast_1d(np.asarray(image_shape, dtype=np.int64))
    patch_size = np.atleast_1d(np.asarray(patch_size, dtype=np.int64))
    if image_shape.ndim != 1 or patch_size.ndim != 1 or image_shape.size != patch_size.size:
        raise ValueError(
            f'The image shape and patch size must be 1-dimensional sequences with the same length, '
            f'but got {image_shape.shape} and {patch_size.shape}.')
    if np.any(image_shape < patch_size):
        raise ValueError(
            f'The image shape {image_shape.tolist()} must not be smaller than the patch size '
            f'{patch_size.tolist()}.')

    margin = get_margin(patch_size)
    mask = np.zeros(tuple(int(i) for i in image_shape), dtype=bool)
    mask[tuple(slice(int(left), int(i) - int(right)) for i, (left, right) in zip(image_shape, margin))] = True
    return mask


def encode_one_hot(data):
    """
    Encode an integer label map as a one-hot array with the category axis at the beginning.

    Args:
        data (numpy.ndarray): non-negative, integer-valued label map.

    Returns:
        numpy.ndarray: boolean array with shape ``(num_classes, *data.shape)``.

    """
    data = np.asarray(data)
    if data.size == 0:
        raise ValueError('Data must not be empty.')

    if np.issubdtype(data.dtype, np.bool_):
        data = data.astype(np.int64)
    elif np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.int64, copy=False)
    elif np.issubdtype(data.dtype, np.floating):
        if not np.all(np.isfinite(data)) or not np.array_equal(data, np.floor(data)):
            raise ValueError('Floating point labels must be finite and integer-valued.')
        data = data.astype(np.int64)
    else:
        raise TypeError(f'Unsupported label dtype {data.dtype!r}; an integer label map is required.')

    if data.min() < 0:
        raise ValueError(
            'Negative labels are not supported. Remap or mask them before calling encode_one_hot.')

    # Do not use uint8 here: it silently wraps labels greater than 255.
    num_classes = int(data.max()) + 1
    result = np.zeros((data.size, num_classes), dtype=bool)
    result[np.arange(data.size), data.ravel()] = True
    result = result.reshape(*data.shape, num_classes)

    # move category axis to the first. [C, ...]
    return np.moveaxis(result, -1, 0)


def central_crop(data: np.ndarray | torch.Tensor, center, shape, copy=True):
    """
    Crop patch with the shape from data at the central coordinate.

    The patch spans ``[center - shape // 2, center - shape // 2 + shape)`` along each spatial dimension. Consequently,
    for an even ``shape`` the requested center must be the right of the two middle pixels, and a full-size crop has
    center ``shape // 2`` (not ``shape // 2 - 1``). Use :func:`crop_patch` when an explicit corner is available.

    Args:
        data (numpy.ndarray or torch.Tensor):
        center (sequence):
        shape (sequence):
        copy (bool, optional, default=True): if `True`, the returned patch does not share memory with `data`. Keep it
            enabled when patches are cached and may be modified in place (e.g. by normalization transforms).

    Returns:

    """
    if not isinstance(data, (np.ndarray, torch.Tensor)):
        raise TypeError(f'Data must be a numpy.ndarray or torch.Tensor, but got {type(data)!r}.')

    shape = np.atleast_1d(np.asarray(shape, dtype=np.int64))
    center = np.atleast_1d(np.asarray(center, dtype=np.int64))
    if shape.ndim != 1 or shape.size == 0:
        raise ValueError(f'The shape must be a non-empty 1-dimensional sequence, but got {shape.shape}.')
    if center.shape != shape.shape:
        raise ValueError(f'The center {center.tolist()} and shape {shape.tolist()} must have the same length.')
    if len(data.shape) < len(shape):
        raise ValueError(
            f'The data shape {tuple(data.shape)} must have at least {len(shape)} dimensions to crop a patch of '
            f'shape {shape.tolist()}.')

    # coordinate is the corner coordinate responding to the central.
    coordinate = center - get_margin(shape)[:, 0]
    cropped = crop_patch(data, coordinate, shape)
    if copy:
        cropped = cropped.clone() if isinstance(cropped, torch.Tensor) else cropped.copy()
    return cropped


def crop_patch(data: np.ndarray | torch.Tensor, coordinate, patch_size):
    """
    Crop patch from `data` at the left/superior/anterior `coordinate`. If data has larger dimensions than shape
    dimensions N, the last N dimensions will be cropped.

    Args:
        data  (numpy.ndarray or torch.Tensor):
        coordinate (sequence):
        patch_size (sequence):

    Returns:
        A view into `data`. Use :func:`central_crop` if an independent patch is required.

    """
    if not isinstance(data, (np.ndarray, torch.Tensor)):
        raise TypeError(f'Data must be a numpy.ndarray or torch.Tensor, but got {type(data)!r}.')

    coordinate = np.atleast_1d(np.asarray(coordinate, dtype=np.int64))
    patch_size = np.atleast_1d(np.asarray(patch_size, dtype=np.int64))
    if coordinate.ndim != 1 or coordinate.size == 0:
        raise ValueError(f'The coordinate must be a non-empty 1-dimensional sequence, but got {coordinate.shape}.')
    if coordinate.shape != patch_size.shape:
        raise ValueError(
            f'The coordinate {coordinate.tolist()} and patch size {patch_size.tolist()} must have the same length.')
    if np.any(patch_size <= 0):
        raise ValueError(f'The patch size must be positive, but got {patch_size.tolist()}.')
    if len(data.shape) < len(coordinate):
        raise ValueError(
            f'The data shape {tuple(data.shape)} must have at least {len(coordinate)} dimensions to crop a patch of '
            f'shape {patch_size.tolist()}.')

    end = coordinate + patch_size
    spatial_shape = np.asarray(data.shape[-len(coordinate):], dtype=np.int64)
    if np.any(coordinate < 0) or np.any(end > spatial_shape):
        raise ValueError(
            f'Cropping coordinate {coordinate.tolist()} with patch size {patch_size.tolist()} is out of bounds for '
            f'data shape {tuple(data.shape)}.')

    slice_sequence = [slice(int(i), int(j)) for i, j in zip(coordinate, end)]
    # append the extra dimensions, batch/channel...
    slice_sequence = [slice(None)] * (len(data.shape) - len(end)) + slice_sequence
    return data[tuple(slice_sequence)]
