import numpy as np


def get_scalar(scalar_type):
    if isinstance(scalar_type, (list, tuple)):
        assert scalar_type[1] >= scalar_type[0]
        if scalar_type[0] == scalar_type[1]:
            scalar = scalar_type[0]
        else:
            scalar = np.random.uniform(*scalar_type)
    else:
        scalar = scalar_type
    return scalar


def get_non_one_scalar(scalar_range):
    """
    Get a scalar uniformly sampled from [scalar_range[0], scalar_range[1]), but skip scalar of 1.
    Args:
        scalar_range:

    Returns:

    """
    assert scalar_range[1] >= scalar_range[0]
    if scalar_range[0] == scalar_range[1]:
        scalar = scalar_range[0]
    else:
        if scalar_range[1] <= 1:
            scalar = np.random.uniform(scalar_range[0], scalar_range[1])
        else:
            if np.random.random_sample() < 0.5 and scalar_range[0] < 1:
                scalar = np.random.uniform(scalar_range[0], 1)
            else:
                scalar = np.random.uniform(max(scalar_range[0], 1), scalar_range[1])
    return scalar


def get_max_spatial_dims(keys, valid_keys, data):
    max_spatial_dims = 0
    for c in valid_keys:
        value = data[keys[c]]
        spatial_dims = len(value.shape) - 1
        max_spatial_dims = max(max_spatial_dims, spatial_dims)
    return max_spatial_dims
