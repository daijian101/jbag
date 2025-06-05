from scipy.ndimage import distance_transform_edt
from skimage.segmentation import find_boundaries


def SDF(X, normalize=False):
    """
    Compute signed distance function(SDF) of the input.

    Args:
        X (numpy.ndarray or torch.Tensor): Input data ndarray or tensor.
        normalize (bool, optional, default=False): If True, perform max-min normalization for SDF.
    """
    pos_distance = distance_transform_edt(X)
    neg_segmentation = ~X
    neg_distance = distance_transform_edt(neg_segmentation)

    boundary = find_boundaries(X, mode="inner")
    eps = 1e-6
    if normalize:
        sdf = (neg_distance - neg_distance.min()) / (neg_distance.max() - neg_distance.min() + eps) - \
              (pos_distance - pos_distance.min()) / (pos_distance.max() - pos_distance.min() + eps)
    else:
        sdf = neg_distance - pos_distance
    sdf[boundary] = 0

    return sdf
