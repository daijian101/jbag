import numpy as np
from skimage.measure import label, regionprops


def get_maximum_connectivity_region(data):
    labeled_image = label(data, connectivity=2)
    regions = regionprops(labeled_image)
    if regions:
        largest_region = max(regions, key=lambda r: r.area)
        max_connectivity_region = np.zeros_like(data, dtype=np.uint8)
        indices = largest_region.coords
        indices = indices.transpose()
        max_connectivity_region[*indices] = 1
        return max_connectivity_region
    return np.zeros_like(data, dtype=np.uint8)
