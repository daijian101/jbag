from jbag.metrics import sdf
from jbag.transforms import Transform


class SDF(Transform):
    def __init__(self, keys, normalize):
        """
        Compute signed distance function.

        Args:
            keys (str or sequence): Binary segmentation for computing SDF.
            normalize (bool): If `True`, normalize the SDF by min-max normalization.
        """
        super().__init__(keys)
        self.normalize = normalize

    def _call_fun(self, data):
        for key in self.keys:
            segmentation = data[key]
            sdf_map = sdf(segmentation, self.normalize)
            data[f'{key}_SDF'] = sdf_map
        return data
