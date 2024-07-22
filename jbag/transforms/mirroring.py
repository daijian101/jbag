import torch

from jbag.transforms.transforms import RandomTransform


class MirrorTransform(RandomTransform):
    def __init__(self, keys,
                 apply_probability,
                 allowed_axes,
                 p_per_axes: float = 0.5):
        """
        Brightness transform.
        Args:
            keys (str or sequence):
            apply_probability (float):
            allowed_axes (int or sequence): Axis(es) for mirroring.
            p_per_axes (bool, optional, default=False): Probability for performing transform on each axis.

        """
        super().__init__(keys, apply_probability)
        if isinstance(allowed_axes, int):
            allowed_axes = [allowed_axes]
        self.allowed_axes = allowed_axes
        self.p_per_axes = p_per_axes

    def _call_fun(self, data):
        allowed_axes = [axis for axis in self.allowed_axes if torch.rand(1) < self.p_per_axes]
        print(allowed_axes)
        if len(allowed_axes) == 0:
            return data
        for key in self.keys:
            value: torch.Tensor = data[key]
            value = torch.flip(value, allowed_axes)
            data[key] = value
        return data


if __name__ == '__main__':
    from cavass.ops import read_cavass_file, save_cavass_file
    import numpy as np

    image = read_cavass_file('/data1/dj/data/bca/cavass_data/images/N007PETCT.IM0')
    image = image[None].astype(np.float64)
    image = torch.from_numpy(image)
    data = {'image': image, 'image1': image.clone()}

    gbt = MirrorTransform(keys=['image', 'image1'], apply_probability=1, allowed_axes=[1,2], p_per_axes=1)
    data = gbt(data)

    image = data['image'][0].numpy()
    image1 = data['image1'][0].numpy()
    save_cavass_file('/data1/dj/tmp/image.IM0', image.astype(np.uint16),
                     reference_file='/data1/dj/data/bca/cavass_data/images/N007PETCT.IM0')
    save_cavass_file('/data1/dj/tmp/image1.IM0', image1.astype(np.uint16),
                     reference_file='/data1/dj/data/bca/cavass_data/images/N007PETCT.IM0')
