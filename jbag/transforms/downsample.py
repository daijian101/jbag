from typing import Union

from torch.nn.functional import interpolate

from jbag.transforms import Transform


class DownsampleTransform(Transform):
    def __init__(self, keys, scales: Union[list, tuple]):
        """
        Down sample transform.
        Args:
            keys (str or sequence):
            scales (list, tuple): The element type should be float (used for all spatial dimensions) or list or tuple (each spatial dimension has a specific scale)
        """
        super().__init__(keys)
        self.scales = scales

    def _call_fun(self, data):
        for key in self.keys:
            value = data[key]
            results = []
            for scale in self.scales:
                if not isinstance(scale, (list, tuple)):
                    scale = [scale] * (value.ndim - 1)
                assert len(scale) == value.ndim - 1

                if all([i == 1 for i in scale]):
                    results.append(value)
                else:
                    new_shape = [round(i * j) for i, j in zip(value.shape[1:], scale)]
                    results.append(interpolate(value[None].float(), new_shape, mode='nearest-exact')[0].to(value.dtype))
            data[key] = results
        return data


if __name__ == '__main__':
    from cavass.ops import read_cavass_file
    import torch

    image = read_cavass_file('/data1/dj/data/bca/cavass_data/OAM/N007PETCT.BIM')
    image = image[None]
    image = torch.from_numpy(image)
    data = {'image': image}

    t = DownsampleTransform(keys=['image'], scales=[0.5, 0.2])

    data = t(data)

    image = data['image_downsampled']
    image1 = image[0][0].numpy()
    image2 = image[1][0].numpy()
