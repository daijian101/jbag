import time

import torch

from jbag.transforms._utils import get_non_one_scalar
from jbag.transforms.transforms import RandomTransform


class GammaTransform(RandomTransform):
    def __init__(self, keys,
                 apply_probability,
                 gamma,
                 p_invert_image,
                 synchronize_channels: bool = False,
                 p_per_channel: float = 1,
                 p_retain_stats: float = 1, ):
        """
        Brightness transform.
        Args:
            keys (str or sequence):
            apply_probability (float):
            multiplier_range (list[2] or tuple [2]): Multiplier for brightness adjustment is sampled from this range without value of `1` if `1` is in range.
            synchronize_channels (bool, optional, default=False): If True, use the same parameters for all channels.
            p_per_channel (float, optional, default=1): Probability of applying transform to each channel.
        """
        assert len(gamma) == 2 and gamma[1] >= gamma[0]
        super().__init__(keys, apply_probability)
        self.gamma = gamma
        self.p_invert_image = p_invert_image
        self.synchronize_channels = synchronize_channels
        self.p_per_channel = p_per_channel
        self.p_retain_stats = p_retain_stats

    def _call_fun(self, data):
        apply_to_channel = torch.where(torch.rand(len(self.keys)) < self.p_per_channel)[0]
        if len(apply_to_channel) == 0:
            return data
        retain_stats = torch.rand(len(apply_to_channel)) < self.p_retain_stats
        invert_images = torch.rand(len(apply_to_channel)) < self.p_invert_image
        if self.synchronize_channels:
            gamma = [get_non_one_scalar(self.gamma), ] * len(apply_to_channel)
        else:
            gamma = [get_non_one_scalar(self.gamma) for _ in range(len(apply_to_channel))]

        print(gamma)
        start = time.time()
        for c, r, i, g in zip(apply_to_channel, retain_stats, invert_images, gamma):
            value = data[self.keys[c]]
            if i:
                value *= -1
            if r:
                mean_intensity = value.mean()
                std_intensity = value.std()
            min_intensity = value.min()
            intensity_range = value.max() - min_intensity
            value = ((value - min_intensity) / intensity_range.clamp(min=1e-7)).pow(g) * intensity_range + min_intensity
            if r:
                mean_here = value.mean()
                std_here = value.std()
                value -= mean_here
                value *= (std_intensity / std_here.clamp(min=1e-7))
                value += mean_intensity
            if i:
                value *= -1
            data[self.keys[c]] = value
        return data


if __name__ == '__main__':
    # from cavass.ops import read_cavass_file, save_cavass_file
    import numpy as np
    import matplotlib.pyplot as plt
    #
    # image = read_cavass_file('/data1/dj/data/bca/cavass_data/images/N007PETCT.IM0')
    # image = image[None].astype(np.float64)
    # print(image.sum())
    # image = torch.from_numpy(image)
    # data = {'image': image, 'image1': image.clone()}
    #
    # gbt = GammaTransform(keys=['image', 'image1'], apply_probability=1, gamma=(1, 10),
    #                      p_invert_image=1, synchronize_channels=False,
    #                      p_per_channel=1, p_retain_stats=1)
    # data = gbt(data)
    #
    # image = data['image'][0]
    # image1 = data['image1'][0]
    # image = image.numpy()
    # image1 = image1.numpy()
    # print(image.sum())
    # print(image1.sum())
    # save_cavass_file('/data1/dj/tmp/image.IM0', image.astype(np.uint16),
    #                  reference_file='/data1/dj/data/bca/cavass_data/images/N007PETCT.IM0')
    # save_cavass_file('/data1/dj/tmp/image1.IM0', image1.astype(np.uint16),
    #                  reference_file='/data1/dj/data/bca/cavass_data/images/N007PETCT.IM0')
    from skimage.data import camera

    image = camera()
    plt.imshow(image, cmap='gray')
    plt.show()
    image = image[None]
    image = torch.from_numpy(image).to(torch.float32)
    data = {'image': image}
    gbt = GammaTransform(keys=['image'], apply_probability=1, gamma=(1, 2),
                         p_invert_image=1, synchronize_channels=False,
                         p_per_channel=1, p_retain_stats=1)
    data = gbt(data)
    gamma_image = data['image']
    image = gamma_image.squeeze(0).numpy().astype(np.uint8)
    plt.imshow(image, cmap='gray')
    plt.show()
    pass
