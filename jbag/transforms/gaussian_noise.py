import matplotlib.pyplot as plt
import torch

from jbag.transforms._utils import get_scalar
from jbag.transforms.transforms import RandomTransform


class GaussianNoiseTransform(RandomTransform):
    def __init__(self, keys,
                 apply_probability,
                 noise_variance=(0, 0.1),
                 synchronize_channels: bool = False,
                 p_per_channel: float = 1):
        """
        Apply Gaussian noise to image.
        Args:
            keys (str or sequence):
            apply_probability (float):
            noise_variance (float or list[2] or tuple [2], optional, default=(0, 0,1)): if this parameter is sequence, the sigma of Gaussian function is uniformly sampled from [sequence[0], sequence[1]).
            synchronize_channels (bool, optional, default=False): If True, use the same parameters for generating Gaussian noise.
            p_per_channel (float, optional, default=1): Probability of applying Gaussian noise for each channel.
        """
        super().__init__(keys, apply_probability)
        self.noise_variance = noise_variance
        self.synchronize_channels = synchronize_channels
        self.p_per_channel = p_per_channel

    def _call_fun(self, data):
        apply_to_channel = torch.where(torch.rand(len(self.keys)) < self.p_per_channel)[0]
        if len(apply_to_channel) == 0:
            return data
        if self.synchronize_channels:
            sigmas = [get_scalar(self.noise_variance), ] * len(apply_to_channel)
        else:
            sigmas = [get_scalar(self.noise_variance) for _ in range(len(apply_to_channel))]

        for c, sigma in zip(apply_to_channel, sigmas):
            value = data[self.keys[c]]
            image_shape = value.shape
            gaussian = torch.normal(0, sigma, image_shape)
            value += gaussian
            data[self.keys[c]] = value
        return data


if __name__ == '__main__':
    image = torch.zeros((512, 512), dtype=torch.float)
    image = image[None]
    data = {'image': image, 'image1': image.clone()}

    t = GaussianNoiseTransform(keys=['image', 'image1'], apply_probability=1, noise_variance=0.15,
                              synchronize_channels=False, p_per_channel=1)

    data = t(data)
    image = data['image'][0].numpy()
    image1 = data['image1'][0].numpy()
    image = (image - image.min()) / (image.max() - image.min())
    image1 = (image1 - image1.min()) / (image1.max() - image1.min())
    plt.imshow(image, cmap='gray')
    plt.show()
    plt.imshow(image1, cmap='gray')
    plt.show()
