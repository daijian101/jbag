import numpy as np
import torch.nn as nn


class DeepSupervisionLossWrapper(nn.Module):
    def __init__(self, loss_criterion, weights=None):
        """
        Loss criterion wrapper for deep supervision.
        Args:
            loss_criterion (torch.nn.Module): Loss criterion function.
            weights (sequence[float, ...], optional, default=None): Weights for losses of different scale outputs.
        """
        super().__init__()
        if weights is not None:
            assert any([x != 0 for x in weights])
        self.loss_criterion = loss_criterion
        self.weights = weights

    def forward(self, *args):
        assert all([isinstance(arg, (list, tuple)) for arg in args])

        if self.weights is None:
            weights = [1] * len(args[0])
        else:
            weights = self.weights

        return sum(
            [weights[i] * self.loss_criterion(*inputs) for i, inputs in enumerate(zip(*args)) if weights[i] != 0])


def get_deep_supervision_loss_weights(network_config):
    deep_supervision_scales = get_deep_supervision_scales(network_config)
    weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
    return weights


def get_deep_supervision_scales(network_config):
    if 'deep_supervision' not in network_config or not network_config['deep_supervision']:
        return None
    strides = network_config['strides']
    deep_supervision_scales = list(list(i) for i in 1 / np.cumprod(np.vstack(strides), axis=0))[:-1]
    return deep_supervision_scales


def set_deep_supervision(network, flag):
    if hasattr(network, 'decoder'):
        if hasattr(network.decoder, 'deep_supervision'):
            network.decoder.deep_supervision = flag
