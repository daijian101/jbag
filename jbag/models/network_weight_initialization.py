from functools import partial

from torch import nn

from jbag import logger
from jbag.config import Config


def kaiming_initialize(module, a=1e-2, nonlinearity='leaky_relu'):
    """
    kaiming initialization. a is set to 0.01 by default, this is only worked for LeakyReLU.
    Args:
        module (torch.nn.Module):
        a (float, optional, default=1e-2): Set 0 for ReLu. Set 1e-2 for LeakyReLU.
        nonlinearity (str, optional, default='leaky_relu'):

    Returns:

    """
    if isinstance(module, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
        module.weight = nn.init.kaiming_normal_(module.weight, a=a, nonlinearity=nonlinearity)
        if module.bias is not None:
            module.bias = nn.init.constant_(module.bias, 0)


def get_initialization_fn(method_name):
    match method_name:
        case 'kaiming_initialize':
            return kaiming_initialize


def initialize_network(network: nn.Module, network_config: Config):
    allow_init = False
    network_initialization_params = None
    method = None
    if 'initialization' in network_config and 'allow_init' in network_config.initialization and 'method' in network_config.initialization:
        allow_init = network_config.initialization.allow_init
        method = network_config.initialization.method

    if allow_init:
        if 'params' in network_config.initialization:
            network_initialization_params = network_config.initialization.params.as_primitive()
        else:
            logger.warning('No initialization parameters were provided for network weight initialization.')

    if allow_init:
        logger.info(f'********Initializing network weights********')
        logger.info(f'Method {method}')
        logger.info(f'Params: {network_initialization_params}')
        logger.info(f'********Initializing network weights********')
        init_fn = partial(get_initialization_fn(method), **network_initialization_params)
        network.apply(init_fn)
