import os.path
from typing import Union, LiteralString

import torch
from torch import nn
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer

from jbag.io import ensure_output_file_dir_existence
from jbag.log import logger

MODEL = 'model'
OPTIMIZER = 'optimizer'


def get_unwrapped_model(model: nn.Module):
    if isinstance(model, (DataParallel, DistributedDataParallel)):
        model = model.module
    return model


def save_weights(file: Union[str, LiteralString], model: nn.Module, optimizer: Union[None, Optimizer] = None,
                 **kwargs):
    checkpoint = {MODEL: get_unwrapped_model(model).state_dict()}
    if optimizer:
        checkpoint[OPTIMIZER] = optimizer.state_dict()
    for k, v in kwargs.items():
        if k in checkpoint:
            raise KeyError(f'Get duplicated key {k}.')
        checkpoint[k] = v
    ensure_output_file_dir_existence(file)
    torch.save(checkpoint, file)


def load_weights(file: Union[str, LiteralString], model: Union[nn.Module, None] = None,
                 optimizer: Union[Optimizer, None] = None):
    assert os.path.isfile(file), f'{file} does not exist or is not a file!'
    checkpoint = torch.load(file)
    if model:
        if MODEL not in checkpoint:
            logger.warning(f'{file} does not include model weights.')
        else:
            model = get_unwrapped_model(model)
            model.load_state_dict(checkpoint[MODEL])
            logger.info(f'Loading model weights from {file}.')

    if optimizer:
        if OPTIMIZER not in checkpoint:
            logger.warning(f'{file} does not include optimizer weights.')
        else:
            optimizer.load_state_dict(checkpoint[OPTIMIZER])
            logger.info(f'Loading optimizer weights from {file}.')
    return checkpoint
