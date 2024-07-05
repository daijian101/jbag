import os.path
from typing import Union

import torch
from torch import nn
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer

from jbag.log import logger

MODEL = 'model'
OPTIMIZER = 'optimizer'


def get_unwrapped_model(model: nn.Module):
    if isinstance(model, (DataParallel, DistributedDataParallel)):
        model = model.module
    return model


def save_checkpoint(file, model: nn.Module, optimizer: Union[None, Optimizer] = None, **kwargs):
    checkpoint = {MODEL: get_unwrapped_model(model).state_dict()}
    if optimizer:
        checkpoint[OPTIMIZER] = optimizer.state_dict()
    for k, v in kwargs.items():
        if k in checkpoint:
            raise KeyError(f'Get duplicated key {k}.')
        checkpoint[k] = v
    file_path = os.path.split(file)[0]
    if not os.path.exists(file_path):
        os.makedirs(file_path, exist_ok=True)
    torch.save(checkpoint, file)


def load_checkpoint(file, model: Union[nn.Module, None] = None, optimizer: Union[Optimizer, None] = None):
    logger.info(f'Loading checkpoint {file}.')
    checkpoint = torch.load(file)
    if model:
        if MODEL not in checkpoint:
            logger.warning(f'Checkpoint {file} does not include model state dict.')
        else:
            model = get_unwrapped_model(model)
            model.load_state_dict(checkpoint[MODEL])
            logger.info('Model state loaded.')

    if optimizer:
        if OPTIMIZER not in checkpoint:
            logger.warning(f'Checkpoint {file} does not include optimizer state dict.')
        else:
            optimizer.load_state_dict(checkpoint[OPTIMIZER])
            logger.info(f'Optimizer state loaded.')
    return checkpoint
