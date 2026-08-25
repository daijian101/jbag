import os.path

import torch
from torch import nn
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer

from jbag.io import ensure_output_file_dir_existence
from jbag.log import log


class CheckpointManager:
    MODEL_STATE_KEY = 'model_state'
    OPTIMIZER_STATE_KEY = 'optimizer_state'

    @staticmethod
    def unwrap_model(model: nn.Module):
        if isinstance(model, (DataParallel, DistributedDataParallel)):
            model = model.module
        return model

    @staticmethod
    def save_checkpoint(output_file: str, model: nn.Module, optimizer: Optimizer | None = None,
                        **kwargs):
        checkpoint = {CheckpointManager.MODEL_STATE_KEY: CheckpointManager.unwrap_model(model).state_dict()}
        if optimizer:
            checkpoint[CheckpointManager.OPTIMIZER_STATE_KEY] = optimizer.state_dict()

        overlap = {CheckpointManager.MODEL_STATE_KEY, CheckpointManager.OPTIMIZER_STATE_KEY} & kwargs.keys()
        if overlap:
            raise KeyError(f'Kwargs contain reserved keys: {overlap}.')
        checkpoint.update(kwargs)
        ensure_output_file_dir_existence(output_file)
        torch.save(checkpoint, output_file)
        log.info(f'Checkpoint saved to {output_file}.')

    @staticmethod
    def load_checkpoint(checkpoint_file: str, model: nn.Module | None= None,
                        optimizer: Optimizer | None = None, map_location=None, weights_only=False):
        if not os.path.isfile(checkpoint_file):
            raise FileNotFoundError(f'Checkpoint file {checkpoint_file} not found.')
        checkpoint = torch.load(checkpoint_file, map_location=map_location, weights_only=weights_only)
        if model is not None:
            model_state = checkpoint.get(CheckpointManager.MODEL_STATE_KEY)
            if model_state is not None:
                CheckpointManager.unwrap_model(model).load_state_dict(model_state)
                log.info(f'Model state loaded from {checkpoint_file}.')
            else:
                log.warning(f'{checkpoint_file} does not contain model state.')

        if optimizer is not None:
            optimizer_state = checkpoint.get(CheckpointManager.OPTIMIZER_STATE_KEY)
            if optimizer_state is not None:
                optimizer.load_state_dict(optimizer_state)
                log.info(f'Optimizer state loaded from {checkpoint_file}.')
            else:
                log.warning(f'{checkpoint_file} does not contain optimizer state.')

        return checkpoint
