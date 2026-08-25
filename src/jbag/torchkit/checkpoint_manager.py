import os.path
from pathlib import Path

import torch
from torch import nn
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer

from jbag.io import ensure_output_file_dir_existence
from jbag.log import log
from jbag.torchkit import is_main_process


class CheckpointManager:
    MODEL_STATE_KEY = 'model_state'
    OPTIMIZER_STATE_KEY = 'optimizer_state'
    GRAD_SCALER_STATE_KEY = 'grad_scaler_state'

    @staticmethod
    def unwrap_model(model: nn.Module):
        if isinstance(model, (DataParallel, DistributedDataParallel)):
            model = model.module
        return model

    @staticmethod
    def save_checkpoint(output_file: str | Path,
                        model: nn.Module,
                        optimizer: Optimizer | None = None,
                        grad_scaler: torch.amp.GradScaler | None = None,
                        on_main_process_only: bool = True,
                        **kwargs):
        if on_main_process_only and not is_main_process():
            return

        checkpoint = {CheckpointManager.MODEL_STATE_KEY: CheckpointManager.unwrap_model(model).state_dict()}
        if optimizer:
            checkpoint[CheckpointManager.OPTIMIZER_STATE_KEY] = optimizer.state_dict()
        if grad_scaler:
            checkpoint[CheckpointManager.GRAD_SCALER_STATE_KEY] = grad_scaler.state_dict()
        overlap = {CheckpointManager.MODEL_STATE_KEY, CheckpointManager.OPTIMIZER_STATE_KEY} & kwargs.keys()
        if overlap:
            raise KeyError(f'Kwargs contain reserved keys: {overlap}.')
        checkpoint.update(kwargs)
        ensure_output_file_dir_existence(output_file)
        torch.save(checkpoint, output_file)
        log.info(f'Checkpoint saved to {output_file}.')

    @staticmethod
    def load_checkpoint(checkpoint_file: str | Path,
                        model: nn.Module | None = None,
                        optimizer: Optimizer | None = None,
                        grad_scaler: torch.amp.GradScaler | None = None,
                        keys: list | tuple | None = None,
                        map_location=None,
                        weights_only=True):
        if not os.path.isfile(checkpoint_file):
            raise FileNotFoundError(f'Checkpoint file {checkpoint_file} not found.')

        checkpoint = torch.load(checkpoint_file, map_location=map_location, weights_only=weights_only)
        if model is not None:
            if CheckpointManager.MODEL_STATE_KEY in checkpoint:
                CheckpointManager.unwrap_model(model).load_state_dict(checkpoint[CheckpointManager.MODEL_STATE_KEY])
            else:
                raise KeyError(f'Checkpoint file {checkpoint_file} does not contain model state.')

        if optimizer is not None:
            if CheckpointManager.OPTIMIZER_STATE_KEY in checkpoint:
                optimizer.load_state_dict(checkpoint[CheckpointManager.OPTIMIZER_STATE_KEY])
            else:
                raise KeyError(f'Checkpoint file {checkpoint_file} does not contain optimizer state.')

        if grad_scaler is not None:
            if CheckpointManager.GRAD_SCALER_STATE_KEY in checkpoint:
                grad_scaler.load_state_dict(checkpoint[CheckpointManager.GRAD_SCALER_STATE_KEY])
            else:
                raise KeyError(f'Checkpoint file {checkpoint_file} does not contain grad_scaler state.')

        values = [] if keys else None
        if keys:
            for key in keys:
                if key in checkpoint:
                    values.append(checkpoint[key])
                else:
                    raise KeyError(f'Checkpoint file {checkpoint_file} does not contain {key}.')

        log.info(f'Checkpoint loaded from {checkpoint_file}.')

        return values if values else checkpoint
