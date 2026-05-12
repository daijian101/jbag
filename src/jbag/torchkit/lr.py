from torch.optim import Optimizer


def get_lr(optimizer: Optimizer, keep_one=True):
    if optimizer is not None:
        if len(optimizer.param_groups) == 1 or keep_one:
            return optimizer.param_groups[0]['lr']
        lrs = [param_group['lr'] for param_group in optimizer.param_groups]
        return lrs
    raise ValueError('Optimizer is None')
