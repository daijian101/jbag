import torch.distributed as dist


def is_main_process() -> bool:
    if not dist.is_available() or not dist.is_initialized():
        return True

    return dist.get_rank() == 0


def get_rank() -> int:
    if not dist.is_available() or not dist.is_initialized():
        return 0

    return dist.get_rank()
