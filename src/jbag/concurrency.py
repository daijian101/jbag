import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Sequence

from tqdm import tqdm

from jbag import setup_logger

logger = setup_logger(__name__)


def parallel_map(fn,
                 max_workers: int,
                 args_list: Sequence[tuple] = (),
                 kwargs_list: Sequence[dict] = (),
                 mp_context: str | None = None,
                 show_progress_bar: bool = True):
    """
    Run parallel processing using concurrent ProcessPoolExecutor.
    Args:
        fn (function): function to run.
        max_workers (int): number of parallel processes.
        args_list (sequence[tuple], optional, default=()): argument groups passed to `fn`.
        kwargs_list (sequence[dict], optional, default=()): keyword-arguments passed to `fn`.
        mp_context (string, optional, default=None): multiprocessing context ('fork', 'spawn', 'forkserver', or None).
        show_progress_bar (bool, optional, default=True): whether to show progress bar.

    Returns:

    """
    if not mp_context in ['fork', 'spawn', 'forkserver', None]:
        raise ValueError(
            f'Unsupported multiprocessing context: {mp_context}. Supported: fork, spawn, forkserver, or None.')

    n_args = len(args_list)
    n_kwds = len(kwargs_list)
    if n_args == 0 and n_kwds > 0:
        args_list = [()] * n_kwds
    elif n_kwds == 0 and n_args > 0:
        kwargs_list = [{}] * n_args
    elif n_args != n_kwds:
        raise ValueError('Mismatched number of args and number of kwargs.')

    max_procs = mp.cpu_count()
    requested_procs = max_workers
    max_workers = min(max_workers, max_procs, len(args_list) or 1)
    if max_workers < requested_procs:
        logger.warning(f'Adjusted processes to {max_workers} due to CPU limitation or number of tasks.')

    mp_context = mp.get_context(mp_context) if mp_context else None
    results = [None] * len(args_list)
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_context) as executor:
        fs = [executor.submit(fn, *args, **kwargs) for args, kwargs in zip(args_list, kwargs_list)]

        with tqdm(total=len(args_list), disable=not show_progress_bar) as pbar:
            for future in as_completed(fs):
                idx = fs.index(future)
                results[idx] = future.result()
                pbar.update()
    return results
