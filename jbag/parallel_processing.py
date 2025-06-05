import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Union

from tqdm import tqdm

from jbag import logger


def execute(fn,
            processes: int,
            starargs: Union[list[tuple], tuple[tuple]] = (),
            starkwargs: Union[list[dict], tuple[dict]] = (),
            mp_context_method=None,
            show_progress_bar=True):
    """
    Run parallel processing using concurrent ProcessPoolExecutor.
    Args:
        fn (function): Function to be run.
        processes (int): Number of parallel processes.
        starargs (sequence, optional, default=()): Groups of args to be passed to `fn`.
        starkwargs (sequence, optional, default=()): Groups of kwargs to be passed to `fn`.
        mp_context_method (string, optional, default=None): Method of multiprocessing context.
            If None, use default mp context of operation system.
        show_progress_bar (bool, optional, default=True): Whether to show progress bar.

    Returns:

    """
    assert mp_context_method in ["fork", "spawn", "forkserver", None], \
        f"Unsupported multiprocessing context method: {mp_context_method}."

    n_starargs = len(starargs)
    n_starkwds = len(starkwargs)
    assert (n_starargs > 0 and n_starkwds == 0) or (n_starargs == 0 and n_starkwds > 0) or (n_starargs == n_starkwds), \
        "The group number of parameters of args and kwargs don't match."

    if n_starargs == 0 and n_starkwds != 0:
        starargs = [()] * n_starkwds

    if n_starargs != 0 and n_starkwds == 0:
        starkwargs = [{}] * n_starargs

    if processes > mp.cpu_count():
        logger.warn(f"Requested number of processes {processes} is greater than the total number of CPU processes. "
                    f"Set the number of processes to the number of CPU processes {mp.cpu_count()}.")
        processes = mp.cpu_count()

    if processes > len(starargs):
        logger.warn(f"Request number of processes {processes} is greater than the number of tasks {len(starargs)}. "
                    f"Set number of processes to {len(starargs)}.")
        processes = len(starargs)

    features = []
    r = []
    mp_context = mp.get_context(mp_context_method) if mp_context_method is not None else None
    with ProcessPoolExecutor(max_workers=processes, mp_context=mp_context) as executor:
        for args, kwargs in zip(starargs, starkwargs):
            features.append(executor.submit(fn, *args, **kwargs))

        with tqdm(total=len(starargs), disable=not show_progress_bar) as pbar:
            for future in as_completed(features):
                r.append(future.result())
                pbar.update()
    return r
