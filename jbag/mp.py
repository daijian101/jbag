import multiprocessing as mp
from time import sleep

from tqdm import tqdm


def fork(fn, n_workers, params):
    """
    Invoke fn with multiple processors.
    Args:
        fn:
        n_workers:
        params:

    Returns:

    """
    r = []
    with mp.get_context('fork').Pool(n_workers) as p:
        for each_param in params:
            if not isinstance(each_param, tuple):
                each_param = tuple(each_param)
            r.append(p.starmap_async(fn, (each_param,)))

        remaining = list(range(len(params)))
        workers = [e for e in p._pool]

        with tqdm(desc=None, total=len(params)) as pbar:
            while len(remaining) > 0:
                all_alive = all([e.is_alive() for e in workers])
                if not all_alive:
                    raise RuntimeError(
                        'One of your background processes is missing. In that case reducing the number of workers might help.')

                done = [i for i in remaining if r[i].ready()]
                for _ in done:
                    pbar.update()

                remaining = [i for i in remaining if i not in done]
                sleep(0.1)

    results = [i.get()[0] for i in r]
    return results
