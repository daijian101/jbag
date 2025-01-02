from time import sleep

from jbag.parallel_processing import execute


def fn(a, b, c):
    # b = 1
    # print(a + b)
    sleep(1)
    return a + b + c


if __name__ == '__main__':
    p = [(1, 2), (3, 4), (5, 6)]
    kwargs = [{'c': 2, 'a': 1, 'b': 2}, {'c': 10, 'a': 1, 'b': 20}, {'c': 3, 'a': 10, 'b': 2}]
    r = execute(fn, 3, starkwargs=kwargs)
    print(r)
