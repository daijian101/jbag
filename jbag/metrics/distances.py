import numpy as np


def L1_distance(X, Y, average: bool = False):
    distance = np.sum(np.abs(X - Y))
    if average:
        distance = distance / X.shape[0]
    return distance
