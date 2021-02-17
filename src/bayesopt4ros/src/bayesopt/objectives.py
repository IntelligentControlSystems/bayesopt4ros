""" A collection of example objectives to dry-run BO """

import numpy as np


def quadratic(x, dim):
    if not isinstance(x, np.ndarray):
        x = np.array(x).squeeze()
    assert x.shape[0] == dim

    return np.sum(x ** 2)
